#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
models_file="$script_dir/models.txt"
model_root=.
output_dir="benchmark_results/$(date +%Y%m%d_%H%M%S)_$$"
image=dog.jpg
prompt='what is this image?'
device=CPU
num_warmup=3
num_iter=10
max_new_tokens=256
hf_python=${HF_PYTHON:-${PYTHON:-python}}
genai_python=${GENAI_PYTHON:-${PYTHON:-python}}
hf_script="$script_dir/benchmark_vlm_hf_exporters.py"
genai_script="$script_dir/benchmark_vlm.py"
profile=false

usage() {
    printf '%s\n' \
        'Usage: bash run_vlm_benchmarks.sh [options]' \
        '  --models-file PATH       Model IDs, one per line (default: sibling models.txt)' \
        '  --model-root PATH        Parent of exported model directories (default: .)' \
        '  --output-dir PATH        New results directory (default: timestamped benchmark_results/)' \
        '  -i, --image PATH         Single input image (default: dog.jpg)' \
        '  -p, --prompt TEXT        Prompt (default: what is this image?)' \
        '  -d, --device DEVICE      OpenVINO device (default: CPU)' \
        '  -N, --num-warmup COUNT   Warmups per benchmark (default: 3)' \
        '  -n, --num-iter COUNT     Measured runs per benchmark (default: 10)' \
        '  -M, --max-new-tokens N   Output token budget (default: 256)' \
        '  --python PATH           Python executable for both backends (default: python)' \
        '  --hf-python PATH        Override HF Python executable' \
        '  --genai-python PATH     Override GenAI Python executable' \
        '  --hf-script PATH        Override native HF benchmark script' \
        '  --genai-script PATH     Override GenAI benchmark script' \
        '  --profile               Include native HF component profiling' \
        '  -h, --help              Show this help'
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 2
}

require_value() {
    [[ $# -ge 2 && -n $2 ]] || die "Missing value for $1"
}

while (( $# )); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --profile) profile=true; shift ;;
        --models-file|--model-root|--output-dir|-i|--image|-p|--prompt|-d|--device|-N|--num-warmup|-n|--num-iter|-M|--max-new-tokens|--python|--hf-python|--genai-python|--hf-script|--genai-script)
            require_value "$@"
            case "$1" in
                --models-file) models_file=$2 ;;
                --model-root) model_root=$2 ;;
                --output-dir) output_dir=$2 ;;
                -i|--image) image=$2 ;;
                -p|--prompt) prompt=$2 ;;
                -d|--device) device=$2 ;;
                -N|--num-warmup) num_warmup=$2 ;;
                -n|--num-iter) num_iter=$2 ;;
                -M|--max-new-tokens) max_new_tokens=$2 ;;
                --python) hf_python=$2; genai_python=$2 ;;
                --hf-python) hf_python=$2 ;;
                --genai-python) genai_python=$2 ;;
                --hf-script) hf_script=$2 ;;
                --genai-script) genai_script=$2 ;;
            esac
            shift 2
            ;;
        *) die "Unknown option: $1" ;;
    esac
done

[[ -f $models_file ]] || die "Model list not found: $models_file"
[[ -d $model_root ]] || die "Model root not found: $model_root"
[[ -f $image ]] || die "Image not found: $image"
[[ -n ${prompt//[[:space:]]/} ]] || die 'Prompt must not be empty'
[[ $num_warmup =~ ^(0|[1-9][0-9]*)$ ]] || die 'Warmups must be a nonnegative integer'
[[ $num_iter =~ ^[1-9][0-9]*$ ]] || die 'Iterations must be a positive integer'
[[ $max_new_tokens =~ ^[1-9][0-9]*$ ]] || die 'Output token budget must be a positive integer'
[[ -f $hf_script ]] || die "HF benchmark not found: $hf_script"
[[ -f $genai_script ]] || die "GenAI benchmark not found: $genai_script"
command -v "$hf_python" >/dev/null || die "HF Python executable not found: $hf_python"
command -v "$genai_python" >/dev/null || die "GenAI Python executable not found: $genai_python"
for dependency in awk tee; do
    command -v "$dependency" >/dev/null || die "Required command not found: $dependency"
done
[[ ! -e $output_dir ]] || die "Results directory already exists: $output_dir"
mkdir -p -- "$(dirname -- "$output_dir")"
mkdir -- "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd)
mkdir -- "$output_dir/logs"
summary="$output_dir/summary.csv"
printf '%s\n' 'model_id,backend,model_dir,status,exit_code,input_tokens,output_tokens,ttft_ms,ttft_std_ms,tpot_ms,tpot_std_ms,throughput_tokens_per_s,throughput_std_tokens_per_s,generation_ms,generation_std_ms,log_file' > "$summary"

csv_row() {
    local separator='' value
    for value in "$@"; do
        value=${value//\"/\"\"}
        printf '%s"%s"' "$separator" "$value"
        separator=,
    done
    printf '\n'
}

failures=0
model_index=0

run_benchmark() {
    local model_id=$1 backend=$2 suffix=$3 python=$4 benchmark_script=$5
    local base_name=${model_id##*/}
    local model_dir="$model_root/$base_name$suffix"
    local log_file="$output_dir/logs/${model_index}_${base_name}_${backend}.log"
    local status=ok exit_code=0
    local input_tokens='' output_tokens='' ttft='' ttft_std='' tpot='' tpot_std=''
    local throughput='' throughput_std='' generation='' generation_std=''
    local -a pipeline_status=(0 0)
    local -a command=("$python" -u "$benchmark_script" -m "$model_dir" -i "$image" -p "$prompt"
                     -d "$device" -N "$num_warmup" -n "$num_iter" -M "$max_new_tokens")
    if [[ $backend == hf_exporters && $profile == true ]]; then
        command+=(--profile)
    fi
    printf '\n[%s] %s\n' "$backend" "$model_id"
    {
        printf 'Model ID: %s\nStarted: %s\nCommand: ' "$model_id" "$(date -u +%FT%TZ)"
        printf '%q ' "${command[@]}"
        printf '\n\n'
    } > "$log_file"
    if [[ ! -d $model_dir ]]; then
        status=missing_model
        exit_code=''
        printf 'Model directory not found: %s\n' "$model_dir" | tee -a "$log_file"
    else
        "${command[@]}" </dev/null 2>&1 | tee -a "$log_file" || pipeline_status=("${PIPESTATUS[@]}")
        exit_code=${pipeline_status[0]}
        if (( pipeline_status[1] != 0 )); then
            die "Cannot write complete benchmark log: $log_file"
        fi
        if (( exit_code == 130 || exit_code == 143 )); then
            exit "$exit_code"
        fi
        if (( exit_code != 0 )); then
            status=failed
        fi
        IFS='|' read -r input_tokens output_tokens ttft ttft_std tpot tpot_std throughput throughput_std generation generation_std < <(
            awk '
                /^Input token size:/ { input_tokens = $4 }
                /^Output token size:/ { output_tokens = $4 }
                /^TTFT:/ { ttft = $2; ttft_std = $4 }
                /^TPOT:/ { tpot = $2; tpot_std = ($2 == "N/A" ? "" : $4) }
                /^Throughput:/ { throughput = $2; throughput_std = ($2 == "N/A" ? "" : $4) }
                /^Generate time:/ { generation = $3; generation_std = $5 }
                END {
                    printf "%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n", input_tokens, output_tokens,
                        ttft, ttft_std, tpot, tpot_std, throughput, throughput_std, generation, generation_std
                }
            ' "$log_file"
        )
        if [[ $status == ok ]]; then
            local metric
            for metric in "$ttft" "$ttft_std" "$generation" "$generation_std"; do
                if [[ ! $metric =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]]; then
                    status=missing_metrics
                fi
            done
            for metric in "$tpot" "$throughput"; do
                if [[ ! $metric =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ && $metric != N/A ]]; then
                    status=missing_metrics
                fi
            done
            [[ $input_tokens =~ ^[0-9]+$ && $output_tokens =~ ^[0-9]+$ ]] || status=missing_metrics
        fi
    fi
    printf '\nFinished: %s\nStatus: %s\nExit code: %s\n' "$(date -u +%FT%TZ)" "$status" "$exit_code" >> "$log_file"
    csv_row "$model_id" "$backend" "$model_dir" "$status" "$exit_code" "$input_tokens" "$output_tokens" \
        "$ttft" "$ttft_std" "$tpot" "$tpot_std" "$throughput" "$throughput_std" "$generation" "$generation_std" \
        "$log_file" >> "$summary"
    if [[ $status != ok ]]; then
        failures=$((failures + 1))
        printf 'Run recorded as %s: %s\n' "$status" "$log_file" >&2
    fi
}

while IFS= read -r model_id || [[ -n $model_id ]]; do
    model_id=${model_id#$'\xef\xbb\xbf'}
    model_id=${model_id%$'\r'}
    model_id=${model_id%%#*}
    model_id="${model_id#"${model_id%%[![:space:]]*}"}"
    model_id="${model_id%"${model_id##*[![:space:]]}"}"
    [[ -n $model_id ]] || continue
    model_index=$((model_index + 1))
    run_benchmark "$model_id" genai '-int4-optimum' "$genai_python" "$genai_script"
    run_benchmark "$model_id" hf_exporters '-int4-hf-export' "$hf_python" "$hf_script"
done < "$models_file"

printf '\nSummary: %s\nLogs: %s/logs\nUnsuccessful runs: %s\n' "$summary" "$output_dir" "$failures"
(( model_index > 0 )) || die 'Model list contains no model IDs'
(( failures == 0 ))