#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
models_file="$script_dir/models_llm.txt"
model_root=models
output_dir="benchmark_results/llm_$(date +%Y%m%d_%H%M%S)_$$"
prompt='Why is the sky blue?'
device=CPU
num_warmup=1
num_iter=3
max_new_tokens=512
hf_python=${HF_PYTHON:-${PYTHON:-python}}
genai_python=${GENAI_PYTHON:-${PYTHON:-python}}
hf_script="$script_dir/benchmark_llm_hf.py"
genai_script="$script_dir/benchmark_llm_genai.py"
optimum_suffix=-int4-optimum
hf_genai_suffix=-int4-hf-export-genai
hf_suffix=-int4-hf-export
prompt_mode=--disable-thinking

usage() {
    printf '%s\n' \
        'Usage: bash run_llm_benchmarks.sh [options]' \
        '  --models-file PATH       Model IDs, one per line (default: sibling models_llm.txt)' \
        '  --model-root PATH        Parent of exported model directories (default: .)' \
        '  --output-dir PATH        New results directory (default: timestamped benchmark_results/llm_...)' \
        '  --optimum-suffix TEXT    Optimum directory suffix (default: -int4-optimum)' \
        '  --hf-genai-suffix TEXT   HF/GenAI directory suffix (default: -int4-hf-export-genai)' \
        '  --hf-suffix TEXT         Native HF directory suffix (default: -hf-export)' \
        '  -p, --prompt TEXT        Prompt (default: Why is the sky blue?)' \
        '  -d, --device DEVICE      OpenVINO device (default: CPU)' \
        '  -N, --num-warmup COUNT   Warmups per benchmark (default: 3)' \
        '  -n, --num-iter COUNT     Measured runs per benchmark (default: 3)' \
        '  -M, --max-new-tokens N   Output token budget (default: 1024)' \
        '  --python PATH           Python executable for all variants (default: python)' \
        '  --hf-python PATH        Override native HF Python executable' \
        '  --genai-python PATH     Override Python for both GenAI variants' \
        '  --hf-script PATH        Override native HF benchmark script' \
        '  --genai-script PATH     Override benchmark script for both GenAI variants' \
        '  --disable-thinking      Request non-thinking chat (default; model-dependent)' \
        '  --enable-thinking       Use model-default thinking behavior' \
        '  --no-chat-template      Use raw prompts without a thinking-mode request' \
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
        --disable-thinking|--enable-thinking|--no-chat-template) prompt_mode=$1; shift ;;
        --models-file|--model-root|--output-dir|--optimum-suffix|--hf-genai-suffix|--hf-suffix|-p|--prompt|-d|--device|-N|--num-warmup|-n|--num-iter|-M|--max-new-tokens|--python|--hf-python|--genai-python|--hf-script|--genai-script)
            require_value "$@"
            case "$1" in
                --models-file) models_file=$2 ;;
                --model-root) model_root=$2 ;;
                --output-dir) output_dir=$2 ;;
                --optimum-suffix) optimum_suffix=$2 ;;
                --hf-genai-suffix) hf_genai_suffix=$2 ;;
                --hf-suffix) hf_suffix=$2 ;;
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
    local -a command=("$python" -u "$benchmark_script" -m "$model_dir" -p "$prompt"
                     -d "$device" -N "$num_warmup" -n "$num_iter" -M "$max_new_tokens" "$prompt_mode")
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
                /^(Prompt|Input) token size:/ { input_tokens = $4 }
                /^Output token size:/ { output_tokens = $4 }
                /^TTFT:/ { ttft = $2; ttft_std = $4 }
                /^TPOT:/ { tpot = $2; tpot_std = ($2 == "N/A" ? "" : $4) }
                /^Throughput[[:space:]]*:/ {
                    sub(/^Throughput[[:space:]]*:[[:space:]]*/, "")
                    throughput = $1; throughput_std = ($1 == "N/A" ? "" : $3)
                }
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
    run_benchmark "$model_id" optimum "$optimum_suffix" "$genai_python" "$genai_script"
    run_benchmark "$model_id" hf-exporters-genai "$hf_genai_suffix" "$genai_python" "$genai_script"
    run_benchmark "$model_id" hf-exporters "$hf_suffix" "$hf_python" "$hf_script"
done < "$models_file"

printf '\nSummary: %s\nLogs: %s/logs\nUnsuccessful runs: %s\n' "$summary" "$output_dir" "$failures"
(( model_index > 0 )) || die 'Model list contains no model IDs'
(( failures == 0 ))
