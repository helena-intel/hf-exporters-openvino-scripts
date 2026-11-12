# INT4 Recipes

## Unified Native Generation Export

`hf_export_models_int4.py` now exports both causal LLMs and image-text-to-text
models using native `OpenVINOExporter.export_for_generation()`. It requires the
multimodal and generation export/runtime APIs from Transformers PRs #50249,
#50251, #50252, and #50254. It does not import Optimum.

```powershell
python hf_export_models_int4.py --models-file models.txt --image example.jpg
python hf_export_models_int4.py --model-id Qwen/Qwen3-0.6B --group-size 64
python hf_export_models_int4.py --model-id Qwen/Qwen3.5-2B --image example.jpg --output-dir qwen-vlm-native
```

Without `--model-id` or `--models-file`, the script reads the sibling `models.txt`.
Mixed batches run each model in an isolated child process and continue on failure,
returning a nonzero exit status if any model fails. Output directories default to
`<repository-basename>-int4-hf-export` under `--output-root` and must be empty.
Use a new output root when preserving older exports. `--dry-run` prints commands
without downloading models or importing Transformers.

Model config determines whether to use `AutoModelForCausalLM` and a tokenizer, or
`AutoModelForImageTextToText` and a processor with an image chat prompt. `--image`
is required only for image-text models; audio/video inputs are not supported.
`--prompt` controls export example text. `--dtype` selects loading precision:
`float32` (default), `float16`, or `bfloat16`. Compression settings remain controlled
by the local recipes and explicit overrides. Gemma4 audio weights are omitted.

Exports contain HF's `export.json`, stateful decoder IR, any embedding/vision IRs,
model/generation configs, tokenizer or processor, and `compression_config.json`
with the effective per-component recipes. Load them using
`transformers.exporters.ExportedModel.from_pretrained(path, device="cpu")` and call
`generate()` with tokenizer/processor outputs. For one-image inference:

```bash
python inference_vlm.py gemma-4-E4B-it-int4-hf-export dog.jpg --prompt "what is this image?" --device CPU
```

The inference CLI detects `export.json` and uses the native HF generation runtime.
Older custom component exports still use the legacy loader. Normal inference stops
at EOS, unlike the benchmark's fixed-length generation. Native exports are not a
promise of OpenVINO GenAI compatibility.

Dynamic export is requested, but upstream determines whether a model can use
multi-token prefill. In particular, its Qwen3.5 path may feed the prompt one token
at a time because its linear-attention kernels differ between prefill and decode.
Model recognition does not guarantee upstream export/runtime support: Gemma4 and
Qwen3.5 full-checkpoint validation remains necessary.

Calibration follows the INT4 recipe's dataset by default or uses the existing `.npz` contract
below. No export example prompt/image is silently substituted for required data.

## Native VLM Benchmark

Benchmark the native package from `hf_export_models_int4.py` with the same CLI:

```powershell
python benchmark_vlm_hf_exporters.py -m gemma-4-E2B-it-int4-hf-export -i dog.jpg -p "what is this image?" -d CPU -N 3 -n 10 -M 20 --profile
```

The benchmark requires native `export.json` artifacts and uses HF's cached
generation runtime, not `inference_vlm.py`. Legacy component packages are rejected.
It preprocesses/tokenizes once, excludes warmups, starts generation with fresh cache
state, and runs vision again once per request. There is no detokenization in measured
runs. TTFT includes native generation setup, vision/embedding preparation and prefill;
TPOT measures intervals between streamed output tokens. `-M` sets `max_new_tokens`
and suppresses EOS with indexed logit updates, preventing early stopping without
HF's vocabulary-wide minimum-length mask on every token. Benchmark runs generate
the requested number of new tokens. The model's saved generation config is unchanged.

`--profile` prints the actual runtime implementation, component compile properties,
decoder execution-layer counts, and decoder/embedding/host decode timing. Component
call times include native input/output handling and synchronous OpenVINO dispatch,
not only device kernels. Qwen3.5's token-at-a-time prefill remains an upstream runtime
constraint; the benchmark does not override its cache, mask or prefill behavior.

`--torch-threads` defaults to 1 for the host-side PyTorch operations, leaving OpenVINO's
thread count under its compile settings. `--compile-config` accepts a JSON object
applied to every component; `--cache-dir` enables OpenVINO's compiled-model cache.
Native runtime defaults can preserve the export's precision rather than selecting
BF16 automatically: compare the printed `INFERENCE_PRECISION_HINT` and
`KV_CACHE_PRECISION` as well as thread counts when comparing to another runtime.
For example, to explicitly request BF16 inference and cache on CPU, a compile config
can contain `{"INFERENCE_PRECISION_HINT": "bf16", "KV_CACHE_PRECISION": "bf16"}`.
This changes execution precision; it is not silently imposed by the benchmark.

### Batch Comparison

Run both backends sequentially in separate processes, using repository basenames
from `models.txt` and sibling `<basename>-int4-optimum` / `<basename>-int4-hf-export`
directories:

```bash
bash run_vlm_benchmarks.sh --image dog.jpg --model-root . -N 3 -n 10 -M 1024 --profile
```

Results go to a new timestamped directory under `benchmark_results/`, or the new
directory selected by `--output-dir`. Each run preserves stdout and stderr plus its
command in `logs/`. `summary.csv` contains mean and standard deviation for TTFT,
TPOT, throughput, and total generation time, along with actual input/output token
counts, backend, model path, exit status, and log path. Units are in column names.
Existing result directories are never overwritten. Failed or missing models are
recorded and the remaining runs continue; the script exits nonzero if any run is
unsuccessful. Blank lines, comments, UTF-8 BOM, and CRLF model lists are supported.

Use `--python /path/to/venv/bin/python`, or separate `--hf-python` and
`--genai-python` executables. `--models-file`, `--hf-script`, and `--genai-script`
override their defaults; `--help` lists all options. Each benchmark process exits
before the next starts, releasing process-owned model memory without dropping OS
file caches. The runner does not change either backend's preprocessing, compile
defaults, or EOS policy: compare the actual token counts in the CSV. The supplied
list includes text-only Phi-4-mini; VLM benchmark errors for unsupported models are
recorded rather than silently skipped. Bash, awk, and tee are required.

## Shared Recipes

The HF export scripts automatically read `int4_configs.json` beside them. Its
`models` table is a snapshot of Optimum-intel's `_DEFAULT_4BIT_WQ_CONFIGS`, including
aliases. The pinned source URL and snapshot date are stored in the JSON file.
Optimum-intel is not imported at export time.

Edit a model's entry to customize its recipe. For example:

```json
"microsoft/Phi-4-mini-instruct": {
  "bits": 4,
  "sym": false,
  "group_size": 64,
  "ratio": 1.0,
  "quant_method": "awq"
}
```

Selection order is: explicit `--group-size` / `--ratio` overrides, then the model's
recipe, then the generic defaults for unknown models. Model lookup uses the full
model ID, a local checkpoint's `_name_or_path`, or an unambiguous repository basename.
If the default config file is absent, the old generic recipe is retained.
An explicitly requested missing file is an error.

Use a different recipe file with:

```powershell
python hf_export_int4.py --model-id microsoft/Phi-4-mini-instruct --int4-config my_recipes.json
python hf_export_multimodels_int4.py --models-file models.txt --image example.jpg --int4-config my_recipes.json
```

The unified batch launcher forwards these options too. Unspecified CLI values no
longer override model defaults. Explicit group-size and ratio overrides apply to
INT4 components, not per-channel INT8 components.

## Multimodal Components

When expanding a flat recipe or generic defaults for a VLM, the decoder keeps
the requested compression settings, while text embeddings and vision components
use symmetric INT8 weight compression. INT4-only group-size/ratio overrides and
decoder calibration settings do not propagate to these INT8 components.
Explicit pipeline recipes instead retain their component settings and
`default_config`; missing components without a default and explicit `null`
components remain uncompressed. This policy does not change text-only LLM exports.

Optimum-style `quantization_configs` entries are mapped as follows:

| Exported component | Recipe key |
| --- | --- |
| Decoder | `lm_model` |
| Text and per-layer embeddings | `text_embeddings_model` |
| Projected vision encoder | `vision_embeddings_model` |

`default_config` applies to components without a specific entry. If neither is
present, that component is not compressed. Snapshot audio entries are unused by
the image-text-to-text exporter. Unsupported sequential/specialized recipes fail
explicitly rather than approximating their settings.

## Calibration

Dataset-free AWQ, including Phi-4-mini's default, uses NNCF's data-free algorithm.
HF still converts the model; the quantizer does not substitute the export sample
as calibration data.

Some defaults require real calibration data. For example, Gemma 4 E2B/E4B specify
TextVQA, AWQ, and scale estimation for the decoder, with INT8 symmetric compression
for vision and embeddings. The unified exporter automatically prepares TextVQA
when the component recipe specifies it, unless `--calibration-data` supplies feeds.
Data-free recipes remain data-free. Unsupported named datasets require manual feeds.

The unified native exporter can prepare calibration directly from HF Datasets:

```bash
python hf_export_models_int4.py --model-id google/gemma-4-E4B-it --image dog.jpg
python hf_export_models_int4.py --models-file models.txt --image dog.jpg --calibration-num-samples 32 --calibration-seed 42
```

This requires `datasets` and Pillow in the export environment (`python -m pip
install datasets pillow`). TextVQA uses `lmms-lab/textvqa`, validation split,
streaming and seeded shuffle, as in Optimum. Downloads use the usual Hugging Face
cache and authentication settings. Other named datasets are not yet supported.
Text-only models in mixed batches skip TextVQA and keep their normal recipes.

Each real image/question is processed with the model's chat template and processor,
then passed through native generation calibration capture. The resulting component
feeds include multimodal embeddings, positions and explicit cache inputs before
OpenVINO state folding. INT8 weight compression remains data-free. Feeds are
spooled to temporary disk storage and replayed for NNCF rather than retaining all
calibration caches in RAM; allow sufficient space in the system temporary directory.
Streaming iterators and temporary storage are closed on success or failure.

`--calibration-num-samples` limits image/question rows, defaulting to the component
recipe's `num_samples` or 32, matching Optimum's visual-language calibration path
(its general weight-compression fallback is 128). An explicit CLI count overrides
the recipe. Effective recipes and the shuffle seed are saved in
`compression_config.json`. Multi-token decoder graphs use one prefill feed per
image, matching Optimum's prompt-based collection; single-token-only graphs use
one decode feed per image instead. NNCF's `subset_size` uses the same sample limit,
so 32 items yield 32 decoder feeds, rather than a mix of 16 decode/prefill pairs.
This is not a promise of bit-identical Optimum calibration.
`--image` is still required for tracing and does not become a calibration sample.

`--calibration-data` accepts a directory of `.npz` files. Each file must contain
representative, preprocessed inputs for the graph being compressed, with array keys
matching its OpenVINO input names, including any explicit cache inputs. Loading uses
`allow_pickle=False`. The text exporter reads files directly in that directory;
the multimodal exporter reads `decoder/`, `vision/`, and `embeddings/` subdirectories
for components requiring data. INT8 weight compression is data-free.
This option is mutually exclusive with `--calibration-dataset`. Use graph input
names from native exports; older custom-export feeds may be incompatible.

Alternatively, edit the local recipe to remove `dataset` and disable
`scale_estimation` / `gptq` if data-free compression is intentional. That is a
different recipe from Optimum's calibration-backed default, not an equivalent export.

The legacy text exporter saves `compression_config.json`; the legacy multimodal
exporter records each effective recipe in `export_manifest.json`. The unified native
exporter saves recipes per component in `compression_config.json`. Matching recipe parameters does not
guarantee identical graphs, compressed weights, accuracy, or performance across exporters.