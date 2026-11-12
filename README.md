# Native Hugging Face OpenVINO Export Experiments

Experimental scripts for comparing native Transformers OpenVINO exports with
separately exported Optimum-Intel models. These are not a drop-in replacement
for `optimum-cli export openvino`. Known differences below reflect the scripts
as of 2026-11-12, not a guarantee about every Optimum or Transformers release.

## Export Paths

| Script | Output and runtime | Limits |
| --- | --- | --- |
| [hf_export_models_int4.py](hf_export_models_int4.py) | Unified LLM/VLM HF `export.json` package; `ExportedModel` runtime | Requires upstream generation export/runtime support; not a GenAI package |
| [hf_export_llm.py](hf_export_llm.py) | Minimal native LLM export, reload, greedy inference | LLM only; HF runtime, not GenAI |
| [hf_export_int4.py](hf_export_int4.py) | Stateful LLM IR plus tokenizer IRs for GenAI | Compatibility depends on architecture and dependency versions; LLM only |
| [hf_export_multimodels_int4.py](hf_export_multimodels_int4.py) | Legacy custom VLM components and `export_manifest.json` | Only `gemma4` and `qwen3_5`; fixed example-image vision shapes; not GenAI |

The legacy component exporter rejects Gemma 3. Use the unified exporter instead,
subject to the installed Transformers version's architecture/export support:

```bash
python hf_export_models_int4.py --model-id google/gemma-3-12b-it --image dog.jpg
python hf_export_models_int4.py --models-file models.txt --image dog.jpg
```

AutoModel recognition is not proof of working OpenVINO export or generation.
[models.txt](models.txt) is an input list, not a validated support matrix.
Image-text workflows require an export image. Audio/video workflows are not
supported, and Gemma 4 audio config/weights are omitted.

## Dependencies and Validation

- Native generation/multimodal APIs require the Transformers exporter work
  associated with PRs #50249, #50251, #50252, and #50254. An arbitrary stable
  release may not expose these experimental APIs.
- Exporter and quantizer implementations patch backend internals and can warn
  about dependency drift. Record Transformers, PyTorch, OpenVINO, GenAI,
  OpenVINO Tokenizers, and NNCF versions with results.
- Local checks have used OpenVINO 2026.3.1, while Linux reports have also used
  OpenVINO/GenAI 2026.4.1. A local pass does not certify the remote combination.
- Validation is primarily small offline model configurations on CPU. Full-size
  checkpoints, long contexts, concurrency, GPU/NPU execution, and all dtypes
  are not comprehensively validated. No quality-equivalence claim is made.
- Gated model downloads require appropriate Hugging Face access/authentication.

## Differences From Optimum Compression

### Data-Free Policy

All four export entry points now use data-free weight compression for both
LLMs and VLMs:

- Remove recipe `dataset`, `num_samples`, `scale_estimation`, `gptq`, and
  `lora_correction` from the effective compression settings.
- Replace activation-dependent sensitivity metrics with
  `weight_quantization_error`.
- Retain AWQ only if the recipe requests `quant_method: "awq"` or `awq: true`.
  NNCF runs data-free AWQ; AWQ is not enabled for every model.
- Do not download calibration datasets, capture representative feeds, or pass
  calibration data to NNCF. Calibration CLI options have been removed.
- Still use the example prompt/image for tracing, not as calibration data.

This intentionally differs from Optimum's calibration-backed recipes.
Matching bit width/group size does not reproduce calibration-backed AWQ,
scale estimation, GPTQ, weights, or accuracy. Evaluate quality separately.
The original recipe snapshot remains unchanged; export metadata records the
normalized effective settings. Lower-level calibration utilities remain in
[int4_config.py](int4_config.py), but export entry points no longer call them.

### Defaults and Recipe Coverage

[int4_configs.json](int4_configs.json) snapshots 78 entries, including aliases,
from Optimum's `_DEFAULT_4BIT_WQ_CONFIGS` on 2026-11-04. The source commit is in
the JSON; recipes are not automatically synchronized with installed Optimum.
Lookup uses full model ID, local `_name_or_path`, or an unambiguous basename.
Ambiguous basenames fail rather than choosing an arbitrary recipe.

Unknown language models default to asymmetric INT4, group size 128, ratio 1.0,
and weight-quantization-error sensitivity. This is not a tuned recipe for every
architecture. Matrix dimensions can be incompatible with the group size:
group size 128 previously failed on a width-576 model. Use a compatible override.

For VLM flat recipes/generic defaults, the decoder gets the requested
compression, while embeddings and vision get symmetric INT8. This mirrors
Optimum's generic regular weight-only VLM policy. Explicit pipeline component
settings, exclusions, and `default_config` take precedence, followed by the
data-free normalization above. Group-size/ratio overrides affect INT4 only.
Missing components without an explicit pipeline default remain uncompressed.

Older exports could leave embeddings uncompressed and compress vision with
INT4. Re-export to apply the current defaults. The observed approximately 4x
Qwen3.5 embedding-size gap was consistent with FP32 versus INT8 embeddings,
not a decoder compression difference.

Only a subset of Optimum configuration semantics is implemented. Unsupported
settings/specialized sequential recipes fail validation. Activation/hybrid
quantization is not implemented. `dq_group_size` writes runtime dynamic
quantization metadata; it is separate from INT4 weight compression.
See [INT4_CONFIGS.md](INT4_CONFIGS.md) for lookup and component mappings.

## Graph and Architecture Differences

- Native HF and Optimum use different decomposition/export paths. Partitioning,
  operators, constant folding, compression targets, and output sizes can differ
  even when NNCF options match.
- The unified VLM script expects language, text-embedding, and vision components.
  Optimum can split vision position/merger components and export an MTP head.
  There is no separate MTP export here. Compare equivalent package scope rather
  than assuming individual BIN files correspond one-to-one.
- Dynamic shapes are requested, not guaranteed for every dimension or image
  pattern. The legacy component export fixes vision shapes to its processed
  example image; arbitrary resolutions/crops are not guaranteed there.
- Qwen3.5's native runtime may prefill one token at a time because its
  linear-attention prefill/decode paths differ. This affects TTFT relative to
  Optimum/GenAI; requesting dynamic shapes does not fix that runtime path.
- The unified exporter defaults to FP32 loading, with FP16/BF16 options. Match
  loading and execution precision explicitly when comparing with Optimum.
  Full-size FP32 loading can require substantial host memory.
- Stateful KV caches are required. The GenAI LLM script supplies a dynamic 2D
  attention mask and FP32 logits, but does not support every cache layout.
- A Phi-3 workaround clears sliding windows at least as large as the maximum
  position range. Model-specific workarounds are not general exporter fixes.

## Package and GenAI Limitations

Native HF packages load through
`transformers.exporters.ExportedModel.from_pretrained(...)`.
The native VLM benchmark requires `export.json` and rejects legacy component
packages. Renaming IR files does not turn HF/custom packages into GenAI packages.

The GenAI LLM exporter saves model IR, model/generation metadata, HF tokenizer
files, and tokenizer/detokenizer XML/BIN pairs. HF tokenizer JSON alone is not
enough for GenAI `Tokenizer.encode()`. Missing/unloadable IRs can produce
`m_ireq_queue_tokenizer`. Repair missing files without exporting weights again:

```bash
python hf_export_int4.py --model-id meta-llama/Llama-3.2-1B-Instruct \
  --output-dir Llama-3.2-1B-Instruct-int4-hf-export-genai --tokenizer-only
```

Use the exact benchmark directory. If IRs already exist, investigate conversion
and runtime versions; tokenizer-only conversion does not prove the remote cause.

The GenAI benchmark uses the default sequential `LLMPipeline`. Forced continuous
batching previously triggered incompatible graph rewrites: undeclared
attention-mask/beam inputs and cache batch-shape mismatches. Continuous batching,
speculative decoding, and production serving parity are not validated here.

The legacy component exporter must unwrap the HF export artifact's `.program`
before calling OpenVINO preprocessing/saving. This fixes the reported
`PrePostProcessor` invalid-model error; INT8 compression itself was not the cause.
Its vision input preparation also uses the current config-based API and retains
the returned input dictionary instead of assuming in-place mutation.

The unified/native minimal LLM exporters reject nonempty output directories.
Other export entry points may write into existing directories, so use fresh
directories to avoid mixing incompatible artifacts. Tokenizer-only repair
intentionally updates tokenizer artifacts in an existing package.

## Benchmark Comparability

- HF LLM TTFT/TPOT use host streamer timestamps, including Python/runtime
  overhead. HF generation timing excludes tokenization, detokenization, and
  template rendering. GenAI reports its own metric boundaries; equal labels
  do not prove equal measurement scope.
- Both current LLM benchmarks set the minimum generated length to the requested
  maximum. Native VLM benchmarking suppresses EOS for a fixed budget. Ordinary
  inference can stop at EOS. Check actual token counts and special tokens.
- The GenAI LLM benchmark does not explicitly override every generation option
  set by the HF benchmark. Match effective sampling/beam configs explicitly.
- Native VLM benchmarking preprocesses once, excludes warmups, starts a new
  cache per request, and reruns vision per generation. It excludes detokenization;
  TTFT includes vision, embeddings, native setup, and prefill.
- Device compile/cache precision and threading differ by backend. Native VLM
  defaults need not select GenAI's BF16 execution/KV-cache settings. Inspect
  profiling output and match compile settings instead of attributing every
  latency difference to INT4 export.
- Non-thinking mode requests `enable_thinking=False`; templates can ignore it.
  `--enable-thinking` restores model template defaults rather than universally
  forcing reasoning. `--no-chat-template` bypasses template thinking control.
  Compare actual tokenized prompts.
- GenAI releases may lack `get_chat_template()`. Saved template metadata is
  used as a fallback; missing templates can still prevent requested chat modes.
- Batch runners use sequential child processes and retain logs/CSV. They do
  not flush OS caches or standardize all backend defaults. Failed/missing models
  are recorded and the batch continues. The VLM list includes a text-only model.

LLM runner defaults distinguish Optimum/GenAI (`-int4-optimum`), native-export/
GenAI (`-int4-hf-export-genai`), and native HF (`-hf-export`) directories.
The unified exporter instead defaults to `-int4-hf-export`; override the runner
suffix when needed. Supply the LLM runner's `models_llm.txt` locally. VLM runners
compare Optimum and native HF exports. Bash runners require Bash, awk, and tee;
native PowerShell counterparts are also provided.

## Before Comparing Results

1. Record model revisions, dependency versions, dtype, export path, and effective
   saved compression recipes for both exports.
2. Match package scope, embedding/vision precision, omitted MTP/audio, and runtime
   execution/cache precision.
3. Match image preprocessing, tokenized prompts, templates/thinking mode,
   generation options, and actual output token counts.
4. Match device, host threads, compile settings, caching, warmups, and iterations;
   preserve logs and account for different timing boundaries.
5. Evaluate output quality separately. Passing small export/inference tests does
   not certify full-checkpoint accuracy or Optimum equivalence.