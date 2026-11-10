"""Benchmark one image with the native HF OpenVINO generation runtime.

Image preprocessing, tokenization, and detokenization are excluded. Every run
repeats vision/text embedding inference and starts with an empty decoder cache.
TTFT includes embedding preparation, prefill, and first-token selection. TPOT is
the mean interval between subsequent selected tokens, including Python overhead.
Reported standard deviations are population statistics across measured runs.
"""

import argparse
from collections import Counter
import copy
import inspect
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import openvino as ov
from PIL import Image
import torch


class SuppressEosLogitsProcessor:
    def __init__(self, eos_token_id):
        self.eos_token_ids = torch.as_tensor(eos_token_id, dtype=torch.long).reshape(-1)

    def __call__(self, input_ids, scores):
        return scores.index_fill(-1, self.eos_token_ids.to(scores.device), -torch.inf)


class TokenTimer:
    def __init__(self):
        self.prompt = True
        self.times = []

    def put(self, value):
        if self.prompt:
            self.prompt = False
            return
        if value.numel() != 1:
            raise ValueError("Benchmark timing requires one generated token at a time")
        self.times.append(perf_counter())

    def end(self):
        pass


class TimedRunner:
    def __init__(self, runner, name, benchmark_runner):
        self.runner = runner
        self.name = name
        self.benchmark_runner = benchmark_runner

    def __getattr__(self, name):
        return getattr(self.runner, name)

    def __call__(self, *args, **kwargs):
        owner = self.benchmark_runner
        if owner.timer is None or (owner.timer.times and not owner.profile):
            return self.runner(*args, **kwargs)
        started = perf_counter()
        result = self.runner(*args, **kwargs)
        phase = "decode" if owner.timer.times else "prefill"
        key = (phase, self.name)
        owner.component_times[key] = owner.component_times.get(key, 0.0) + perf_counter() - started
        return result


class NativeOpenVINOVLM:
    def __init__(self, model_dir, device="CPU", *, processor=None, compile_config=None, torch_threads=1):
        from transformers import AutoProcessor
        from transformers.exporters import ExportArtifacts
        from transformers.exporters.configs import ExportFormat

        self.model_dir = Path(model_dir)
        if not (self.model_dir / "export.json").is_file():
            raise ValueError("Native export.json is required; use hf_export_models_int4.py, not the legacy exporter")
        artifacts = ExportArtifacts.from_pretrained(self.model_dir, local_files_only=True)
        if artifacts.export_format != ExportFormat.OPENVINO or not artifacts.generation or not artifacts.multimodal:
            raise ValueError("Expected a native OpenVINO multimodal generation export")
        normalized = device.upper()
        if normalized == "AUTO":
            runtime_device = None
        elif normalized == "CPU":
            runtime_device = "cpu"
        elif normalized == "GPU" or normalized.startswith("GPU."):
            runtime_device = "xpu" + (":" + normalized[4:] if normalized.startswith("GPU.") else "")
        else:
            runtime_device = device.lower()
        torch.set_num_threads(torch_threads)
        self.model = artifacts.runtime(device=runtime_device, config=compile_config or {})
        self.processor = processor if processor is not None else AutoProcessor.from_pretrained(
            self.model_dir, local_files_only=True,
        )
        self.timer = None
        self.profile = False
        self.component_times = {}
        self.model.runner = TimedRunner(self.model.runner, "decoder", self)
        self.model.embedder.runner = self.model.runner
        self.model.embedder.text_runner = TimedRunner(self.model.embedder.text_runner, "embeddings", self)
        self.model.embedder.modality_runners = [
            (spec, TimedRunner(runner, spec.component, self))
            for spec, runner in self.model.embedder.modality_runners
        ]
        self.decoder = self.model.runner.request
        self.embeddings = self.model.embedder.text_runner.request
        self.vision = next(
            runner.request for spec, runner in self.model.embedder.modality_runners
            if spec.component == "vision_embeddings_model"
        )
        self.embedding_prepare_s = 0.0
        original_embed = self.model.embedder.embed

        def timed_embed(**kwargs):
            started = perf_counter()
            result = original_embed(**kwargs)
            if self.timer is not None and not self.timer.times:
                self.embedding_prepare_s += perf_counter() - started
            return result

        self.model.embedder.embed = timed_embed

    def preprocess(self, image, prompt="Describe this image."):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        with Image.open(image) as source:
            return self.processor(text=text, images=source.convert("RGB"), return_tensors="pt")

    def generate_processed(self, processed, *, max_new_tokens=20, decode_text=False, profile=False):
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if processed["input_ids"].shape[0] != 1:
            raise ValueError("Benchmark supports one image and one prompt at a time")
        generation = copy.deepcopy(self.model.generation_config)
        generation.update(max_new_tokens=max_new_tokens, min_new_tokens=None, min_length=0,
                          do_sample=False, num_beams=1, num_return_sequences=1,
                          use_cache=True, return_dict_in_generate=False, output_scores=False, output_logits=False)
        processors = [] if generation.eos_token_id is None else [SuppressEosLogitsProcessor(generation.eos_token_id)]
        self.timer = TokenTimer()
        self.profile = profile
        self.component_times = {}
        self.embedding_prepare_s = 0.0
        started = perf_counter()
        try:
            with torch.inference_mode():
                outputs = self.model.generate(**processed, generation_config=generation,
                                              logits_processor=processors, streamer=self.timer)
            finished = perf_counter()
            token_times = self.timer.times
        finally:
            self.timer = None
        input_tokens = processed["input_ids"].shape[1]
        token_ids = outputs[0, input_tokens:].tolist()
        if len(token_times) != len(token_ids) or not token_ids:
            raise RuntimeError("Native generation did not stream one timestamp per generated token")
        decode_steps = len(token_ids) - 1
        decode_seconds = token_times[-1] - token_times[0]
        timings = {
            "generate_s": finished - started,
            "embeddings_prepare_s": self.embedding_prepare_s,
            "vision_s": self.component_times.get(("prefill", "vision_embeddings_model"), 0.0),
            "text_embedding_s": self.component_times.get(("prefill", "embeddings"), 0.0),
            "ttft_s": token_times[0] - started,
            "tpot_s": decode_seconds / decode_steps if decode_steps else None,
            "throughput_tokens_per_s": decode_steps / decode_seconds if decode_steps and decode_seconds else None,
            "decode_steps": decode_steps,
        }
        if profile:
            decoder = self.component_times.get(("decode", "decoder"), 0.0)
            embeddings = self.component_times.get(("decode", "embeddings"), 0.0)
            other = sum(duration for (phase, name), duration in self.component_times.items()
                        if phase == "decode" and name not in ("decoder", "embeddings"))
            timings.update(decode_decoder_call_s=decoder, decode_embedding_call_s=embeddings,
                           decode_other_call_s=other,
                           decode_host_s=max(0.0, decode_seconds - decoder - embeddings - other))
        result = {"input_tokens": input_tokens, "token_ids": token_ids, "timings": timings}
        if decode_text:
            result["text"] = self.processor.tokenizer.decode(token_ids, skip_special_tokens=True)
        return result


def benchmark(runner, processed, *, num_warmup=1, num_iter=2, max_new_tokens=20, profile=False):
    if num_warmup < 0 or num_iter < 1 or max_new_tokens < 1:
        raise ValueError("Warmup count must be nonnegative; iterations and max_new_tokens must be positive")
    options = {"max_new_tokens": max_new_tokens, "decode_text": False}
    if profile:
        options["profile"] = True
    for _ in range(num_warmup):
        runner.generate_processed(processed, **options)
    return [runner.generate_processed(processed, **options)
            for _ in range(num_iter)]


def print_report(results, load_seconds):
    last = results[-1]
    print("Number of images: 1")
    print(f"Input token size: {last['input_tokens']}")
    counts = [len(result["token_ids"]) for result in results]
    print(f"Output token size: {counts[-1]}")
    if len(set(counts)) > 1:
        print(f"Output token sizes vary across runs: {min(counts)}-{max(counts)}")
    print(f"Load time: {load_seconds * 1000:.2f} ms")
    metrics = (
        ("Generate time", "generate_s", 1000, "ms"),
        ("Embeddings preparation time", "embeddings_prepare_s", 1000, "ms"),
        ("  Vision encoding time", "vision_s", 1000, "ms"),
        ("  Text embedding time", "text_embedding_s", 1000, "ms"),
        ("TTFT", "ttft_s", 1000, "ms"),
        ("TPOT", "tpot_s", 1000, "ms/token"),
        ("Throughput", "throughput_tokens_per_s", 1, "tokens/s"),
    )
    for label, key, scale, unit in metrics:
        values = [result["timings"][key] * scale for result in results
                  if result["timings"][key] is not None]
        if values:
            print(f"{label}: {np.mean(values):.2f} +/- {np.std(values):.2f} {unit}")
        else:
            print(f"{label}: N/A (fewer than two output tokens)")
    if "decode_decoder_request_s" in last["timings"]:
        for label, key in (("Decoder request", "decode_decoder_request_s"),
                           ("Embedding request", "decode_embedding_request_s"),
                           ("Host / I/O / token selection", "decode_host_s")):
            values = [result["timings"][key] * 1000 / result["timings"]["decode_steps"]
                      for result in results if result["timings"]["decode_steps"]]
            if values:
                print(f"  Decode {label}: {np.mean(values):.2f} +/- {np.std(values):.2f} ms/token")
        print("Request times include synchronous OpenVINO dispatch, not just device kernels.")
    if "decode_decoder_call_s" in last["timings"]:
        for label, key in (("Decoder call", "decode_decoder_call_s"),
                           ("Embedding call", "decode_embedding_call_s"),
                           ("Other component calls", "decode_other_call_s"),
                           ("Host / token selection", "decode_host_s")):
            values = [result["timings"][key] * 1000 / result["timings"]["decode_steps"]
                      for result in results if result["timings"]["decode_steps"]]
            if values:
                print(f"  Decode {label}: {np.mean(values):.2f} +/- {np.std(values):.2f} ms/token")
        print("Component call times include native input/output handling and synchronous OpenVINO dispatch.")
    print("Timing excludes image preprocessing, tokenization, and detokenization.")
    print("Statistics are across measured runs; TTFT includes embeddings and prefill.")
    if any(count < 2 for count in counts) and any(count >= 2 for count in counts):
        print("TPOT and throughput exclude runs with fewer than two output tokens.")


def print_runtime_info(runner):
    for name in ("vision", "embeddings", "decoder"):
        compiled = getattr(runner, name).get_compiled_model()
        properties = {}
        for key in ("EXECUTION_DEVICES", "PERFORMANCE_HINT", "INFERENCE_NUM_THREADS", "NUM_STREAMS",
                    "INFERENCE_PRECISION_HINT", "KV_CACHE_PRECISION", "DYNAMIC_QUANTIZATION_GROUP_SIZE"):
            try:
                properties[key] = compiled.get_property(key)
            except RuntimeError:
                continue
        print(f"{name} compile properties: " + json.dumps(properties, default=str))
        if name == "decoder":
            layers = Counter()
            for node in compiled.get_runtime_model().get_ops():
                information = node.get_rt_info()
                if "layerType" in information:
                    layers[str(information["layerType"].value)] += 1
            print("Decoder execution layer types: " + json.dumps(dict(sorted(layers.items()))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model", type=Path, required=True)
    parser.add_argument("-i", "--image", type=Path, default=Path("image.jpg"))
    parser.add_argument("-p", "--prompt", default="What is on the image?")
    parser.add_argument("-d", "--device", default="CPU")
    parser.add_argument("-N", "--num_warmup", "--num-warmup", type=int, default=1)
    parser.add_argument("-n", "--num_iter", "--num-iter", type=int, default=2)
    parser.add_argument("-M", "--max_new_tokens", "--max-new-tokens", type=int, default=20,
                        help="Fixed number of new tokens; suppresses EOS until the limit (default: 20)")
    parser.add_argument("--profile", action="store_true", help="Break decode time into decoder, embeddings, and host/I/O")
    parser.add_argument("--compile-config", type=Path, help="JSON object of OpenVINO compile properties, applied to all components")
    parser.add_argument("--cache-dir", type=Path, help="OpenVINO compiled-model cache directory (affects load time)")
    parser.add_argument("--torch-threads", type=int, default=1, help="PyTorch host-operation threads (default: 1)")
    args = parser.parse_args(argv)
    if not args.model.is_dir():
        parser.error(f"Model directory does not exist: {args.model}")
    if not args.image.is_file():
        parser.error(f"Image file does not exist: {args.image}")
    if not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if args.num_warmup < 0 or args.num_iter < 1 or args.max_new_tokens < 1:
        parser.error("Warmup count must be nonnegative; iterations and max_new_tokens must be positive")
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    compile_config = {}
    if args.compile_config is not None:
        try:
            compile_config = json.loads(args.compile_config.read_text(encoding="utf-8-sig"))
        except (ValueError, OSError) as error:
            parser.error(str(error))
        if not isinstance(compile_config, dict):
            parser.error("--compile-config must contain a JSON object")
    if args.cache_dir is not None:
        compile_config["CACHE_DIR"] = str(args.cache_dir)
    print(f"openvino runtime version: {ov.get_version()}, backend: native HF OpenVINO generation runtime")
    print(f"PyTorch host threads: {args.torch_threads}")
    if compile_config:
        print("Compile properties: " + json.dumps(compile_config))
    started = perf_counter()
    try:
        runner = NativeOpenVINOVLM(args.model, args.device, compile_config=compile_config,
                                  torch_threads=args.torch_threads)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    load_seconds = perf_counter() - started
    if args.profile:
        print(f"Inference runtime: {inspect.getfile(type(runner.model))}")
        print_runtime_info(runner)
    processed = runner.preprocess(args.image, args.prompt)
    print(f"Warmup iterations: {args.num_warmup}, measured iterations: {args.num_iter}")
    results = benchmark(
        runner, processed, num_warmup=args.num_warmup,
        num_iter=args.num_iter, max_new_tokens=args.max_new_tokens, profile=args.profile,
    )
    print_report(results, load_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())