"""Benchmark native HF OpenVINO text generation without a chat template.

Generation and TTFT exclude tokenization and detokenization, reported separately.
Each call starts a new generation without passing a previous cache. TPOT measures
intervals between streamed tokens, including host overhead. Standard deviations
are population statistics across measured runs. EOS may stop generation early.
"""

import argparse
import copy
from pathlib import Path
from time import perf_counter

import numpy as np
import openvino as ov
import torch
import transformers

from inference_llm_hf import load_model


class TokenTimer:
    def __init__(self):
        self.prompt = True
        self.times = []

    def put(self, value):
        if self.prompt:
            self.prompt = False
            return
        if value.numel() != 1:
            raise ValueError("Benchmark requires one generated token at a time")
        self.times.append(perf_counter())

    def end(self):
        pass


def generate(model, tokenizer, prompt, *, max_new_tokens=20):
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    started = perf_counter()
    inputs = tokenizer(prompt, return_tensors="pt")
    tokenization_s = perf_counter() - started
    generation = copy.deepcopy(model.generation_config)
    generation.update(max_new_tokens=max_new_tokens, min_new_tokens=None, min_length=0,
                      do_sample=False, num_beams=1, num_return_sequences=1, use_cache=True,
                      return_dict_in_generate=False, output_scores=False, output_logits=False)
    timer = TokenTimer()
    started = perf_counter()
    with torch.inference_mode():
        outputs = model.generate(**inputs, generation_config=generation, streamer=timer)
    generate_s = perf_counter() - started
    input_tokens = inputs["input_ids"].shape[1]
    token_ids = outputs[0, input_tokens:].tolist()
    if not token_ids or len(timer.times) != len(token_ids):
        raise RuntimeError("Generation did not stream one timestamp per generated token")
    started_decode = perf_counter()
    tokenizer.decode(token_ids, skip_special_tokens=True)
    detokenization_s = perf_counter() - started_decode
    decode_steps = len(token_ids) - 1
    decode_s = timer.times[-1] - timer.times[0]
    return {
        "input_tokens": input_tokens, "output_tokens": len(token_ids),
        "timings": {
            "generate_s": generate_s, "tokenization_s": tokenization_s,
            "detokenization_s": detokenization_s, "ttft_s": timer.times[0] - started,
            "tpot_s": decode_s / decode_steps if decode_steps else None,
            "throughput_tokens_per_s": decode_steps / decode_s if decode_steps and decode_s else None,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("-p", "--prompt")
    source.add_argument("-F", "--prompt_file", "--prompt-file", type=Path)
    parser.add_argument("-N", "--num_warmup", "--num-warmup", type=int, default=1)
    parser.add_argument("-n", "--num_iter", "--num-iter", type=int, default=3)
    parser.add_argument("-M", "--max_new_tokens", "--max-new-tokens", type=int, default=20)
    parser.add_argument("-d", "--device", default="CPU")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.num_warmup < 0 or args.num_iter < 1 or args.max_new_tokens < 1 or args.torch_threads < 1:
        parser.error("Warmups must be nonnegative; iterations, max_new_tokens, and threads must be positive")
    if args.prompt_file is not None:
        try:
            prompt = args.prompt_file.read_text(encoding="utf-8")
        except OSError as error:
            parser.error(str(error))
    else:
        prompt = "The Sky is blue because" if args.prompt is None else args.prompt
    if not prompt.strip():
        parser.error("Prompt must not be empty")
    torch.set_num_threads(args.torch_threads)
    print(f"openvino runtime version: {ov.get_version()}, transformers version: {transformers.__version__}")
    print(f"PyTorch host threads: {args.torch_threads}")
    started = perf_counter()
    model, tokenizer = load_model(args.model, args.device)
    load_s = perf_counter() - started
    for _ in range(args.num_warmup):
        generate(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
    results = [generate(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
               for _ in range(args.num_iter)]
    counts = [result["output_tokens"] for result in results]
    print(f"Prompt token size: {results[-1]['input_tokens']}")
    print(f"Input token size: {results[-1]['input_tokens']}")
    print(f"Output token size: {counts[-1]}")
    if len(set(counts)) > 1:
        print(f"Output token sizes vary across runs: {min(counts)}-{max(counts)}")
    print(f"Load time: {load_s * 1000:.2f} ms")
    for label, key, scale, unit in (
        ("Generate time", "generate_s", 1000, "ms"),
        ("Tokenization time", "tokenization_s", 1000, "ms"),
        ("Detokenization time", "detokenization_s", 1000, "ms"),
        ("TTFT", "ttft_s", 1000, "ms"),
        ("TPOT", "tpot_s", 1000, "ms/token"),
        ("Throughput", "throughput_tokens_per_s", 1, "tokens/s"),
    ):
        values = [result["timings"][key] * scale for result in results
                  if result["timings"][key] is not None]
        if values:
            print(f"{label}: {np.mean(values):.2f} +/- {np.std(values):.2f} {unit}")
        else:
            print(f"{label}: N/A (fewer than two output tokens)")
    print("Generation timing excludes tokenization and detokenization; no chat template is applied.")
    print("TPOT and throughput exclude runs with fewer than two output tokens.")


if __name__ == "__main__":
    main()