# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# source: https://github.com/openvinotoolkit/openvino.genai/blob/master/samples/python/text_generation/benchmark_genai.py

import argparse

import openvino_genai as ov_genai
from openvino import get_version

from llm_prompt import add_prompt_arguments, disable_genai_thinking


def main():
    parser = argparse.ArgumentParser(description="Help command")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to model and tokenizers base directory")
    parser.add_argument("-p", "--prompt", type=str, default=None, help="Prompt")
    parser.add_argument("-F", "--prompt_file", type=str, help="Read prompt from file")
    parser.add_argument("-N", "--num_warmup", type=int, default=1, help="Number of warmup iterations")
    parser.add_argument("-n", "--num_iter", type=int, default=3, help="Number of iterations")
    parser.add_argument("-M", "--max_new_tokens", type=int, default=20, help="Maximal number of new tokens")
    parser.add_argument("-d", "--device", type=str, default="CPU", help="Device")
    add_prompt_arguments(parser)

    args = parser.parse_args()

    if args.prompt is not None and args.prompt_file is not None:
        raise RuntimeError("Cannot specify both --prompt and --prompt_file options simultaneously!")
    else:
        if args.prompt_file is not None:
            with open(args.prompt_file, "r", encoding="utf-8") as f:
                prompt = [f.read()]
        else:
            prompt = ["The Sky is blue because"] if args.prompt is None else [args.prompt]
    if len(prompt) == 0:
        raise RuntimeError("Prompt is empty!")

    print(f"openvino runtime version: {get_version()}, genai version: {ov_genai.__version__}")

    # Perf metrics is stored in DecodedResults.
    # In order to get DecodedResults instead of a string input should be a list.
    models_path = args.model
    device = args.device
    num_warmup = args.num_warmup
    num_iter = args.num_iter

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens
    config.min_new_tokens = args.max_new_tokens
    config.apply_chat_template = not args.no_chat_template

    # Sequential requests do not need continuous batching, whose graph rewrites
    # are not compatible with every native HF stateful export.
    pipe = ov_genai.LLMPipeline(models_path, device)

    tokenizer = pipe.get_tokenizer()
    if config.apply_chat_template:
        if args.disable_thinking:
            try:
                disable_genai_thinking(tokenizer, models_path)
            except (ValueError, OSError) as error:
                parser.error(str(error))
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt[0]}], add_generation_prompt=True,
        )
        input_data = tokenizer.encode([formatted_prompt], add_special_tokens=False)
    else:
        input_data = tokenizer.encode(prompt)
    if args.no_chat_template:
        print("Chat template: not applied.")
    elif args.disable_thinking:
        print("Chat template: enable_thinking=False requested (model-dependent).")
    else:
        print("Chat template: model-default thinking behavior.")
    prompt_token_size = input_data.input_ids.get_shape()[1]
    print(f"Prompt token size: {prompt_token_size}")

    for _ in range(num_warmup):
        pipe.generate(prompt, config)

    res = pipe.generate(prompt, config)
    perf_metrics = res.perf_metrics
    for _ in range(num_iter - 1):
        res = pipe.generate(prompt, config)
        perf_metrics += res.perf_metrics

    print(f"Output token size: {res.perf_metrics.get_num_generated_tokens()}")
    print(f"Load time: {perf_metrics.get_load_time():.2f} ms")
    print(
        f"Generate time: {perf_metrics.get_generate_duration().mean:.2f} ± {perf_metrics.get_generate_duration().std:.2f} ms"
    )
    print(
        f"Tokenization time: {perf_metrics.get_tokenization_duration().mean:.2f} ± {perf_metrics.get_tokenization_duration().std:.2f} ms"
    )
    print(
        f"Detokenization time: {perf_metrics.get_detokenization_duration().mean:.2f} ± {perf_metrics.get_detokenization_duration().std:.2f} ms"
    )
    print(f"TTFT: {perf_metrics.get_ttft().mean:.2f} ± {perf_metrics.get_ttft().std:.2f} ms")
    print(f"TPOT: {perf_metrics.get_tpot().mean:.2f} ± {perf_metrics.get_tpot().std:.2f} ms/token")
    print(f"Throughput : {perf_metrics.get_throughput().mean:.2f} ± {perf_metrics.get_throughput().std:.2f} tokens/s")


if __name__ == "__main__":
    main()
