"""Export an INT4 LLM with native HF OpenVINO exporters and run greedy inference."""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.exporters import ExportedModel, OpenVINOConfig, OpenVINOExporter

from int4_config import (
    apply_runtime_config, data_free_config, make_quantizer, resolve_int4_config,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", type=Path,
                        help="Output directory (default: <model-basename>-int4-hf-export)")
    parser.add_argument("--prompt", default="What is OpenVINO?")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = Path(f"{Path(args.model_id).name}-int4-hf-export")
    if not args.prompt.strip() or args.max_new_tokens < 1:
        parser.error("The prompt must not be empty and --max-new-tokens must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("--output-dir must be empty to avoid mixing incompatible exports")
    try:
        recipe = data_free_config(resolve_int4_config(args.model_id))
        quantizer = make_quantizer(recipe)
    except (ValueError, OSError) as error:
        parser.error(str(error))

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    text = args.prompt
    if tokenizer.chat_template:
        # Templates that support this option skip reasoning; others may ignore it.
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
    # Chat templates already include special tokens; do not insert them twice.
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=not bool(tokenizer.chat_template))
    model = AutoModelForCausalLM.from_pretrained(args.model_id).eval()
    exporter = OpenVINOExporter()
    # Allow varying input shapes and keep the KV cache in OpenVINO state variables.
    configs = {"model": OpenVINOConfig(dynamic=True, stateful=True, quantizer=quantizer)}
    with torch.no_grad():
        artifacts = exporter.export_for_generation(model, inputs, configs)
    # Store optional dq_group_size as DYNAMIC_QUANTIZATION_GROUP_SIZE metadata
    # for activation quantization at inference, separate from INT4 weight compression.
    # This is a no-op when the recipe has no dq_group_size.
    apply_runtime_config(artifacts["model"].program, recipe)
    artifacts.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    del model

    # Reload the saved native HF artifacts and use their generate() wrapper, not GenAI.
    exported_model = ExportedModel.from_pretrained(args.output_dir, device="cpu")
    with torch.inference_mode():
        outputs = exported_model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    print(tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()