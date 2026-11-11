"""Export an INT4 LLM with native HF OpenVINO exporters and run greedy inference."""

import argparse
from contextlib import ExitStack
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.exporters import ExportedModel, OpenVINOConfig, OpenVINOExporter

from int4_config import (
    TEXT_CALIBRATION_DATASETS, apply_runtime_config, make_quantizer,
    prepare_text_calibration, resolve_int4_config,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", type=Path, default=Path("hf_exporters_llm_int4"))
    parser.add_argument("--prompt", default="What is OpenVINO?")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--calibration-data", type=Path, help="Preprocessed .npz inputs for recipes requiring calibration")
    parser.add_argument("--calibration-num-samples", type=int, help="Override the recipe's calibration sample count (default: 128)")
    args = parser.parse_args()
    if not args.prompt.strip() or args.max_new_tokens < 1:
        parser.error("The prompt must not be empty and --max-new-tokens must be positive")
    if args.calibration_num_samples is not None and args.calibration_num_samples < 1:
        parser.error("--calibration-num-samples must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("--output-dir must be empty to avoid mixing incompatible exports")
    try:
        recipe = resolve_int4_config(args.model_id)
        if args.calibration_num_samples is not None:
            recipe["num_samples"] = args.calibration_num_samples
        # Explicit .npz inputs override automatic sampling from the recipe's dataset.
        automatic_calibration = args.calibration_data is None and recipe.get("dataset") in TEXT_CALIBRATION_DATASETS
        quantizer = None if automatic_calibration else make_quantizer(recipe, args.calibration_data)
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
    with ExitStack() as stack, torch.no_grad():
        if automatic_calibration:
            # Capture real decoder feeds, not just token IDs. ExitStack keeps their
            # temporary files alive until quantization finishes during export.
            providers = stack.enter_context(prepare_text_calibration(
                model, tokenizer, recipe, exporter=exporter, export_configs=configs,
            ))
            configs["model"].quantizer = make_quantizer(recipe, calibration_samples=providers["model"])
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