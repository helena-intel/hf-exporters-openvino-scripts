"""Export an INT4 LLM with native HF OpenVINO exporters and run greedy inference."""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.exporters import ExportedModel, OpenVINOConfig, OpenVINOExporter

from int4_config import apply_runtime_config, make_quantizer, resolve_int4_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", type=Path, default=Path("hf_exporters_llm_int4"))
    parser.add_argument("--prompt", default="What is OpenVINO?")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--calibration-data", type=Path, help="Preprocessed .npz inputs for recipes requiring calibration")
    args = parser.parse_args()
    if not args.prompt.strip() or args.max_new_tokens < 1:
        parser.error("The prompt must not be empty and --max-new-tokens must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("--output-dir must be empty to avoid mixing incompatible exports")
    try:
        recipe = resolve_int4_config(args.model_id)
        quantizer = make_quantizer(recipe, args.calibration_data)
    except (ValueError, OSError) as error:
        parser.error(str(error))

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    text = args.prompt
    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=not bool(tokenizer.chat_template))
    model = AutoModelForCausalLM.from_pretrained(args.model_id).eval()
    with torch.no_grad():
        artifacts = OpenVINOExporter().export_for_generation(
            model, inputs, {"model": OpenVINOConfig(dynamic=True, stateful=True, quantizer=quantizer)},
        )
    apply_runtime_config(artifacts["model"].program, recipe)
    artifacts.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    del model

    exported_model = ExportedModel.from_pretrained(args.output_dir, device="cpu")
    with torch.inference_mode():
        outputs = exported_model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    print(tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()