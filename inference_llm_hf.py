"""Greedy text inference with a native HF OpenVINO generation export."""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.exporters import ExportArtifacts
from transformers.exporters.configs import ExportFormat

from llm_prompt import add_prompt_arguments, prepare_prompt


def load_model(model_dir, device="CPU"):
    artifacts = ExportArtifacts.from_pretrained(model_dir, local_files_only=True)
    if artifacts.export_format != ExportFormat.OPENVINO or not artifacts.generation or artifacts.multimodal:
        raise ValueError("Expected a native OpenVINO text-only generation export (export.json)")
    normalized = device.upper()
    if normalized == "AUTO":
        runtime_device = None
    elif normalized == "GPU" or normalized.startswith("GPU."):
        runtime_device = "xpu" + (":" + normalized[4:] if normalized.startswith("GPU.") else "")
    else:
        runtime_device = device.lower()
    model = artifacts.runtime(device=runtime_device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    return model, tokenizer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--prompt", default="What is OpenVINO?")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    add_prompt_arguments(parser)
    args = parser.parse_args(argv)
    if not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    model, tokenizer = load_model(args.model_dir, args.device)
    try:
        text, templated = prepare_prompt(
            tokenizer, args.prompt, use_chat_template=not args.no_chat_template,
            disable_thinking=args.disable_thinking,
        )
    except ValueError as error:
        parser.error(str(error))
    inputs = tokenizer(text, return_tensors="pt", **({"add_special_tokens": False} if templated else {}))
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 num_beams=1, num_return_sequences=1, return_dict_in_generate=False)
    print(tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()