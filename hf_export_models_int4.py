import argparse
import json
from pathlib import Path
import subprocess
import sys


def export_model(model_id, output_dir, *, image=None, prompt="Describe this image.",
                 group_size=None, ratio=None, int4_config=None, dtype="float32"):
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"Output directory must be empty to avoid mixing incompatible exports: {output_dir}")
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    from transformers.exporters import OpenVINOConfig, OpenVINOExporter
    from int4_config import (
        apply_runtime_config, data_free_config, make_quantizer, resolve_int4_config,
    )

    config = AutoConfig.from_pretrained(model_id)
    multimodal = type(config) in AutoModelForImageTextToText._model_mapping
    if not multimodal and type(config) not in AutoModelForCausalLM._model_mapping:
        raise ValueError(f"{config.model_type} is not a supported causal LLM or image-text-to-text model")
    if multimodal and image is None:
        raise ValueError(f"{model_id} requires --image with a representative export image")
    if multimodal:
        components = {"language_model": "decoder", "text_embeddings_model": "embeddings",
                      "vision_embeddings_model": "vision"}
    else:
        components = {"model": "decoder"}
    recipes = {}
    export_configs = {}
    for name, component in components.items():
        recipe = data_free_config(resolve_int4_config(
            model_id, int4_config, component=component, group_size=group_size, ratio=ratio,
        ))
        recipes[name] = recipe
        export_configs[name] = OpenVINOConfig(
            dynamic=True, stateful=True, quantizer=make_quantizer(recipe),
        )
    print("Compression recipes: " + json.dumps(recipes), flush=True)
    if config.model_type == "phi3":
        window = config.sliding_window
        if window is not None and window >= config.max_position_embeddings:
            config.sliding_window = None
    if multimodal:
        from PIL import Image

        if config.model_type == "gemma4":
            config.audio_config = None
        processor = AutoProcessor.from_pretrained(model_id)
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        with Image.open(image) as source:
            inputs = processor(text=text, images=source.convert("RGB"), return_tensors="pt")
        model_class = AutoModelForImageTextToText
    else:
        processor = AutoTokenizer.from_pretrained(model_id)
        inputs = processor(prompt, return_tensors="pt")
        model_class = AutoModelForCausalLM
    model = model_class.from_pretrained(
        model_id, config=config, dtype=getattr(torch, dtype), attn_implementation="sdpa",
    ).eval()
    exporter = OpenVINOExporter()
    with torch.no_grad():
        artifacts = exporter.export_for_generation(model, inputs, export_configs)
    if set(artifacts) != set(components):
        raise RuntimeError(f"Unexpected export components: {list(artifacts)}; expected {list(components)}")
    decoder = "language_model" if multimodal else "model"
    if not artifacts[decoder].program.get_variables():
        raise RuntimeError("The generation decoder must contain OpenVINO cache state")
    for name, component in artifacts.items():
        apply_runtime_config(component.program, recipes[name])
    artifacts.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    (output_dir / "compression_config.json").write_text(
        json.dumps({"model_id": str(model_id), "transformers": transformers.__version__,
                    "task": "image-text-to-text" if multimodal else "text-generation",
                    "components": recipes, "calibration": None}, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Saved HF generation export to {output_dir}; load with ExportedModel.from_pretrained().", flush=True)


def main(argv=None):
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Export causal LLMs and image-text models with native HF OpenVINO exporters.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--models-file", type=Path)
    source.add_argument("--model-id")
    parser.add_argument("--output-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, help="Output directory for --model-id only")
    parser.add_argument("--image", type=Path, help="Representative export image, required for image-text models")
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--group-size", type=int, choices=[-1, 32, 64, 128])
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--int4-config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.ratio is not None and not 0 <= args.ratio <= 1:
        parser.error("--ratio must be between 0 and 1")
    if args.output_dir is not None and args.model_id is None:
        parser.error("--output-dir requires --model-id")
    if args.image is not None and not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    if not args.prompt.strip():
        parser.error("--prompt must not be empty")
    try:
        model_ids = [args.model_id] if args.model_id else [
            line.strip()
            for line in (args.models_file or script_dir / "models.txt").read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    except OSError as error:
        parser.error(str(error))
    if not model_ids:
        parser.error("The model list is empty")

    exports = []
    output_names = set()
    for model_id in model_ids:
        model_name = model_id.rsplit("/", 1)[-1]
        if model_name in ("", ".", "..") or "\\" in model_name:
            parser.error(f"Invalid model ID: {model_id}")
        output_name = f"{model_name}-int4-hf-export"
        if output_name.casefold() in output_names:
            parser.error(f"Multiple models would write to {output_name}")
        output_names.add(output_name.casefold())
        exports.append((model_id, args.output_dir or args.output_root / output_name))

    if args.model_id and not args.dry_run:
        try:
            export_model(
                args.model_id, exports[0][1], image=args.image, prompt=args.prompt, dtype=args.dtype,
                group_size=args.group_size, ratio=args.ratio, int4_config=args.int4_config,
            )
        except Exception as error:
            print(f"Export failed for {args.model_id}: {type(error).__name__}: {error}", file=sys.stderr)
            return 1
        return 0

    failures = []
    for model_id, output_dir in exports:
        command = [
            sys.executable,
            str(script_dir / "hf_export_models_int4.py"),
            "--model-id", model_id,
            "--output-dir", str(output_dir),
            "--prompt", args.prompt,
            "--dtype", args.dtype,
        ]
        if args.image is not None:
            command.extend(["--image", str(args.image)])
        for option, value in (("--group-size", args.group_size), ("--ratio", args.ratio),
                              ("--int4-config", args.int4_config)):
            if value is not None:
                command.extend([option, str(value)])
        print(f"\n{model_id} -> {output_dir}", flush=True)
        if args.dry_run:
            print(subprocess.list2cmdline(command), flush=True)
            continue
        result = subprocess.run(command, check=False)
        if result.returncode:
            failures.append(model_id)
            print(f"Export failed for {model_id} (exit code {result.returncode})", file=sys.stderr)

    if failures:
        print("\nFailed models: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())