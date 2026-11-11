import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys


def export_model(model_id, output_dir, *, image=None, prompt="Describe this image.",
                 group_size=None, ratio=None, int4_config=None, calibration_data=None,
                 dtype="float32", calibration_dataset=None, calibration_num_samples=None,
                 calibration_seed=42):
    if calibration_data is not None and calibration_dataset is not None:
        raise ValueError("Choose either --calibration-data or --calibration-dataset")
    if calibration_dataset not in (None, "textvqa", "wikitext2", "gsm8k"):
        raise ValueError("Supported calibration datasets: textvqa, wikitext2, gsm8k")
    if calibration_num_samples is not None and (
            not isinstance(calibration_num_samples, int) or isinstance(calibration_num_samples, bool)
            or calibration_num_samples <= 0):
        raise ValueError("--calibration-num-samples must be a positive integer")
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"Output directory must be empty to avoid mixing incompatible exports: {output_dir}")
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    from transformers.exporters import OpenVINOConfig, OpenVINOExporter
    from int4_config import (
        DEFAULT_TEXT_CALIBRATION_SAMPLES, DEFAULT_VLM_CALIBRATION_SAMPLES, TEXT_CALIBRATION_DATASETS,
        apply_runtime_config, make_quantizer, prepare_text_calibration,
        prepare_textvqa_calibration, resolve_int4_config,
    )

    config = AutoConfig.from_pretrained(model_id)
    multimodal = type(config) in AutoModelForImageTextToText._model_mapping
    if not multimodal and type(config) not in AutoModelForCausalLM._model_mapping:
        raise ValueError(f"{config.model_type} is not a supported causal LLM or image-text-to-text model")
    if multimodal and image is None:
        raise ValueError(f"{model_id} requires --image with a representative export image")
    if calibration_dataset == "textvqa" and not multimodal:
        print("TextVQA calibration is only used for image-text models; skipping it for this text model.", flush=True)
        calibration_dataset = None
    if multimodal:
        components = {"language_model": "decoder", "text_embeddings_model": "embeddings",
                      "vision_embeddings_model": "vision"}
    else:
        components = {"model": "decoder"}
    recipes = {}
    export_configs = {}
    dataset_recipes = {}
    for name, component in components.items():
        recipe = resolve_int4_config(
            model_id, int4_config, component=component, group_size=group_size, ratio=ratio,
        )
        calibration = Path(calibration_data) if calibration_data is not None else None
        if calibration is not None and multimodal:
            calibration = calibration / component
        dataset = calibration_dataset
        if dataset is None and calibration_data is None and recipe is not None:
            dataset = recipe.get("dataset")
        dataset_backed = bool(dataset) and recipe is not None and recipe.get("bits", 4) == 4
        if dataset_backed and recipe is not None:
            if not ((dataset == "textvqa" and multimodal) or (dataset in TEXT_CALIBRATION_DATASETS and not multimodal)):
                raise ValueError(
                    f"Automatic calibration for dataset={dataset!r} is not supported for this model. "
                    "Use TextVQA for image-text models or Wikitext2/GSM8K for LLMs, "
                    "or supply --calibration-data with preprocessed .npz inputs."
                )
                default_samples = DEFAULT_VLM_CALIBRATION_SAMPLES if multimodal else DEFAULT_TEXT_CALIBRATION_SAMPLES
            recipe = {**recipe, "dataset": dataset,
                      "num_samples": calibration_num_samples or recipe.get("num_samples", default_samples)}
            dataset_recipes[name] = recipe
        recipes[name] = recipe
        export_configs[name] = OpenVINOConfig(
            dynamic=True, stateful=True, quantizer=None if dataset_backed else make_quantizer(recipe, calibration),
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
    with ExitStack() as stack, torch.no_grad():
        if dataset_recipes:
            if multimodal:
                calibration = prepare_textvqa_calibration(
                    model, processor, dataset_recipes, exporter=exporter,
                    export_configs=export_configs, seed=calibration_seed,
                )
            else:
                calibration = prepare_text_calibration(
                    model, processor, dataset_recipes["model"], exporter=exporter,
                    export_configs=export_configs, seed=calibration_seed,
                )
            providers = stack.enter_context(calibration)
            for name, provider in providers.items():
                export_configs[name].quantizer = make_quantizer(recipes[name], calibration_samples=provider)
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
                    "components": recipes,
                    "calibration": {"dataset": next(iter(dataset_recipes.values()))["dataset"], "seed": calibration_seed}
                    if dataset_recipes else None}, indent=2) + "\n", encoding="utf-8",
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
    calibration = parser.add_mutually_exclusive_group()
    calibration.add_argument("--calibration-data", type=Path)
    calibration.add_argument("--calibration-dataset", choices=["textvqa", "wikitext2", "gsm8k"],
                             help="Override the recipe's calibration dataset (default: follow INT4 recipe)")
    parser.add_argument("--calibration-num-samples", type=int, help="Calibration samples (default: recipe, 32 TextVQA, or 128 text)")
    parser.add_argument("--calibration-seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.ratio is not None and not 0 <= args.ratio <= 1:
        parser.error("--ratio must be between 0 and 1")
    if args.calibration_num_samples is not None and args.calibration_num_samples <= 0:
        parser.error("--calibration-num-samples must be positive")
    if (args.calibration_num_samples is not None or args.calibration_seed is not None) and args.calibration_data is not None:
        parser.error("--calibration-num-samples and --calibration-seed cannot be used with --calibration-data")
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
                calibration_data=args.calibration_data,
                calibration_dataset=args.calibration_dataset, calibration_num_samples=args.calibration_num_samples,
                calibration_seed=args.calibration_seed if args.calibration_seed is not None else 42,
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
                              ("--int4-config", args.int4_config), ("--calibration-data", args.calibration_data),
                              ("--calibration-dataset", args.calibration_dataset),
                              ("--calibration-num-samples", args.calibration_num_samples),
                              ("--calibration-seed", args.calibration_seed)):
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