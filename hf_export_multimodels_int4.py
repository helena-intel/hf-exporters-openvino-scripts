"""HF component export and plain OpenVINO inference for multimodal models."""

# The current implementation is limited to image-text-to-text.

from pathlib import Path


def gemma_decoder_sample(model):
    import copy
    import torch
    from transformers import DynamicCache
    from transformers.exporters.utils import get_leaf_tensors

    cache_config = copy.deepcopy(model.config)
    cache_config.layer_types = ["full_attention"] * cache_config.num_hidden_layers
    cache = DynamicCache(config=cache_config)
    warm_ids = torch.full((2, 4), 2, dtype=torch.long)
    with torch.no_grad():
        model(input_ids=warm_ids, past_key_values=cache, use_cache=True)
    ids = torch.full((2, 3), 3, dtype=torch.long)
    query = torch.arange(4, 7).view(1, 1, 3, 1)
    keys = torch.arange(7).view(1, 1, 1, 7)
    allowed = keys <= query
    minimum = torch.finfo(model.dtype).min
    full_mask = torch.where(allowed, 0.0, minimum).expand(2, 1, 3, 7).to(model.dtype)
    sliding = allowed & (keys > query - model.config.sliding_window)
    sliding_mask = torch.where(sliding, 0.0, minimum).expand(2, 1, 3, 7).to(model.dtype)
    inputs = {
        "inputs_embeds": model.get_input_embeddings()(ids).detach(),
        "position_ids": query.reshape(1, 3).expand(2, 3),
        "attention_mask": {"full_attention": full_mask, "sliding_attention": sliding_mask},
        "past_key_values": cache,
        "use_cache": True,
        "return_dict": True,
    }
    automatic = torch.export.Dim.AUTO
    shapes = {
        "inputs_embeds": {0: automatic, 1: automatic},
        "position_ids": {0: automatic, 1: automatic},
        "attention_mask": {
            name: {0: automatic, 2: automatic, 3: automatic}
            for name in inputs["attention_mask"]
        },
        "past_key_values": [
            {0: automatic, 2: automatic} for _ in get_leaf_tensors(cache).values()
        ],
    }
    if model.config.hidden_size_per_layer_input:
        inputs["per_layer_inputs"] = model.model.get_per_layer_inputs(ids, None).detach()
        shapes["per_layer_inputs"] = {0: automatic, 1: automatic}
    return inputs, shapes


def qwen_decoder_sample(model):
    import torch
    from transformers.exporters.utils import decompose_for_generation, get_leaf_tensors

    components = decompose_for_generation(
        model, {"input_ids": torch.tensor([[2, 3, 4, 5]] * 2)}, multi_token_decode=True,
    )
    _, captured = components["decode"]
    ids = captured["input_ids"]
    batch, length = ids.shape
    past = captured["past_key_values"].get_seq_length()
    query = torch.arange(past, past + length).view(1, 1, length, 1)
    keys = torch.arange(past + length).view(1, 1, 1, -1)
    inputs = {
        "inputs_embeds": model.get_input_embeddings()(ids).detach(),
        "position_ids": captured["position_ids"].unsqueeze(0).expand(3, batch, length),
        "attention_mask": {
            "full_attention": torch.where(keys <= query, 0.0, torch.finfo(model.dtype).min)
            .expand(batch, 1, length, past + length).to(model.dtype),
            "linear_attention": torch.ones((batch, length), dtype=model.dtype),
        },
        "past_key_values": captured["past_key_values"],
        "use_cache": True,
        "return_dict": True,
    }
    automatic = torch.export.Dim.AUTO
    shapes = {
        "inputs_embeds": {0: automatic, 1: automatic},
        "position_ids": {1: automatic, 2: automatic},
        "attention_mask": {
            "full_attention": {0: automatic, 2: automatic, 3: automatic},
            "linear_attention": {0: automatic, 1: automatic},
        },
        "past_key_values": [
            {0: automatic, 2: automatic} if name.endswith((".keys", ".values"))
            else {0: automatic}
            for name in get_leaf_tensors(captured["past_key_values"])
        ],
    }
    return inputs, shapes


def cached_decoder(model, family):
    import torch

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language = model
            self.config = model.config

        def forward(self, inputs_embeds, position_ids, attention_mask, past_key_values,
                    per_layer_inputs=None, mm_token_type_ids=None):
            length = inputs_embeds.shape[1]
            total = attention_mask.shape[1]
            query = torch.arange(total - length, total, device=inputs_embeds.device).view(1, -1, 1)
            keys = torch.arange(total, device=inputs_embeds.device).view(1, 1, -1)
            allowed = keys <= query
            if mm_token_type_ids is not None:
                from transformers.models.gemma4.modeling_gemma4 import get_block_sequence_ids_for_mask

                blocks = get_block_sequence_ids_for_mask(mm_token_type_ids, device=inputs_embeds.device)
                matching = blocks[:, -length:, None] == blocks[:, None, :]
                allowed = allowed | (matching & (blocks[:, None, :] >= 0))
            allowed = allowed & attention_mask[:, None, :].bool()
            minimum = torch.finfo(inputs_embeds.dtype).min
            masks = {"full_attention": torch.where(allowed, 0.0, minimum)[:, None].to(inputs_embeds.dtype)}
            if family == "gemma4":
                window = self.config.sliding_window
                sliding_allowed = allowed & (keys > query - window)
                sliding = torch.where(sliding_allowed, 0.0, minimum)[:, None]
                masks["sliding_attention"] = sliding[..., -(window - 1 + length):].to(inputs_embeds.dtype)
            else:
                masks["linear_attention"] = attention_mask[:, -length:].to(inputs_embeds.dtype)
            extra = {"per_layer_inputs": per_layer_inputs} if per_layer_inputs is not None else {}
            return self.language(
                inputs_embeds=inputs_embeds, position_ids=position_ids,
                attention_mask=masks, past_key_values=past_key_values,
                use_cache=True, return_dict=True, logits_to_keep=1, **extra,
            )

    inputs, shapes = (gemma_decoder_sample(model) if family == "gemma4"
                      else qwen_decoder_sample(model))
    full_mask = inputs["attention_mask"]["full_attention"]
    inputs["attention_mask"] = torch.ones(
        (full_mask.shape[0], full_mask.shape[-1]), dtype=torch.long,
    )
    shapes["attention_mask"] = {0: torch.export.Dim.AUTO, 1: torch.export.Dim.AUTO}
    if family == "gemma4" and model.config.use_bidirectional_attention == "vision":
        inputs["mm_token_type_ids"] = torch.zeros_like(inputs["attention_mask"])
        inputs["mm_token_type_ids"][:, 1:3] = 1
        shapes["mm_token_type_ids"] = shapes["attention_mask"].copy()
    return Decoder().eval(), inputs, shapes


def export_component(model, inputs, path, dynamic_shapes, *, group_size=128,
                     ratio=1.0, stateful=False, quantize=True, sliding_window=None,
                     layer_types=(), compression_config=None):
    import openvino as ov
    from transformers.exporters import (
        OpenVINOConfig,
        OpenVINOExporter,
    )
    from int4_config import FALLBACK_CONFIG, apply_runtime_config, data_free_config, make_quantizer

    quantizer = None
    if quantize:
        compression_config = compression_config or {**FALLBACK_CONFIG, "group_size": group_size, "ratio": ratio}
        compression_config = data_free_config(compression_config)
        quantizer = make_quantizer(compression_config)
    config = OpenVINOConfig(
        dynamic_shapes=dynamic_shapes, stateful=stateful, quantizer=quantizer,
    )
    graph = OpenVINOExporter().export(model, inputs, config).program
    if stateful and not graph.get_variables():
        raise RuntimeError("HF export did not produce decoder cache state variables")
    if sliding_window is not None:
        import re
        import numpy as np
        from openvino import opset13 as ops

        trimmed = 0
        for sink in graph.get_sinks():
            if sink.get_type_name() != "Assign":
                continue
            match = re.search(r"layers\.(\d+)\.(keys|values)", sink.get_variable_id())
            if match and layer_types[int(match.group(1))] == "sliding_attention":
                cached = ops.slice(
                    sink.input_value(0), np.array([1 - sliding_window], np.int64),
                    np.array([np.iinfo(np.int64).max], np.int64),
                    np.array([1], np.int64), np.array([2], np.int64),
                )
                sink.input(0).replace_source_output(cached.output(0))
                trimmed += 1
        if "sliding_attention" in layer_types and not trimmed:
            raise RuntimeError("Could not identify Gemma sliding KV state for window trimming")
        graph.validate_nodes_and_infer_types()
    preprocessing = ov.preprocess.PrePostProcessor(graph)
    for output_index, output in enumerate(graph.outputs):
        if output.get_element_type().is_real():
            preprocessing.output(output_index).tensor().set_element_type(ov.Type.f32)
    graph = preprocessing.build()
    if quantize:
        apply_runtime_config(graph, compression_config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(graph, path)
    return graph


def model_adapters(model):
    import torch
    from transformers import AutoModelForCausalLM

    family = model.config.model_type
    if family not in ("gemma4", "qwen3_5"):
        raise ValueError(f"Unsupported model type {family!r}; supported: gemma4, qwen3_5")
    base = model.model
    with torch.device("meta"):
        decoder = AutoModelForCausalLM.from_config(model.config.get_text_config())
    decoder.model = base.language_model
    decoder.lm_head = model.lm_head
    decoder.generation_config = model.generation_config
    decoder.eval()

    class Embeddings(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language = base.language_model
            self.config = self.language.config

        def forward(self, input_ids):
            result = {"inputs_embeds": self.language.get_input_embeddings()(input_ids)}
            if family == "gemma4" and self.config.hidden_size_per_layer_input:
                result["per_layer_inputs"] = self.language.get_per_layer_inputs(input_ids, None)
            return result

    class GemmaVision(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = base.vision_tower
            self.projection = base.embed_vision
            self.config = self.vision.config

        def forward(self, pixel_values, image_position_ids):
            encoded = self.vision(pixel_values, pixel_position_ids=image_position_ids)
            return {"image_features": self.projection(inputs_embeds=encoded.last_hidden_state)}

    class QwenVision(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = base.visual
            self.config = self.vision.config

        def forward(self, hidden_states, grid_thw, **kwargs):
            encoded = self.vision(hidden_states, grid_thw=grid_thw, **kwargs)
            return {"image_features": encoded.pooler_output}

    adapters = {"decoder": decoder, "embeddings": Embeddings().eval()}
    adapters["vision"] = (GemmaVision() if family == "gemma4" else QwenVision()).eval()
    return adapters


def export_model(model_id, output_dir, image_path, *, group_size=None, ratio=None,
                 int4_config=None):
    import importlib.metadata
    import json

    from PIL import Image
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    from transformers.exporters.utils import get_leaf_tensors, precompute_export_inputs
    from int4_config import make_quantizer, data_free_config, resolve_int4_config

    recipes = {
        name: data_free_config(resolve_int4_config(
            model_id, int4_config, component=name, group_size=group_size, ratio=ratio,
        ))
        for name in ("decoder", "vision", "embeddings")
    }
    for name, recipe in recipes.items():
        make_quantizer(recipe)
    print("Compression recipes: " + json.dumps(recipes), flush=True)

    config = AutoConfig.from_pretrained(model_id)
    family = config.model_type
    if family not in ("gemma4", "qwen3_5"):
        raise ValueError(f"Unsupported model type {family!r}; supported: gemma4, qwen3_5")
    if family == "gemma4":
        config.audio_config = None
    processor = AutoProcessor.from_pretrained(model_id)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "Describe this image."},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    processed = processor(text=prompt, images=image, return_tensors="pt")
    if family == "gemma4":
        vision_inputs = {name: processed[name] for name in ("pixel_values", "image_position_ids")}
    else:
        vision_inputs = {
            "hidden_states": processed["pixel_values"],
            "grid_thw": processed["image_grid_thw"],
        }
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, config=config, attn_implementation="sdpa",
    ).eval()
    adapters = model_adapters(model)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_id": model_id,
        "model_type": family,
        "task": "image-text-to-text",
        "exporter": "transformers.exporters.OpenVINOExporter",
        "versions": {name: importlib.metadata.version(name)
                     for name in ("transformers", "openvino", "nncf")},
        "compression": recipes,
        "vision_shapes": "fixed to the processed example image",
        "components": {},
    }

    def save_component(name, component, inputs, shapes, **options):
        filename = f"openvino_{name}.xml"
        print(f"Exporting {model_id}: {name} -> {output_dir / filename}", flush=True)
        graph = export_component(
            component, inputs, output_dir / filename, shapes,
            compression_config=recipes[name],
            quantize=recipes[name] is not None, **options,
        )
        manifest["components"][name] = {
            "file": filename,
            "inputs": [{"name": port.get_any_name(), "shape": str(port.get_partial_shape()),
                        "dtype": str(port.get_element_type())} for port in graph.inputs],
            "outputs": [port.get_any_name() for port in graph.outputs],
            "stateful": bool(graph.get_variables()),
            "quantized": recipes[name] is not None,
            "compression": recipes[name],
        }

    with torch.no_grad():
        ids = torch.tensor([[2, 3, 4], [3, 4, 5]], dtype=torch.long)
        automatic = torch.export.Dim.AUTO
        save_component(
            "embeddings", adapters["embeddings"], {"input_ids": ids},
            {"input_ids": {0: automatic, 1: automatic}},
        )
        vision_inputs = precompute_export_inputs(adapters["vision"].config, vision_inputs)
        manifest["vision_input_shapes"] = {
            name: list(value.shape) for name, value in get_leaf_tensors(vision_inputs).items()
        }
        if family == "qwen3_5":
            manifest["image_grid_thw"] = processed["image_grid_thw"].tolist()
        save_component(
            "vision", adapters["vision"], vision_inputs,
            {name: None for name in vision_inputs},
        )
        decoder, inputs, shapes = cached_decoder(adapters["decoder"], family)
        options = {"stateful": True}
        if family == "gemma4":
            options.update(
                sliding_window=decoder.config.sliding_window,
                layer_types=decoder.config.layer_types,
            )
        save_component("decoder", decoder, inputs, shapes, **options)
    config.save_pretrained(output_dir)
    model.generation_config.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    (output_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )


def main(argv=None):
    import argparse
    import subprocess
    import sys

    script = Path(__file__).resolve()
    parser = argparse.ArgumentParser(
        description="HF-only INT4 export of Gemma 4 and Qwen3.5 image-text-to-text models.",
        epilog="Vision shapes are fixed to --image; decoder lengths are dynamic. "
               "These are plain OpenVINO components, not a GenAI model package.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--model-id", "--model-ids", nargs="+", dest="model_ids")
    selection.add_argument("--models-file", type=Path)
    parser.add_argument("--image", type=Path, required=True, help="Example image for vision export shapes")
    parser.add_argument("--output-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, help="Explicit output directory for a single model")
    parser.add_argument("--group-size", type=int, choices=[-1, 32, 64, 128])
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--int4-config", type=Path, help="Override the local per-model INT4 recipes")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.ratio is not None and not 0 <= args.ratio <= 1:
        parser.error("--ratio must be between 0 and 1")
    model_ids = args.model_ids
    if model_ids is None:
        models_file = args.models_file or script.with_name("models.txt")
        try:
            model_ids = [line.strip() for line in models_file.read_text(encoding="utf-8-sig").splitlines()
                         if line.strip() and not line.lstrip().startswith("#")]
        except (OSError, UnicodeError) as error:
            parser.error(str(error))
    if not model_ids:
        parser.error("The model list is empty")
    if args.output_dir is not None and len(model_ids) != 1:
        parser.error("--output-dir requires exactly one model; use --output-root for a list")
    exports = []
    names = set()
    for model_id in model_ids:
        basename = model_id.rsplit("/", 1)[-1]
        if basename in ("", ".", "..") or "\\" in basename:
            parser.error(f"Invalid model ID: {model_id}")
        name = f"{basename}-int4-hf-export"
        if name.casefold() in names:
            parser.error(f"Multiple models would write to {name}")
        names.add(name.casefold())
        exports.append((model_id, args.output_dir if args.output_dir is not None else args.output_root / name))
    if not args.dry_run and not args.image.is_file():
        parser.error(f"Image file does not exist: {args.image}")
    if args.output_dir is not None and not args.dry_run:
        export_model(model_ids[0], args.output_dir, args.image,
                     group_size=args.group_size, ratio=args.ratio,
                     int4_config=args.int4_config)
        return 0
    failures = []
    for model_id, output_dir in exports:
        command = [
            sys.executable, str(script), "--model-id", model_id,
            "--output-dir", str(output_dir), "--image", str(args.image),
        ]
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