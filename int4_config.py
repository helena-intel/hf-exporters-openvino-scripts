"""Resolve local Optimum-style compression recipes without importing Optimum.

Explicit CLI settings override the matching model/component entry in int4_configs.json.
Unknown models retain the generic fallback recipe. Calibration can use preprocessed
.npz files or Hugging Face text/TextVQA samples selected by the exporter's recipe
or an explicit dataset override.
"""

import copy
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import tempfile


DEFAULT_CONFIG_PATH = Path(__file__).with_name("int4_configs.json")
DEFAULT_VLM_CALIBRATION_SAMPLES = 32
DEFAULT_TEXT_CALIBRATION_SAMPLES = 128
TEXT_CALIBRATION_DATASETS = {
    "wikitext2": ("Salesforce/wikitext", "wikitext-2-raw-v1"),
    "gsm8k": ("openai/gsm8k", "main"),
}
FALLBACK_CONFIG = {
    "bits": 4, "sym": False, "group_size": 128, "ratio": 1.0,
    "all_layers": False, "sensitivity_metric": "weight_quantization_error",
}
COMPONENT_NAMES = {
    "decoder": "lm_model", "embeddings": "text_embeddings_model",
    "vision": "vision_embeddings_model",
}


def resolve_int4_config(model_id, config_path=None, *, component=None, group_size=None, ratio=None):
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    document = {}
    if path.exists():
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(document, dict) or not isinstance(document.get("models", {}), dict):
            raise ValueError(f"Invalid INT4 config file: {path}; expected a models object")
    elif config_path is not None:
        raise FileNotFoundError(f"INT4 config file does not exist: {path}")
    models = document.get("models", {})
    candidates = [str(model_id)]
    local_config = Path(model_id) / "config.json"
    if local_config.is_file():
        candidates.append(json.loads(local_config.read_text(encoding="utf-8")).get("_name_or_path", ""))
    recipe = next((models[name] for name in candidates if name in models), None)
    if recipe is None:
        matches = [value for name, value in models.items()
                   if name.rsplit("/", 1)[-1] == Path(model_id).name]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous INT4 model basename: {model_id}; use the full model ID")
        recipe = matches[0] if matches else None
    matched = recipe is not None
    if matched:
        recipe = copy.deepcopy(recipe)
        if not isinstance(recipe, dict):
            raise ValueError(f"INT4 recipe for {model_id} must be an object")
        if "quantization_configs" in recipe:
            name = COMPONENT_NAMES.get(component, component or "lm_model")
            recipe = recipe["quantization_configs"].get(name, recipe.get("default_config"))
            if recipe is None:
                return None
        result = {"bits": 4, "sym": False, "ratio": 1.0, **recipe}
    else:
        result = {**FALLBACK_CONFIG, **document.get("defaults", {})}
        if component == "embeddings":
            return None
    if group_size is not None and result.get("bits") == 4:
        result["group_size"] = group_size
    if ratio is not None and result.get("bits") == 4:
        result["ratio"] = ratio
    validate_config(result)
    return result


def validate_config(recipe):
    supported = {
        "bits", "sym", "group_size", "ratio", "all_layers", "sensitivity_metric",
        "quant_method", "awq", "dataset", "num_samples", "scale_estimation", "gptq",
        "lora_correction", "backup_precision", "ignored_scope", "weight_only",
        "group_size_fallback", "dq_group_size",
    }
    unknown = set(recipe) - supported
    if unknown:
        raise ValueError(f"Unsupported compression settings: {', '.join(sorted(unknown))}")
    if recipe.get("bits", 4) not in (4, 8):
        raise ValueError("Only INT4 and INT8 weight compression are supported")
    if not isinstance(recipe.get("sym", False), bool):
        raise ValueError("sym must be a boolean")
    if not 0 <= recipe.get("ratio", 1.0) <= 1:
        raise ValueError("ratio must be between 0 and 1")
    size = recipe.get("group_size")
    if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size == 0 or size < -1):
        raise ValueError("group_size must be positive or -1")
    if recipe.get("quant_method", "default") not in ("default", "awq"):
        raise ValueError("Only default and awq weight compression methods are supported")
    if recipe.get("weight_only", True) is not True:
        raise ValueError("Activation quantization is not supported by this weight-only exporter")


def make_compression_config(recipe):
    import nncf
    from transformers.exporters.quantizers.quantizer_nncf import NNCFWeightCompressionConfig

    validate_config(recipe)
    bits = recipe.get("bits", 4)
    kwargs = {
        "mode": nncf.CompressWeightsMode(f"int{bits}_{'sym' if recipe.get('sym') else 'asym'}"),
    }
    for name in ("group_size", "all_layers", "scale_estimation", "gptq", "lora_correction"):
        if name in recipe:
            kwargs[name] = recipe[name]
    if bits == 4:
        kwargs["ratio"] = recipe.get("ratio", 1.0)
        kwargs["awq"] = recipe.get("awq", recipe.get("quant_method") == "awq")
        metric = recipe.get("sensitivity_metric")
        if metric is not None or not recipe.get("dataset"):
            kwargs["sensitivity_metric"] = nncf.SensitivityMetric(metric or "weight_quantization_error")
    if recipe.get("num_samples") is not None:
        kwargs["subset_size"] = recipe["num_samples"]
    elif recipe.get("dataset") == "textvqa":
        kwargs["subset_size"] = DEFAULT_VLM_CALIBRATION_SAMPLES
    elif recipe.get("dataset") in TEXT_CALIBRATION_DATASETS:
        kwargs["subset_size"] = DEFAULT_TEXT_CALIBRATION_SAMPLES
    if recipe.get("backup_precision") is not None:
        kwargs["backup_mode"] = nncf.BackupMode(recipe["backup_precision"])
    if recipe.get("ignored_scope") is not None:
        kwargs["ignored_scope"] = nncf.IgnoredScope(**recipe["ignored_scope"])
    if recipe.get("group_size_fallback") is not None:
        kwargs["advanced_parameters"] = nncf.AdvancedCompressionParameters(
            group_size_fallback_mode=nncf.GroupSizeFallbackMode(recipe["group_size_fallback"]),
        )
    return NNCFWeightCompressionConfig(**kwargs)


def make_quantizer(recipe, calibration_data=None, *, calibration_samples=None):
    import nncf
    from transformers.exporters.quantizers.quantizer_nncf import NNCFOpenVINOQuantizer

    if recipe is None:
        return None
    if calibration_data is not None and calibration_samples is not None:
        raise ValueError("Choose either calibration .npz files or a calibration sample provider")
    compression = make_compression_config(recipe)
    needs_data = recipe.get("dataset") or any(recipe.get(name) for name in ("scale_estimation", "gptq"))
    if needs_data and calibration_data is None and calibration_samples is None:
        raise ValueError(
            f"This recipe requires representative calibration data (dataset={recipe.get('dataset')!r}). "
            "Supported recipe datasets are prepared automatically by the LLM/VLM export scripts. "
            "For other datasets, supply --calibration-data with preprocessed OpenVINO input .npz files, or edit the "
            "local recipe to remove dataset and disable data-dependent algorithms. "
            "Export example inputs will not be substituted for calibration data."
        )
    paths = []
    if calibration_data is not None and recipe.get("bits", 4) == 4:
        paths = sorted(Path(calibration_data).glob("*.npz"))
        if not paths:
            raise ValueError(f"No calibration .npz files in {calibration_data}")

    class OpenVINOWeightQuantizer(NNCFOpenVINOQuantizer):
        def quantize(self, model, sample_inputs, export_format):
            dataset = None
            if calibration_samples is not None and recipe.get("bits", 4) == 4:
                dataset = nncf.Dataset(calibration_samples(model, sample_inputs))
            elif calibration_data is not None and recipe.get("bits", 4) == 4:
                import numpy as np

                names = {port.get_any_name() for port in model.inputs}

                class Samples:
                    def __iter__(self):
                        for path in paths:
                            with np.load(path, allow_pickle=False) as sample:
                                missing = names - set(sample.files)
                                if missing:
                                    raise ValueError(f"{path} is missing inputs: {sorted(missing)}")
                                yield {name: sample[name] for name in names}

                dataset = nncf.Dataset(Samples())
            return nncf.compress_weights(model, dataset=dataset, **self.config.to_kwargs())

    return OpenVINOWeightQuantizer(compression)


def text_calibration_samples(tokenizer, recipe, seed=42, sequence_length=128):
    from datasets import load_dataset

    name = recipe.get("dataset")
    if name not in TEXT_CALIBRATION_DATASETS:
        raise ValueError(f"Unsupported text calibration dataset: {name!r}")
    limit = recipe.get("num_samples", DEFAULT_TEXT_CALIBRATION_SAMPLES)
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("Calibration num_samples must be a positive integer")
    dataset = load_dataset(*TEXT_CALIBRATION_DATASETS[name], split="train", streaming=True)
    shuffle_kwargs = {"seed": seed}
    if "max_buffer_input_shards" in inspect.signature(dataset.shuffle).parameters:
        shuffle_kwargs["max_buffer_input_shards"] = 1
    iterator = iter(dataset.shuffle(**shuffle_kwargs))
    count = 0
    try:
        for row in iterator:
            text = row.get("text") if name == "wikitext2" else "\n".join(
                str(row.get(field, "")) for field in ("question", "answer")
            )
            if not isinstance(text, str) or not text.strip():
                continue
            sample = dict(tokenizer(text, return_tensors="pt", truncation=True, max_length=sequence_length))
            if sample["input_ids"].shape[1] < 2:
                continue
            yield sample
            count += 1
            if count >= limit:
                break
        if count == 0:
            raise ValueError(f"{name} calibration dataset has no usable text samples")
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


def textvqa_samples(processor, num_samples, seed):
    from datasets import load_dataset

    dataset = load_dataset("lmms-lab/textvqa", split="validation", streaming=True)
    shuffle_kwargs = {"seed": seed}
    if "max_buffer_input_shards" in inspect.signature(dataset.shuffle).parameters:
        shuffle_kwargs["max_buffer_input_shards"] = 1
    dataset = dataset.shuffle(**shuffle_kwargs)
    iterator = iter(dataset)
    try:
        for index in range(num_samples):
            try:
                row = next(iterator)
            except StopIteration:
                if index == 0:
                    raise ValueError("TextVQA calibration dataset is empty") from None
                break
            question = row.get("question")
            image = row.get("image")
            if not isinstance(question, str) or not question.strip() or image is None:
                raise ValueError(f"Invalid TextVQA calibration sample {index}: expected image and question")
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": question},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            with image.convert("RGB") as rgb:
                yield dict(processor(text=text, images=rgb, return_tensors="pt"))
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


@contextmanager
def prepare_textvqa_calibration(model, processor, recipes, *, exporter, export_configs, seed=42):
    limit = max((recipe.get("num_samples", DEFAULT_VLM_CALIBRATION_SAMPLES)
                 for recipe in recipes.values() if recipe is not None and recipe.get("bits", 4) == 4), default=0)
    samples = textvqa_samples(processor, limit, seed)
    with _prepare_calibration(model, recipes, samples, exporter=exporter, export_configs=export_configs,
                              default_samples=DEFAULT_VLM_CALIBRATION_SAMPLES, label="TextVQA") as providers:
        yield providers


@contextmanager
def prepare_text_calibration(model, tokenizer, recipe, *, exporter, export_configs, seed=42,
                             preserve_attention_mask=False):
    samples = text_calibration_samples(tokenizer, recipe, seed)
    with _prepare_calibration(model, {"model": recipe}, samples, exporter=exporter, export_configs=export_configs,
                              default_samples=DEFAULT_TEXT_CALIBRATION_SAMPLES,
                              label=recipe["dataset"], preserve_attention_mask=preserve_attention_mask) as providers:
        yield providers


@contextmanager
def _prepare_calibration(model, recipes, samples, *, exporter, export_configs, default_samples, label,
                         preserve_attention_mask=False):
    import torch
    from transformers.exporters.decompose import decode_graph_name
    from transformers.exporters.precompute import precompute_export_inputs
    from transformers.exporters.quantizers.utils import capture_calibration_inputs, openvino_feed
    from transformers.exporters.utils import get_leaf_tensors

    selected = {name: recipe for name, recipe in recipes.items()
                if recipe is not None and recipe.get("bits", 4) == 4}
    limits = {name: recipe.get("num_samples", default_samples) for name, recipe in selected.items()}
    if not limits:
        yield {}
        return
    if any(not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0 for limit in limits.values()):
        raise ValueError("Calibration num_samples must be a positive integer")
    generation_config, multi_token_decode, decoder_writes_cross_cache = exporter._resolve_generation_options(
        model, export_configs, None, None, None,
    )
    with tempfile.TemporaryDirectory(prefix="hf-calibration-") as directory:
        root = Path(directory)
        paths = {name: [] for name in selected}
        try:
            with torch.no_grad():
                for index, sample in enumerate(samples):
                    captured = capture_calibration_inputs(
                        model, [sample], generation_config=generation_config,
                        multi_token_decode=multi_token_decode,
                        decoder_writes_cross_cache=decoder_writes_cross_cache,
                    )
                    for name in selected:
                        if index >= limits[name]:
                            continue
                        if not captured.get(name):
                            raise ValueError(f"Native calibration capture did not produce component {name}")
                        component_inputs = captured[name]
                        if multi_token_decode and name == decode_graph_name(captured):
                            component_inputs = component_inputs[-1:]
                        for step, inputs in enumerate(component_inputs):
                            if preserve_attention_mask and multi_token_decode and name == decode_graph_name(captured):
                                inputs = {**inputs, "attention_mask": sample["attention_mask"]}
                            inputs = precompute_export_inputs(model.config, inputs)
                            values = {key: value.detach().cpu().clone()
                                      for key, value in get_leaf_tensors(inputs).items()}
                            values.update({key: value for key, value in inputs.items()
                                           if isinstance(value, (bool, int, float))})
                            path = root / f"{name}-{index:05d}-{step}.pt"
                            torch.save(values, path)
                            paths[name].append(path)
                    del captured, sample
                    print(f"Prepared {label} calibration sample {index + 1}/{max(limits.values())}", flush=True)
        finally:
            samples.close()
        if any(not files for files in paths.values()):
            raise ValueError(f"{label} calibration produced no samples")

        def provider(files):
            def feeds(graph, sample_inputs):
                fp32_inputs = {
                    name.removeprefix("input.") for port in graph.inputs
                    if port.get_element_type().get_type_name() == "f32" for name in port.get_names()
                }
                class Samples:
                    def __iter__(self):
                        for path in files:
                            values = torch.load(path, map_location="cpu", weights_only=True)
                            values = {
                                name: value.float() if name in fp32_inputs and isinstance(value, torch.Tensor)
                                and value.dtype == torch.bfloat16
                                else value for name, value in values.items()
                            }
                            feed = openvino_feed(graph, values)
                            missing = [port.get_any_name() for port in graph.inputs
                                       if not port.get_names().intersection(feed)]
                            if missing:
                                raise ValueError(f"{label} calibration is missing inputs: {sorted(missing)}")
                            yield feed

                return Samples()

            return feeds

        yield {name: provider(files) for name, files in paths.items()}


def apply_runtime_config(graph, recipe):
    if recipe is not None and recipe.get("dq_group_size") is not None:
        graph.set_rt_info(str(recipe["dq_group_size"]), ["runtime_options", "DYNAMIC_QUANTIZATION_GROUP_SIZE"])