"""One-image, greedy inference for native and legacy HF OpenVINO VLM exports."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from types import MethodType, SimpleNamespace

import numpy as np
import openvino as ov
from PIL import Image
import torch
from transformers import AutoConfig, AutoProcessor


def infer(request, values, *, copy_outputs=True, timings=None):
    """Return snapshots by default; output views expire when this request runs again."""
    started = perf_counter() if timings is not None else 0.0
    feeds = {}
    for port in request.get_compiled_model().inputs:
        name = port.get_any_name()
        if name not in values:
            raise ValueError(f"Missing exported input {name!r}")
        value = values[name]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            value = value.float().numpy() if value.dtype == torch.bfloat16 else value.numpy()
        array = np.asarray(value)
        if not port.get_partial_shape().compatible(ov.PartialShape(array.shape)):
            raise ValueError(
                f"Input {name!r} has shape {array.shape}, expected {port.get_partial_shape()}. "
                "Vision shapes are fixed: use the same processed image geometry as at export."
            )
        if port.get_element_type() == ov.Type.bf16:
            tensor = ov.Tensor(ov.Type.bf16, array.shape)
            bits = torch.from_numpy(np.ascontiguousarray(array)).to(torch.bfloat16).view(torch.int16)
            tensor.data.view(np.uint16)[...] = bits.numpy().view(np.uint16)
            feeds[name] = tensor
        else:
            feeds[name] = np.ascontiguousarray(array, dtype=port.get_element_type().to_dtype())
    inference_start = perf_counter() if timings is not None else 0.0
    result = request.infer(feeds, share_inputs=True, share_outputs=True)
    inference_end = perf_counter() if timings is not None else 0.0
    outputs = {port.get_any_name(): np.array(result[port], copy=True) if copy_outputs else result[port]
               for port in request.get_compiled_model().outputs}
    if timings is not None:
        timings["request_s"] = timings.get("request_s", 0.0) + inference_end - inference_start
        timings["overhead_s"] = timings.get("overhead_s", 0.0) + (
            inference_start - started + perf_counter() - inference_end
        )
    return outputs


class OpenVINOVLM:
    def __init__(self, model_dir, device="CPU", *, processor=None, compile_config=None):
        self.model_dir = Path(model_dir)
        self.config = AutoConfig.from_pretrained(self.model_dir, local_files_only=True)
        self.family = self.config.model_type
        if self.family not in ("gemma4", "qwen3_5"):
            raise ValueError(f"Unsupported model type {self.family!r}; supported: gemma4, qwen3_5")
        self.processor = processor if processor is not None else AutoProcessor.from_pretrained(
            self.model_dir, local_files_only=True,
        )
        manifest_path = self.model_dir / "export_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        self.vision_shapes = manifest.get("vision_input_shapes", {})
        self.image_grid_thw = manifest.get("image_grid_thw")
        core = ov.Core()
        requests = {}
        for name in ("embeddings", "vision", "decoder"):
            filename = manifest.get("components", {}).get(name, {}).get("file", f"openvino_{name}.xml")
            path = self.model_dir / filename
            if not path.is_file():
                raise ValueError(f"Missing {path}; this runner needs the HF component export, not a GenAI package")
            compiled = core.compile_model(path, device, compile_config or {})
            requests[name] = compiled.create_infer_request()
        self.embeddings = requests["embeddings"]
        self.vision = requests["vision"]
        self.decoder = requests["decoder"]
        if not self.decoder.query_state():
            raise ValueError("The decoder must contain stateful cache variables")
        generation_path = self.model_dir / "generation_config.json"
        generation = json.loads(generation_path.read_text(encoding="utf-8")) if generation_path.exists() else {}
        eos = generation.get("eos_token_id")
        if eos is None:
            eos = self.config.get_text_config().eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos]) - {None}

    def prepare_inputs(self, processed, *, timings=None):
        ids = processed["input_ids"].detach().cpu().numpy()
        if ids.shape[0] != 1:
            raise ValueError("Only one image and one prompt at a time are supported")
        image_mask = ids == self.config.image_token_id
        if not image_mask.any():
            raise ValueError("The processed prompt contains no image tokens")
        mask = processed.get("attention_mask", torch.ones_like(processed["input_ids"]))
        mask = mask.detach().cpu().numpy()
        if not np.all(mask == 1):
            raise ValueError("Use a single unpadded prompt")
        token_types = processed.get("mm_token_type_ids")
        types = (token_types.detach().cpu().numpy() if token_types is not None
             else image_mask.astype(np.int64))
        if self.image_grid_thw is not None and processed["image_grid_thw"].tolist() != self.image_grid_thw:
            raise ValueError("Image grid differs from export; use the same processed image geometry as at export")
        if self.family == "gemma4":
            embedding_ids = np.where(image_mask, self.config.get_text_config().pad_token_id, ids)
            vision_inputs = {name: processed[name] for name in ("pixel_values", "image_position_ids")}
            positions = np.arange(ids.shape[1], dtype=np.int64)[None, :]
            next_position = ids.shape[1]
        else:
            from transformers.exporters.utils import precompute_export_inputs
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

            embedding_ids = ids
            vision_inputs = {"hidden_states": processed["pixel_values"],
                             "grid_thw": processed["image_grid_thw"]}
            vision_inputs = precompute_export_inputs(self.config.vision_config, vision_inputs)
            metadata = SimpleNamespace(config=self.config)
            metadata.get_vision_position_ids = MethodType(Qwen3_5Model.get_vision_position_ids, metadata)
            get_rope_index = MethodType(Qwen3_5Model.get_rope_index, metadata)
            positions, delta = get_rope_index(
                torch.from_numpy(ids), torch.from_numpy(types),
                image_grid_thw=processed["image_grid_thw"], attention_mask=torch.from_numpy(mask),
            )
            positions = positions.numpy()
            next_position = ids.shape[1] + int(delta.item())
        vision_start = perf_counter()
        for name, shape in self.vision_shapes.items():
            if name in vision_inputs and list(vision_inputs[name].shape) != shape:
                raise ValueError(
                    f"Vision input {name!r} has shape {list(vision_inputs[name].shape)}, expected {shape}. "
                    "Use the same processed image geometry as at export."
                )
        try:
            features = infer(self.vision, vision_inputs, copy_outputs=False)["image_features"]
        except RuntimeError as error:
            raise ValueError(
                "Vision inference failed. Vision shapes are fixed to the export image; "
                "use the same processed image geometry as at export. " + str(error)
            ) from error
        vision_seconds = perf_counter() - vision_start
        embedding_start = perf_counter()
        embeddings = infer(self.embeddings, {"input_ids": embedding_ids}, copy_outputs=False)
        if timings is not None:
            timings["text_embedding_s"] = perf_counter() - embedding_start
        embeds = embeddings["inputs_embeds"]
        features = features.reshape(-1, embeds.shape[-1])
        if features.shape[0] != int(image_mask.sum()):
            raise ValueError(
                f"Image token/feature mismatch: {image_mask.sum()} tokens, {features.shape[0]} features. "
                "Use the same processed image geometry as at export."
            )
        embeds[image_mask] = features
        return {
            **embeddings, "position_ids": positions, "attention_mask": mask,
            "mm_token_type_ids": types, "beam_idx": np.array([0], np.int32),
        }, next_position, vision_seconds

    def preprocess(self, image, prompt="Describe this image."):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        with Image.open(image) as source:
            return self.processor(text=text, images=source.convert("RGB"), return_tensors="pt")

    def generate(self, image, prompt="Describe this image.", *, max_new_tokens=100):
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        started = perf_counter()
        processed = self.preprocess(image, prompt)
        preprocessing_seconds = perf_counter() - started
        result = self.generate_processed(processed, max_new_tokens=max_new_tokens)
        for name in ("prepare_s", "ttft_s", "generate_s", "total_s"):
            result["timings"][name] += preprocessing_seconds
        return result

    def generate_processed(self, processed, *, max_new_tokens=100, decode_text=True, profile=False):
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        started = perf_counter()
        self.decoder.reset_state()
        preparation_start = perf_counter()
        preparation_timings = {}
        feeds, next_position, vision_seconds = self.prepare_inputs(processed, timings=preparation_timings)
        prepared = perf_counter()
        logits = infer(self.decoder, feeds, copy_outputs=False)["logits"]
        prefill_seconds = perf_counter() - prepared
        token_ids = []
        token_times = []
        decode_seconds = 0.0
        decode_steps = 0
        decoder_timings = {} if profile else None
        embedding_timings = {} if profile else None
        prompt_length = feeds["attention_mask"].shape[1]
        attention_buffer = np.ones((1, prompt_length + max_new_tokens), dtype=np.int64)
        type_buffer = np.zeros((1, prompt_length + max_new_tokens), dtype=np.int64)
        type_buffer[:, :prompt_length] = feeds["mm_token_type_ids"]
        for step in range(max_new_tokens):
            token = int(np.argmax(logits[0, -1]))
            token_ids.append(token)
            token_times.append(perf_counter())
            if token in self.eos_ids or step + 1 == max_new_tokens:
                break
            decode_start = perf_counter()
            embeddings = infer(
                self.embeddings, {"input_ids": np.array([[token]], np.int64)},
                copy_outputs=False, timings=embedding_timings,
            )
            length = prompt_length + step + 1
            feeds["attention_mask"] = attention_buffer[:, :length]
            feeds["mm_token_type_ids"] = type_buffer[:, :length]
            feeds.update(embeddings)
            shape = (1, 1) if self.family == "gemma4" else (3, 1, 1)
            feeds["position_ids"] = np.full(shape, next_position, dtype=np.int64)
            logits = infer(self.decoder, feeds, copy_outputs=False, timings=decoder_timings)["logits"]
            next_position += 1
            decode_steps += 1
            decode_seconds += perf_counter() - decode_start
        generated = perf_counter()
        decode_intervals = len(token_times) - 1
        tpot_seconds = ((token_times[-1] - token_times[0]) / decode_intervals
                        if decode_intervals else None)
        profile_timings = {}
        if profile and decoder_timings is not None and embedding_timings is not None:
            decoder_seconds = decoder_timings.get("request_s", 0.0)
            embedding_seconds = embedding_timings.get("request_s", 0.0)
            profile_timings = {
                "decode_decoder_request_s": decoder_seconds,
                "decode_embedding_request_s": embedding_seconds,
                "decode_host_s": (token_times[-1] - token_times[0]) - decoder_seconds - embedding_seconds,
            }
        return {
            "text": self.processor.tokenizer.decode(token_ids, skip_special_tokens=True) if decode_text else "",
            "token_ids": token_ids,
            "input_tokens": int(processed["input_ids"].shape[1]),
            "timings": {
                "prepare_s": prepared - started, "vision_s": vision_seconds,
                "embeddings_prepare_s": prepared - preparation_start,
                "text_embedding_s": preparation_timings["text_embedding_s"],
                "prefill_s": prefill_seconds, "decode_s": decode_seconds,
                "decode_steps": decode_steps,
                "generate_s": generated - started,
                "ttft_s": token_times[0] - started,
                "tpot_s": tpot_seconds,
                "throughput_tokens_per_s": 1 / tpot_seconds if tpot_seconds else None,
                "decode_tokens_per_s": decode_steps / decode_seconds if decode_seconds else 0.0,
                "total_s": perf_counter() - started,
                **profile_timings,
            },
        }


def generate_native(model_dir, image, prompt, device="CPU", *, max_new_tokens=100):
    from transformers.exporters.base import ExportArtifacts
    from transformers.exporters.configs import ExportFormat
    from transformers.exporters.runtimes.generation import ExportedMultimodalModelForCausalLM
    from benchmark_vlm_hf_exporters import TokenTimer

    artifacts = ExportArtifacts.from_pretrained(model_dir, local_files_only=True)
    if artifacts.export_format != ExportFormat.OPENVINO or not artifacts.generation or not artifacts.multimodal:
        raise ValueError("Expected a native OpenVINO multimodal generation export")
    normalized = device.upper()
    if normalized == "AUTO":
        runtime_device = None
    elif normalized == "CPU":
        runtime_device = "cpu"
    elif normalized == "GPU" or normalized.startswith("GPU."):
        runtime_device = "xpu" + (":" + normalized[4:] if normalized.startswith("GPU.") else "")
    else:
        runtime_device = device.lower()
    model = artifacts.runtime(device=runtime_device)
    if not isinstance(model, ExportedMultimodalModelForCausalLM):
        raise ValueError("Expected the native multimodal causal generation runtime")
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    started = perf_counter()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    with Image.open(image) as source:
        processed = processor(text=text, images=source.convert("RGB"), return_tensors="pt")
    prepared = perf_counter()
    timer = TokenTimer()
    with torch.inference_mode():
        outputs = model.generate(
            **processed, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
            num_return_sequences=1, return_dict_in_generate=False, streamer=timer,
        )
    generated = perf_counter()
    token_ids = outputs[0, processed["input_ids"].shape[1]:].tolist()
    intervals = len(timer.times) - 1
    tpot = (timer.times[-1] - timer.times[0]) / intervals if intervals > 0 else None
    return {
        "text": processor.tokenizer.decode(token_ids, skip_special_tokens=True),
        "token_ids": token_ids,
        "input_tokens": int(processed["input_ids"].shape[1]),
        "timings": {
            "preprocess_s": prepared - started, "generate_s": generated - prepared,
            "ttft_s": timer.times[0] - started if timer.times else None,
            "tpot_s": tpot, "throughput_tokens_per_s": 1 / tpot if tpot else None,
            "total_s": perf_counter() - started,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args(argv)
    if not args.image.is_file():
        parser.error(f"Image file does not exist: {args.image}")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if (args.model_dir / "export.json").is_file():
        result = generate_native(args.model_dir, args.image, args.prompt, args.device,
                                 max_new_tokens=args.max_new_tokens)
    else:
        runner = OpenVINOVLM(args.model_dir, args.device)
        result = runner.generate(args.image, args.prompt, max_new_tokens=args.max_new_tokens)
    print(result["text"])
    print(json.dumps(result["timings"], indent=2))


if __name__ == "__main__":
    main()