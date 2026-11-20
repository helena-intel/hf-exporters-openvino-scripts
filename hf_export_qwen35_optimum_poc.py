"""Experimental native HF Qwen3.5 VLM export using Optimum Intel's recurrent patch."""

import argparse
from contextlib import ExitStack
from pathlib import Path
from types import MethodType
from unittest.mock import patch


def export_patched(export, model, inputs, config, **kwargs):
    import torch
    from openvino.frontend.pytorch import ConversionExtension
    from optimum.exporters.openvino._ov_ops import convert_recurrent_attention_cell
    from optimum.exporters.openvino.model_patcher import (
        RecurrentAttentionCell,
        patched_recurrent_gated_delta_rule,
        qwen3_5_gated_delta_net_forward,
    )
    from transformers.exporters import exporter_openvino

    if model.config.model_type != "qwen3_5":
        raise ValueError("This proof of concept only supports dense Qwen3.5 VLMs")

    @torch.library.custom_op("qwen35_poc::recurrent_cell", mutates_args=())
    def recurrent_cell(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                       decay: torch.Tensor, beta: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return RecurrentAttentionCell()(query, key, value, decay, beta, state)

    @recurrent_cell.register_fake
    def fake_recurrent_cell(query, key, value, decay, beta, state):
        return value.new_empty((value.numel() + state.numel(),))

    def forward(layer, hidden_states, cache_params=None, attention_mask=None, **options):
        if cache_params is not None and cache_params.layers[layer.layer_idx].conv_states[0] is None:
            return type(layer).forward(
                layer, hidden_states, cache_params=cache_params, attention_mask=attention_mask, **options,
            )
        return qwen3_5_gated_delta_net_forward(
            layer, hidden_states, cache_params=cache_params, attention_mask=attention_mask, **options,
        )

    extension = ConversionExtension("qwen35_poc.recurrent_cell.default", convert_recurrent_attention_cell)
    with ExitStack() as stack:
        stack.enter_context(patch.object(
            exporter_openvino, "_OV_CONVERSION_EXTENSIONS",
            [*exporter_openvino._OV_CONVERSION_EXTENSIONS, extension],
        ))
        for layer_type, decoder_layer in zip(model.config.text_config.layer_types,
                                             model.model.language_model.layers, strict=True):
            if layer_type == "linear_attention":
                layer = decoder_layer.linear_attn
                stack.enter_context(patch.object(layer, "forward", MethodType(forward, layer)))
                stack.enter_context(patch.object(layer, "recurrent_gated_delta_rule",
                                                 patched_recurrent_gated_delta_rule, create=True))
                stack.enter_context(patch.object(layer, "recurrent_attention_cell", recurrent_cell, create=True))
        artifacts = export(model, inputs, config, multi_token_decode=True, **kwargs)
    decoder = artifacts["language_model"]
    if not decoder.program.input("inputs_embeds").partial_shape[1].is_dynamic:
        raise RuntimeError("The decoder does not accept a dynamic query length")
    if not any(node.get_type_name() == "Loop" for node in decoder.program.get_ops()):
        raise RuntimeError("The decoder is missing the Optimum recurrent Loop")
    return artifacts


def main(argv=None):
    from hf_export_models_int4 import export_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="Describe this image")
    parser.add_argument("--int4-config", type=Path)
    args = parser.parse_args(argv)
    if not args.image.is_file():
        parser.error(f"Image file does not exist: {args.image}")

    from transformers.exporters import OpenVINOExporter

    original_export = OpenVINOExporter.export_for_generation

    def patched_export(exporter, *export_args, **export_kwargs):
        return export_patched(original_export.__get__(exporter), *export_args, **export_kwargs)

    with patch.object(OpenVINOExporter, "export_for_generation", patched_export):
        export_model(args.model_id, args.output_dir, image=args.image, prompt=args.prompt,
                     int4_config=args.int4_config)


if __name__ == "__main__":
    main()