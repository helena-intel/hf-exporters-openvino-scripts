import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import openvino as ov
import torch
from transformers import (
    Gemma4Config, Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration, Gemma4TextConfig, Gemma4VisionConfig,
    Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5TextConfig, Qwen3_5VisionConfig,
)
from transformers.exporters.utils import get_leaf_tensors

from hf_export_multimodels_int4 import cached_decoder, export_component, export_model, main, model_adapters


def tiny_gemma_config():
    config = Gemma4TextConfig(
        vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=64, global_head_dim=64,
        num_global_key_value_heads=1, layer_types=["sliding_attention", "full_attention"],
        num_kv_shared_layers=0, sliding_window=16, hidden_size_per_layer_input=16,
        vocab_size_per_layer_input=256, eos_token_id=1, pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    return config


class CachedDecoderTests(unittest.TestCase):
    def test_gemma_bounded_cache_prefill_decode_and_reset(self):
        torch.manual_seed(7)
        model = Gemma4ForCausalLM(tiny_gemma_config()).eval()
        decoder, inputs, shapes = cached_decoder(model, "gemma4")
        with tempfile.TemporaryDirectory() as directory:
            graph = export_component(
            decoder, inputs, Path(directory) / "decoder.xml", shapes,
                stateful=True, quantize=False, sliding_window=16,
                layer_types=model.config.layer_types,
            )
            compiled = ov.Core().compile_model(
                graph, "CPU", {"INFERENCE_PRECISION_HINT": "f32"},
            )
            request = compiled.create_infer_request()
            for prompt_length in (5, 23, 7):
                request.reset_state()
                reference_cache = None
                seen = 0
                for length in (prompt_length, 1, 1):
                    ids = torch.arange(2, 2 + length).view(1, -1)
                    positions = torch.arange(seen, seen + length).view(1, -1)
                    embeds = model.get_input_embeddings()(ids).detach()
                    per_layer = model.model.get_per_layer_inputs(ids, None).detach()
                    with torch.no_grad():
                        expected = model(
                            inputs_embeds=embeds, per_layer_inputs=per_layer,
                            position_ids=positions, past_key_values=reference_cache,
                            use_cache=True,
                        )
                    reference_cache = expected.past_key_values
                    query = positions.view(1, 1, -1, 1)
                    masks = {}
                    for kind in ("full_attention", "sliding_attention"):
                        start = max(0, seen - 15) if kind == "sliding_attention" else 0
                        keys = torch.arange(start, seen + length).view(1, 1, 1, -1)
                        allowed = keys <= query
                        if kind == "sliding_attention":
                            allowed &= keys > query - 16
                        masks[kind] = torch.where(allowed, 0.0, torch.finfo(torch.float32).min)
                    flattened = get_leaf_tensors({
                        "inputs_embeds": embeds, "per_layer_inputs": per_layer,
                        "position_ids": positions,
                        "attention_mask": torch.ones((1, seen + length), dtype=torch.long),
                    })
                    feeds = {
                        port.get_any_name(): np.array([0], np.int32)
                        if port.get_any_name() == "beam_idx"
                        else flattened[port.get_any_name()].numpy()
                        for port in compiled.inputs
                    }
                    actual = request.infer(feeds)["logits"]
                    np.testing.assert_allclose(actual, expected.logits[:, -1:].numpy(), atol=.003, rtol=.003)
                    seen += length
                    for state in request.query_state():
                        if "layers.0." in state.name:
                            self.assertLessEqual(state.state.shape[2], 15)


class MediaTests(unittest.TestCase):
    def test_int8_embedding_export_returns_openvino_model(self):
        class Embeddings(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(128, 32)

            def forward(self, input_ids):
                return {"inputs_embeds": self.embedding(input_ids)}

        torch.manual_seed(7)
        model = Embeddings().eval()
        inputs = {"input_ids": torch.tensor([[2, 3, 4]], dtype=torch.long)}
        with torch.no_grad():
            expected = model(**inputs)["inputs_embeds"].numpy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.xml"
            graph = export_component(
                model, inputs, path, {"input_ids": {0: torch.export.Dim.AUTO, 1: torch.export.Dim.AUTO}},
                compression_config={"bits": 8, "sym": True},
            )
            self.assertIsInstance(graph, ov.Model)
            restored = ov.Core().read_model(path)
            self.assertTrue(any(node.get_output_element_type(0) == ov.Type.i8
                                for node in restored.get_ops() if node.get_type_name() == "Constant"))
            compiled = ov.Core().compile_model(restored, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
            actual = compiled({compiled.inputs[0].get_any_name(): inputs["input_ids"].numpy()})[0]
            np.testing.assert_allclose(actual, expected, atol=.04, rtol=.02)

    def check_component(self, component, inputs):
        from transformers.exporters.utils import precompute_export_inputs

        inputs = precompute_export_inputs(component.config, inputs)
        with torch.no_grad():
            expected = component(**inputs)
        with tempfile.TemporaryDirectory() as directory:
            graph = export_component(
                component, inputs, Path(directory) / "component.xml",
                {name: None for name in inputs}, quantize=False,
            )
            compiled = ov.Core().compile_model(graph, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
            flattened = get_leaf_tensors(inputs)
            actual = compiled({
                port.get_any_name(): flattened[port.get_any_name()].numpy()
                for port in compiled.inputs
            })
            for name, value in expected.items():
                np.testing.assert_allclose(actual[name], value.numpy(), atol=.003, rtol=.003)

    def gemma_model(self):
        config = Gemma4Config(
            text_config=tiny_gemma_config(),
            vision_config=Gemma4VisionConfig(
                hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                num_attention_heads=2, num_key_value_heads=2, head_dim=32,
                pooling_kernel_size=2, patch_size=2, position_embedding_size=64,
            ),
            audio_config=None,
            image_token_id=250, video_token_id=252,
        )
        config._attn_implementation = "sdpa"
        return Gemma4ForConditionalGeneration(config).eval()

    def test_gemma_vision_and_embeddings(self):
        adapters = model_adapters(self.gemma_model())
        self.assertEqual(set(adapters), {"decoder", "embeddings", "vision"})
        positions = torch.tensor([[column, row] for row in range(4) for column in range(4)])
        self.check_component(adapters["vision"], {
            "pixel_values": torch.randn(1, 16, 12),
            "image_position_ids": positions.unsqueeze(0),
        })
        self.check_component(adapters["embeddings"], {"input_ids": torch.tensor([[2, 3, 4]])})

    def test_single_model_export_driver(self):
        model = self.gemma_model()
        model.config.text_config.use_bidirectional_attention = "vision"
        positions = torch.tensor([[column, row] for row in range(4) for column in range(4)])
        self.check_export_and_inference(model, {
            "input_ids": torch.tensor([[2, 3, 250, 250, 250, 250, 4, 5]]),
            "attention_mask": torch.ones((1, 8), dtype=torch.long),
            "mm_token_type_ids": torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0]]),
            "pixel_values": torch.randn(1, 16, 12),
            "image_position_ids": positions.unsqueeze(0),
        })

    def test_custom_component_compression(self):
        import json
        from PIL import Image

        model = self.gemma_model()
        positions = torch.tensor([[column, row] for row in range(4) for column in range(4)])
        processor = Mock(return_value={
            "pixel_values": torch.randn(1, 16, 12),
            "image_position_ids": positions.unsqueeze(0),
        })
        processor.apply_chat_template.return_value = "Describe this image."
        recipes = {
            "lm_model": {"bits": 4, "sym": True, "group_size": 32, "quant_method": "awq",
                         "group_size_fallback": "ignore"},
            "text_embeddings_model": {"bits": 8, "sym": True, "weight_only": True},
            "vision_embeddings_model": {"bits": 8, "sym": True, "weight_only": True},
        }
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            Image.new("RGB", (8, 8)).save(image)
            config_path = Path(directory) / "recipes.json"
            config_path.write_text(json.dumps({"models": {"local/tiny-gemma": {
                "quantization_configs": recipes,
            }}}))
            output = Path(directory) / "export"
            with patch("transformers.AutoConfig.from_pretrained", return_value=model.config), \
                    patch("transformers.AutoProcessor.from_pretrained", return_value=processor), \
                    patch("transformers.AutoModelForImageTextToText.from_pretrained", return_value=model):
                export_model("local/tiny-gemma", output, image, int4_config=config_path)
            manifest = json.loads((output / "export_manifest.json").read_text())
            self.assertTrue(manifest["compression"]["decoder"]["sym"])
            self.assertEqual(manifest["compression"]["decoder"]["group_size"], 32)
            for name in ("vision", "embeddings"):
                self.assertEqual(manifest["compression"][name]["bits"], 8)
                self.assertTrue(manifest["components"][name]["quantized"])
                graph = ov.Core().read_model(output / manifest["components"][name]["file"])
                ov.Core().compile_model(graph, "CPU")

    def test_original_calibration_recipe_is_normalized_before_export(self):
        with patch("transformers.AutoConfig.from_pretrained", return_value=Mock(model_type="unsupported")), \
                patch("int4_config.make_quantizer") as quantizer:
            with self.assertRaisesRegex(ValueError, "Unsupported model type"):
                export_model("google/gemma-4-E2B-it", Path("unused"), Path("missing.png"))
        self.assertEqual(quantizer.call_count, 3)
        for call in quantizer.call_args_list:
            self.assertEqual(len(call.args), 1)
            for name in ("dataset", "num_samples", "scale_estimation", "gptq", "lora_correction"):
                self.assertNotIn(name, call.args[0])
        self.assertEqual(quantizer.call_args_list[0].args[0]["quant_method"], "awq")

    def check_export_and_inference(self, model, processed):
        import json
        from PIL import Image
        from inference_vlm import OpenVINOVLM, infer

        processor = Mock(return_value=processed)
        processor.apply_chat_template.return_value = "Describe this image."
        processor.tokenizer.decode.return_value = "tiny-model output"

        def unquantized_export(*args, **kwargs):
            kwargs["quantize"] = False
            return export_component(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            Image.new("RGB", (8, 8)).save(image)
            output = Path(directory) / "export"
            with patch("transformers.AutoConfig.from_pretrained", return_value=model.config), \
                    patch("transformers.AutoProcessor.from_pretrained", return_value=processor), \
                    patch("transformers.AutoModelForImageTextToText.from_pretrained", return_value=model), \
                    patch("hf_export_multimodels_int4.export_component", side_effect=unquantized_export):
                export_model("local/tiny-gemma", output, image, group_size=32)
            manifest = json.loads((output / "export_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["task"], "image-text-to-text")
            self.assertEqual(set(manifest["components"]), {"embeddings", "vision", "decoder"})
            self.assertTrue(manifest["components"]["decoder"]["stateful"])
            self.assertFalse(manifest["components"]["embeddings"]["quantized"])
            for component in manifest["components"].values():
                self.assertTrue((output / component["file"]).is_file())
            self.assertTrue((output / "config.json").is_file())
            processor.save_pretrained.assert_called_once_with(output)
            runner = OpenVINOVLM(
                output, processor=processor, compile_config={"INFERENCE_PRECISION_HINT": "f32"},
            )
            feeds, _, _ = runner.prepare_inputs(processed)
            actual = infer(runner.decoder, feeds)["logits"]
            with torch.no_grad():
                expected = model(**processed, use_cache=False).logits[:, -1:].numpy()
                tokens = model.generate(**processed, max_new_tokens=3, do_sample=False)
            np.testing.assert_allclose(actual, expected, atol=.005, rtol=.005)
            expected_tokens = tokens[0, processed["input_ids"].shape[1]:].tolist()
            for _ in range(2):
                result = runner.generate(image, "Describe this image.", max_new_tokens=3)
                self.assertEqual(result["token_ids"], expected_tokens)
                self.assertEqual(result["text"], "tiny-model output")
                self.assertGreater(result["timings"]["vision_s"], 0)
            from benchmark_vlm_hf_exporters import benchmark

            decode_calls = processor.tokenizer.decode.call_count
            measured = benchmark(runner, processed, num_warmup=1, num_iter=2, max_new_tokens=3)
            self.assertEqual(processor.tokenizer.decode.call_count, decode_calls)
            for result in measured:
                self.assertEqual(result["token_ids"], expected_tokens)
                self.assertEqual(result["input_tokens"], processed["input_ids"].shape[1])
                self.assertEqual(result["text"], "")
                timings = result["timings"]
                self.assertGreater(timings["text_embedding_s"], 0)
                self.assertGreaterEqual(timings["embeddings_prepare_s"],
                                        timings["vision_s"] + timings["text_embedding_s"])
                self.assertGreaterEqual(timings["generate_s"], timings["ttft_s"])
                if len(result["token_ids"]) > 1:
                    self.assertGreater(timings["tpot_s"], 0)
                    self.assertAlmostEqual(timings["throughput_tokens_per_s"], 1 / timings["tpot_s"])
            runner.eos_ids = {expected_tokens[0]}
            stopped = runner.generate(image, max_new_tokens=5)
            self.assertEqual(stopped["token_ids"], expected_tokens[:1])
            self.assertEqual(stopped["timings"]["decode_steps"], 0)
            self.assertIsNone(stopped["timings"]["tpot_s"])
            self.assertIsNone(stopped["timings"]["throughput_tokens_per_s"])
            with self.assertRaisesRegex(ValueError, "shape"):
                incorrect = dict(processed)
                incorrect["pixel_values"] = processed["pixel_values"][..., :3]
                runner.prepare_inputs(incorrect)

    def qwen_model(self):
        text = Qwen3_5TextConfig(
            vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=32,
            linear_key_head_dim=16, linear_value_head_dim=16,
            linear_num_key_heads=2, linear_num_value_heads=4,
            layer_types=["linear_attention", "full_attention"], eos_token_id=1, pad_token_id=0,
        )
        config = Qwen3_5Config(
            text_config=text,
            vision_config=Qwen3_5VisionConfig(
                depth=1, hidden_size=64, intermediate_size=128, num_heads=2,
                patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
                out_hidden_size=128, num_position_embeddings=64,
            ),
            image_token_id=250, video_token_id=251, vision_start_token_id=252,
        )
        config._attn_implementation = "sdpa"
        return Qwen3_5ForConditionalGeneration(config).eval()

    def test_qwen_vision_and_embeddings(self):
        adapters = model_adapters(self.qwen_model())
        self.assertEqual(set(adapters), {"decoder", "embeddings", "vision"})
        self.check_component(adapters["vision"], {
            "hidden_states": torch.randn(16, 24), "grid_thw": torch.tensor([[1, 4, 4]]),
        })
        self.check_component(adapters["embeddings"], {"input_ids": torch.tensor([[2, 3, 4]])})

    def test_qwen_export_and_inference(self):
        self.check_export_and_inference(self.qwen_model(), {
            "input_ids": torch.tensor([[2, 252, 250, 250, 250, 250, 3, 4]]),
            "attention_mask": torch.ones((1, 8), dtype=torch.long),
            "mm_token_type_ids": torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0]]),
            "pixel_values": torch.randn(16, 24),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        })


class InferenceHelperTests(unittest.TestCase):
    def test_bfloat16_inputs(self):
        from openvino import opset13 as ops
        from inference_vlm import infer

        parameter = ops.parameter([1, 3], ov.Type.bf16, name="value")
        parameter.output(0).set_names({"value"})
        converted = ops.convert(parameter, ov.Type.f32)
        converted.output(0).set_names({"converted"})
        graph = ov.Model([converted], [parameter])
        request = ov.Core().compile_model(graph, "CPU").create_infer_request()
        values = np.array([[1.25, -2.5, 0.125]], dtype=np.float32)
        np.testing.assert_array_equal(infer(request, {"value": values})["converted"], values)


class BenchmarkTests(unittest.TestCase):
    def result(self, seconds=1.0, tokens=3):
        return {"input_tokens": 272, "token_ids": list(range(tokens)), "timings": {
            "generate_s": seconds, "embeddings_prepare_s": seconds / 4,
            "vision_s": seconds / 5, "text_embedding_s": seconds / 20,
            "ttft_s": seconds / 2, "tpot_s": seconds / 10 if tokens > 1 else None,
            "throughput_tokens_per_s": 10 / seconds if tokens > 1 else None,
        }}

    def test_warmups_are_excluded_and_no_text_decoding(self):
        from benchmark_vlm_hf_exporters import benchmark

        runner = Mock()
        runner.generate_processed.side_effect = [self.result(100), self.result(100),
                                                self.result(1), self.result(2)]
        processed = {"input_ids": torch.tensor([[2, 3, 4]])}
        results = benchmark(runner, processed, num_warmup=2, num_iter=2, max_new_tokens=20)
        self.assertEqual([result["timings"]["generate_s"] for result in results], [1, 2])
        self.assertEqual(runner.generate_processed.call_count, 4)
        for call in runner.generate_processed.call_args_list:
            self.assertIs(call.args[0], processed)
            self.assertEqual(call.kwargs, {"max_new_tokens": 20, "decode_text": False})

    def test_report_statistics_and_single_token_case(self):
        import contextlib
        import io
        from benchmark_vlm_hf_exporters import print_report

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_report([self.result(1), self.result(3)], 2)
        text = output.getvalue()
        self.assertIn("Generate time: 2000.00 +/- 1000.00 ms", text)
        self.assertIn("TTFT: 1000.00 +/- 500.00 ms", text)
        self.assertIn("TPOT: 200.00 +/- 100.00 ms/token", text)
        self.assertIn("Load time: 2000.00 ms", text)
        self.assertNotIn("Tokenization time:", text)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_report([self.result(tokens=1)], 0)
        self.assertIn("TPOT: N/A", output.getvalue())
        self.assertIn("Throughput: N/A", output.getvalue())

    def test_cli_preprocesses_once(self):
        import contextlib
        import io
        from benchmark_vlm_hf_exporters import main as benchmark_main

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.jpg"
            image.touch()
            runner = Mock()
            runner.generate_processed.return_value = self.result()
            with patch("benchmark_vlm_hf_exporters.NativeOpenVINOVLM", return_value=runner), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(benchmark_main([
                    "-m", directory, "-i", str(image), "-p", "What is this image?",
                    "-d", "CPU", "-N", "1", "-n", "2", "-M", "20",
                ]), 0)
            runner.preprocess.assert_called_once_with(image, "What is this image?")
            self.assertEqual(runner.generate_processed.call_count, 3)

    def test_cli_rejects_legacy_package(self):
        import contextlib
        import io
        from benchmark_vlm_hf_exporters import main as benchmark_main

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.jpg"
            image.touch()
            output = io.StringIO()
            with contextlib.redirect_stderr(output), self.assertRaises(SystemExit) as error:
                benchmark_main(["-m", directory, "-i", str(image)])
            self.assertEqual(error.exception.code, 2)
            message = output.getvalue()
            self.assertIn("export.json", message)
            self.assertIn("hf_export_models_int4.py", message)

    def test_invalid_iteration_counts(self):
        from benchmark_vlm_hf_exporters import benchmark

        for options in ({"num_warmup": -1}, {"num_iter": 0}, {"max_new_tokens": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                benchmark(Mock(), {}, **options)


class CommandLineTests(unittest.TestCase):
    def test_dry_run_does_not_export_or_start_workers(self):
        with patch("hf_export_multimodels_int4.export_model") as export, \
                patch("subprocess.run") as run:
            self.assertEqual(main([
                "--model-id", "google/gemma-4-E2B-it", "Qwen/Qwen3.5-2B",
                "--image", "missing.jpg", "--dry-run",
            ]), 0)
        export.assert_not_called()
        run.assert_not_called()

    def test_list_file_continues_after_failure_and_forwards_options(self):
        import sys

        with tempfile.TemporaryDirectory() as directory:
            models = Path(directory) / "models.txt"
            models.write_text("\ufeff# Models\n\norg/First\norg/Second\n", encoding="utf-8")
            image = Path(directory) / "image.jpg"
            image.touch()
            with patch("subprocess.run", side_effect=[Mock(returncode=1), Mock(returncode=0)]) as run:
                self.assertEqual(main([
                    "--models-file", str(models), "--image", str(image),
                    "--output-root", directory, "--group-size", "64", "--ratio", "0.8",
                ]), 1)
            self.assertEqual(run.call_count, 2)
            command = run.call_args_list[1].args[0]
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(command[command.index("--model-id") + 1], "org/Second")
            self.assertEqual(command[command.index("--output-dir") + 1],
                             str(Path(directory) / "Second-int4-hf-export"))
            self.assertEqual(command[command.index("--group-size") + 1], "64")
            self.assertEqual(command[command.index("--ratio") + 1], "0.8")

    def test_single_model_explicit_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.jpg"
            image.touch()
            with patch("hf_export_multimodels_int4.export_model") as export:
                self.assertEqual(main([
                    "--model-id", "org/Model", "--image", str(image),
                    "--output-dir", directory,
                ]), 0)
            export.assert_called_once_with("org/Model", Path(directory), image,
                                           group_size=None, ratio=None, int4_config=None)

    def test_invalid_export_plans(self):
        for options in (
            ["--model-id", "org/Model", "other/model"],
            ["--model-id", "org/First", "org/Second", "--output-dir", "output"],
            ["--model-id", "org/Model", "--ratio", "1.1"],
            ["--model-id", "org/.."],
        ):
            with self.subTest(options=options), self.assertRaises(SystemExit) as error:
                main([*options, "--image", "missing.jpg", "--dry-run"])
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()