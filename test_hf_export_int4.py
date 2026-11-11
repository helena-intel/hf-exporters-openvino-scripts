import runpy
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import openvino as ov
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Phi3Config, Phi3ForCausalLM, PreTrainedTokenizerFast
from int4_config import DEFAULT_CONFIG_PATH, make_compression_config, make_quantizer, resolve_int4_config


class CompressionConfigTests(unittest.TestCase):
    def test_upstream_phi_and_symmetric_defaults(self):
        self.assertEqual(len(json.loads(DEFAULT_CONFIG_PATH.read_text())["models"]), 78)
        recipe = resolve_int4_config("microsoft/Phi-4-mini-instruct")
        config = make_compression_config(recipe)
        self.assertEqual(config.group_size, 64)
        self.assertTrue(config.awq)
        self.assertEqual(config.mode.value, "int4_asym")
        symmetric = make_compression_config(resolve_int4_config("Qwen/Qwen3-4B"))
        self.assertEqual(symmetric.mode.value, "int4_sym")

    def test_gemma_component_configs_and_calibration_guard(self):
        decoder = resolve_int4_config("google/gemma-4-E2B-it", component="decoder")
        self.assertEqual(decoder["dataset"], "textvqa")
        self.assertTrue(decoder["scale_estimation"])
        self.assertEqual(decoder["group_size"], 64)
        with self.assertRaisesRegex(ValueError, "representative calibration"):
            make_quantizer(decoder)
        for component in ("vision", "embeddings"):
            recipe = resolve_int4_config("google/gemma-4-E2B-it", component=component, group_size=32)
            self.assertEqual(make_compression_config(recipe).mode.value, "int8_sym")
            self.assertNotIn("group_size", recipe)

    def test_custom_file_overrides_and_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipes.json"
            path.write_text(json.dumps({"models": {"org/Model": {
                "bits": 4, "sym": True, "group_size": 64, "ratio": 0.8,
            }}}), encoding="utf-8")
            recipe = resolve_int4_config("org/Model", path, group_size=32, ratio=0.6)
            self.assertEqual((recipe["group_size"], recipe["ratio"], recipe["sym"]), (32, 0.6, True))
            self.assertEqual(resolve_int4_config("Model", path)["group_size"], 64)
            model_path = Path(directory) / "checkpoint"
            model_path.mkdir()
            (model_path / "config.json").write_text(json.dumps({"_name_or_path": "org/Model"}))
            self.assertTrue(resolve_int4_config(model_path, path)["sym"])
            self.assertEqual(resolve_int4_config("org/Unknown", path)["group_size"], 128)
            self.assertIsNone(resolve_int4_config("org/Unknown", path, component="embeddings"))
            missing = Path(directory) / "missing.json"
            with self.assertRaises(FileNotFoundError):
                resolve_int4_config("org/Model", missing)
            with patch("int4_config.DEFAULT_CONFIG_PATH", missing):
                self.assertEqual(resolve_int4_config("org/Unknown")["group_size"], 128)

    def test_data_free_awq_does_not_use_export_example(self):
        recipe = resolve_int4_config("microsoft/Phi-4-mini-instruct")
        quantizer = make_quantizer(recipe)
        with patch("nncf.compress_weights", return_value="compressed") as compress:
            self.assertEqual(quantizer.quantize("model", {"dummy": "sample"}, None), "compressed")
        self.assertIsNone(compress.call_args.kwargs["dataset"])
        self.assertTrue(compress.call_args.kwargs["awq"])
        self.assertEqual(compress.call_args.kwargs["group_size"], 64)

    def test_preprocessed_calibration_feeds_and_missing_inputs(self):
        recipe = {"bits": 4, "sym": False, "group_size": 32,
                  "dataset": "textvqa", "scale_estimation": True}
        parameter = ov.opset13.parameter([1, 32], name="inputs")
        graph = ov.Model([parameter], [parameter])
        graph.inputs[0].get_tensor().set_names({"inputs"})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No calibration"):
                make_quantizer(recipe, directory)
            sample_path = Path(directory) / "sample.npz"
            np.savez(sample_path, inputs=np.ones((1, 32), np.float32))
            with patch("nncf.compress_weights", return_value=graph) as compress:
                make_quantizer(recipe, directory).quantize(graph, {}, None)
                samples = list(compress.call_args.kwargs["dataset"].get_inference_data())
            self.assertEqual(samples[0]["inputs"].shape, (1, 32))
            np.savez(sample_path, wrong=np.ones((1, 32), np.float32))
            with patch("nncf.compress_weights", return_value=graph) as compress:
                make_quantizer(recipe, directory).quantize(graph, {}, None)
                with self.assertRaisesRegex(ValueError, "missing inputs"):
                    list(compress.call_args.kwargs["dataset"].get_inference_data())

    def test_calibration_provider_is_used_instead_of_export_example(self):
        from unittest.mock import Mock

        recipe = {"bits": 4, "sym": True, "group_size": 32, "dataset": "textvqa", "scale_estimation": True}
        sample = {"inputs": np.ones((1, 32), np.float32)}
        provider = Mock(return_value=[sample])
        quantizer = make_quantizer(recipe, calibration_samples=provider)
        with patch("nncf.compress_weights", return_value="compressed") as compress:
            quantizer.quantize("graph", {"export": "example"}, None)
            actual = list(compress.call_args.kwargs["dataset"].get_inference_data())
        provider.assert_called_once_with("graph", {"export": "example"})
        np.testing.assert_array_equal(actual[0]["inputs"], sample["inputs"])
        with self.assertRaisesRegex(ValueError, "either"):
            make_quantizer(recipe, "samples", calibration_samples=provider)

    def test_unsupported_recipes_are_not_silently_approximated(self):
        with self.assertRaisesRegex(ValueError, "Unsupported compression settings"):
            resolve_int4_config("openai/gpt-oss-20b")

    def test_text_calibration_selects_recipe_dataset_and_closes_stream(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from int4_config import text_calibration_samples

        for name, source, subset in (("wikitext2", "Salesforce/wikitext", "wikitext-2-raw-v1"),
                                     ("gsm8k", "openai/gsm8k", "main")):
            with self.subTest(dataset=name):
                events = []

                def rows():
                    try:
                        for index in range(4):
                            events.append(index)
                            yield {"text": f"Text {index}", "question": "Question", "answer": "Answer"}
                    finally:
                        events.append("closed")

                def shuffle(seed, max_buffer_input_shards=None):
                    self.assertEqual((seed, max_buffer_input_shards), (17, 1))
                    return rows()

                tokenizer = Mock(return_value={"input_ids": torch.tensor([[1, 2, 3]])})
                with patch("datasets.load_dataset", return_value=SimpleNamespace(shuffle=shuffle)) as load:
                    samples = list(text_calibration_samples(tokenizer, {"dataset": name, "num_samples": 2}, 17))
                self.assertEqual(len(samples), 2)
                self.assertEqual(events, [0, 1, "closed"])
                load.assert_called_once_with(source, subset, split="train", streaming=True)
                self.assertEqual(tokenizer.call_args.kwargs["max_length"], 128)

    def test_text_calibration_rejects_empty_data_and_closes_on_tokenizer_error(self):
        from unittest.mock import Mock
        from int4_config import text_calibration_samples

        tokenizer = Mock(side_effect=RuntimeError("tokenizer failed"))
        events = []

        def rows():
            try:
                yield {"text": "Representative text"}
            finally:
                events.append("closed")

        dataset = Mock()
        dataset.shuffle.return_value = rows()
        with patch("datasets.load_dataset", return_value=dataset):
            with self.assertRaisesRegex(RuntimeError, "tokenizer failed"):
                list(text_calibration_samples(tokenizer, {"dataset": "wikitext2"}))
        self.assertEqual(events, ["closed"])
        dataset.shuffle.return_value = iter([{"text": ""}])
        with patch("datasets.load_dataset", return_value=dataset):
            with self.assertRaisesRegex(ValueError, "no usable text"):
                list(text_calibration_samples(tokenizer, {"dataset": "wikitext2"}))

    def test_text_calibration_captures_decoder_feeds_and_cleans_spool(self):
        from unittest.mock import Mock
        from int4_config import prepare_text_calibration

        recipe = {"bits": 4, "dataset": "wikitext2", "num_samples": 2}
        exporter = Mock()
        exporter._resolve_generation_options.return_value = (None, True, True)
        events = []

        def samples(*args):
            try:
                for index in range(2):
                    yield {"inputs": torch.tensor([[index]], dtype=torch.float32)}
            finally:
                events.append("closed")

        parameter = ov.opset13.parameter([1, 1], name="inputs")
        graph = ov.Model([parameter], [parameter])
        graph.inputs[0].get_tensor().set_names({"inputs"})
        with patch("int4_config.text_calibration_samples", side_effect=samples), \
                patch("transformers.exporters.quantizers.utils.capture_calibration_inputs",
                      side_effect=lambda model, samples, **kwargs: {"model": [samples[0], samples[0]]}), \
                patch("transformers.exporters.precompute.precompute_export_inputs", side_effect=lambda config, inputs: inputs):
            with prepare_text_calibration(Mock(), Mock(), recipe, exporter=exporter, export_configs={}) as providers:
                feeds = list(providers["model"](graph, {}))
        self.assertEqual([float(feed["inputs"][0, 0]) for feed in feeds], [0, 1])
        self.assertEqual(events, ["closed"])
        self.assertEqual(make_compression_config({"bits": 4, "dataset": "wikitext2"}).subset_size, 128)

    def test_text_calibration_bf16_values_for_fp32_graph(self):
        from unittest.mock import Mock
        from int4_config import prepare_text_calibration

        recipe = {"bits": 4, "dataset": "wikitext2", "num_samples": 1}
        exporter = Mock()
        exporter._resolve_generation_options.return_value = (None, True, True)
        activation = torch.tensor([[1.5, -2.25]], dtype=torch.bfloat16)
        token_ids = torch.tensor([[1, 2]], dtype=torch.long)

        def samples(*args):
            yield {"input_ids": token_ids}

        for precision in (ov.Type.f32, ov.Type.bf16):
            for input_name in ("inputs", "input.inputs"):
                with self.subTest(precision=precision, input_name=input_name):
                    parameter = ov.opset13.parameter([1, 2], dtype=precision, name=input_name)
                    parameter.output(0).get_tensor().set_names({input_name})
                    ids_parameter = ov.opset13.parameter([1, 2], dtype=ov.Type.i64, name="input_ids")
                    ids_parameter.output(0).get_tensor().set_names({"input_ids"})
                    graph = ov.Model([ov.opset13.convert(parameter, ov.Type.f32), ids_parameter],
                                     [parameter, ids_parameter])
                    with patch("int4_config.text_calibration_samples", side_effect=samples), \
                            patch("transformers.exporters.quantizers.utils.capture_calibration_inputs",
                                  return_value={"model": [{"inputs": activation, "input_ids": token_ids}]}), \
                            patch("transformers.exporters.precompute.precompute_export_inputs",
                                  side_effect=lambda config, inputs: inputs):
                        with prepare_text_calibration(
                            Mock(), Mock(), recipe, exporter=exporter, export_configs={},
                        ) as providers:
                            feeds = list(providers["model"](graph, {}))
                    compiled = ov.Core().compile_model(graph, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
                    actual = compiled.create_infer_request().infer(feeds[0], share_inputs=True)
                    np.testing.assert_array_equal(actual[0], activation.float().numpy())
                    np.testing.assert_array_equal(actual[1], token_ids.numpy())
                    self.assertEqual(feeds[0]["input_ids"].dtype, np.int64)
        self.assertEqual(activation.dtype, torch.bfloat16)

    def test_text_calibration_preserves_explicit_mask_only_when_requested(self):
        from unittest.mock import Mock
        from int4_config import prepare_text_calibration

        recipe = {"bits": 4, "dataset": "wikitext2", "num_samples": 1}
        original_mask = torch.tensor([[0, 1, 1]], dtype=torch.long)
        transformed_mask = torch.zeros((1, 1, 3, 3), dtype=torch.float32)
        for captured_mask in (None, transformed_mask):
            for preserve in (False, True):
                with self.subTest(captured_mask=captured_mask is not None, preserve=preserve):
                    exporter = Mock()
                    exporter._resolve_generation_options.return_value = (None, True, True)
                    captured = {"input_ids": torch.tensor([[0, 2, 3]]), "attention_mask": captured_mask}
                    expected = original_mask if preserve else captured_mask
                    parameter = ov.opset13.parameter(
                        list(expected.shape) if expected is not None else [1, 3],
                        dtype=ov.Type.i64 if expected is None or expected.dtype == torch.long else ov.Type.f32,
                        name="attention_mask",
                    )
                    graph = ov.Model([parameter], [parameter])
                    graph.inputs[0].get_tensor().set_names({"attention_mask"})

                    def samples(*args):
                        yield {"input_ids": captured["input_ids"], "attention_mask": original_mask}

                    with patch("int4_config.text_calibration_samples", side_effect=samples), \
                            patch("transformers.exporters.quantizers.utils.capture_calibration_inputs",
                                  return_value={"model": [captured, captured]}), \
                            patch("transformers.exporters.precompute.precompute_export_inputs",
                                  side_effect=lambda config, inputs: inputs):
                        with prepare_text_calibration(
                            Mock(), Mock(), recipe, exporter=exporter, export_configs={},
                            preserve_attention_mask=preserve,
                        ) as providers:
                            if expected is None:
                                with self.assertRaisesRegex(ValueError, "missing inputs.*attention_mask"):
                                    list(providers["model"](graph, {}))
                            else:
                                feeds = list(providers["model"](graph, {}))
                                np.testing.assert_array_equal(feeds[0]["attention_mask"], expected.numpy())
                    self.assertIs(captured["attention_mask"], captured_mask)

    def test_qwen3_genai_export_with_text_calibration(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        config = Qwen3Config(
            vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=64,
            max_position_embeddings=256, bos_token_id=1, eos_token_id=2, pad_token_id=0,
        )
        config._attn_implementation = "sdpa"
        model = Qwen3ForCausalLM(config).eval()
        backend = Tokenizer(WordLevel({
            "[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3,
            "Hello": 4, ",": 5, "world": 6, "!": 7,
        }, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]",
            eos_token="[EOS]", unk_token="[UNK]",
        )
        recipe = {"bits": 4, "sym": False, "group_size": 32, "ratio": 1.0,
                  "dataset": "wikitext2", "num_samples": 2, "awq": True}

        def samples(*args):
            for text in ("Hello world!", "Hello world! Hello world!"):
                yield dict(tokenizer(text, return_tensors="pt"))

        with tempfile.TemporaryDirectory() as directory:
            arguments = ["hf_export_int4.py", "--model-id", "local/tiny-qwen3", "--output-dir", directory]
            with patch.object(sys, "argv", arguments), \
                    patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model), \
                    patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                    patch("int4_config.resolve_int4_config", return_value=recipe), \
                    patch("int4_config.text_calibration_samples", side_effect=samples):
                runpy.run_path(str(Path(__file__).with_name("hf_export_int4.py")), run_name="__main__")
            graph = ov.Core().read_model(Path(directory) / "openvino_model.xml")
            self.assertTrue(graph.get_variables())
            self.assertEqual(graph.input("attention_mask").get_partial_shape().rank.get_length(), 2)
            self.assertTrue(any(
                node.get_type_name() == "Constant" and node.get_output_element_type(0) in (ov.Type.i4, ov.Type.u4)
                for node in graph.get_ops()
            ))
            compiled = ov.Core().compile_model(graph, "CPU")
            logits = compiled({
                "input_ids": np.array([[4, 6, 7]], dtype=np.int64),
                "position_ids": np.array([[0, 1, 2]], dtype=np.int64),
                "attention_mask": np.ones((1, 3), dtype=np.int64),
                "beam_idx": np.array([0], dtype=np.int32),
            })["logits"]
            self.assertEqual(logits.shape, (1, 3, 128))
            self.assertTrue(np.isfinite(logits).all())

    def test_llama_genai_cache_batch_is_dynamic(self):
        import contextlib
        import io
        import openvino_genai as ov_genai
        from transformers import LlamaConfig, LlamaForCausalLM
        from llm_prompt import disable_genai_thinking

        config = LlamaConfig(
            vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
        )
        config._attn_implementation = "sdpa"
        model = LlamaForCausalLM(config).eval()
        backend = Tokenizer(WordLevel({
            "[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3,
            "Hello": 4, ",": 5, "world": 6, "!": 7,
        }, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]",
            eos_token="[EOS]", unk_token="[UNK]",
            chat_template="{% for message in messages %}{{ message['content'] }}{% endfor %}"
                          "{% if enable_thinking is defined and not enable_thinking %} Hello{% endif %}",
        )
        with tempfile.TemporaryDirectory() as directory:
            arguments = ["hf_export_int4.py", "--model-id", "local/tiny-llama", "--output-dir", directory]
            with patch.object(sys, "argv", arguments), \
                    patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model), \
                    patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                    patch("int4_config.make_quantizer", return_value=None):
                runpy.run_path(str(Path(__file__).with_name("hf_export_int4.py")), run_name="__main__")
            graph = ov.Core().read_model(Path(directory) / "openvino_model.xml")
            self.assertTrue(graph.get_variables())
            for variable in graph.get_variables():
                self.assertTrue(variable.get_info().data_shape[0].is_dynamic,
                                str(variable.get_info().data_shape))
            pipe = ov_genai.LLMPipeline(directory, "CPU")
            native_tokenizer = pipe.get_tokenizer()
            disable_genai_thinking(native_tokenizer, directory)
            text = native_tokenizer.apply_chat_template(
                [{"role": "user", "content": "Hello world!"}], add_generation_prompt=True,
            )
            self.assertEqual(text, "Hello world! Hello")
            results = pipe.generate(["Hello", "Hello world!"], max_new_tokens=2,
                                    do_sample=False, apply_chat_template=False, ignore_eos=True)
            self.assertEqual(len(results.texts), 2)
            output = io.StringIO()
            benchmark_arguments = ["benchmark_llm_genai.py", "-m", directory, "-p", "Hello world!",
                                   "-N", "0", "-n", "1", "-M", "2"]
            with patch.object(sys, "argv", benchmark_arguments), contextlib.redirect_stdout(output):
                runpy.run_path(str(Path(__file__).with_name("benchmark_llm_genai.py")), run_name="__main__")
            self.assertIn("enable_thinking=False requested", output.getvalue())
            self.assertIn("TTFT:", output.getvalue())

    def test_textvqa_stream_limits_and_closes_on_success_and_error(self):
        from unittest.mock import Mock
        from PIL import Image
        from int4_config import textvqa_samples

        for fail in (False, True):
            with self.subTest(fail=fail):
                events = []

                def rows():
                    try:
                        for index in range(3):
                            events.append(index)
                            yield {"image": Image.new("L", (8, 8)), "question": f"Question {index}"}
                    finally:
                        events.append("closed")

                processor = Mock(return_value={"input_ids": torch.tensor([[42]])})
                processor.apply_chat_template.return_value = "formatted question"
                if fail:
                    processor.side_effect = RuntimeError("processor failed")
                dataset = Mock()
                dataset.shuffle = Mock(return_value=rows())
                with patch("datasets.load_dataset", return_value=dataset) as load:
                    samples = textvqa_samples(processor, 2, 17)
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "processor failed"):
                            list(samples)
                        self.assertEqual(events, [0, "closed"])
                    else:
                        self.assertEqual(len(list(samples)), 2)
                        self.assertEqual(events, [0, 1, "closed"])
                        self.assertEqual(processor.call_args.kwargs["images"].mode, "RGB")
                    load.assert_called_once_with("lmms-lab/textvqa", split="validation", streaming=True)
                    dataset.shuffle.assert_called_once_with(seed=17)

    def test_text_launcher_only_forwards_explicit_overrides(self):
        from unittest.mock import Mock

        with tempfile.TemporaryDirectory() as directory:
            models = Path(directory) / "models.txt"
            models.write_text("microsoft/Phi-4-mini-instruct\n")
            for overrides in ([], ["--group-size", "32", "--ratio", "0.8",
                                "--int4-config", "custom.json", "--calibration-data", "calibration"],
                            ["--calibration-dataset", "textvqa", "--calibration-num-samples", "2",
                             "--calibration-seed", "17"]):
                arguments = ["hf_export_models_int4.py", "--models-file", str(models), *overrides]
                with patch.object(sys, "argv", arguments), \
                        patch("subprocess.run", return_value=Mock(returncode=0)) as run, \
                        self.assertRaises(SystemExit) as result:
                    runpy.run_path(str(Path(__file__).with_name("hf_export_models_int4.py")), run_name="__main__")
                self.assertEqual(result.exception.code, 0)
                command = run.call_args.args[0]
                if overrides:
                    self.assertEqual(command[-len(overrides):], overrides)
                else:
                    self.assertNotIn("--group-size", command)
                    self.assertNotIn("--ratio", command)

    def test_textvqa_empty_stream_uses_single_shard_shuffle(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from int4_config import textvqa_samples

        calls = []

        def shuffle(seed, max_buffer_input_shards=None):
            calls.append((seed, max_buffer_input_shards))
            return iter(())

        with patch("datasets.load_dataset", return_value=SimpleNamespace(shuffle=shuffle)):
            with self.assertRaisesRegex(ValueError, "empty"):
                list(textvqa_samples(Mock(), 2, 17))
        self.assertEqual(calls, [(17, 1)])

    def test_textvqa_default_count_and_one_feed_per_image(self):
        from unittest.mock import Mock
        from int4_config import prepare_textvqa_calibration

        recipe = resolve_int4_config("google/gemma-4-E4B-it", component="decoder")
        self.assertEqual(make_compression_config(recipe).subset_size, 32)
        self.assertEqual(make_compression_config({**recipe, "num_samples": 7}).subset_size, 7)
        for limit in (None, 3):
            with self.subTest(limit=limit):
                effective = recipe if limit is None else {**recipe, "num_samples": limit}
                count = 32 if limit is None else limit
                exporter = Mock()
                exporter._resolve_generation_options.return_value = (None, True, True)
                model = Mock()
                model.config = self

                def samples(processor, num_samples, seed):
                    self.assertEqual(num_samples, count)
                    for index in range(num_samples):
                        yield {"inputs": torch.tensor([[index]], dtype=torch.float32)}

                def capture(model, samples, **kwargs):
                    return {"language_model": [{"inputs": torch.tensor([[-1.]])}, samples[0]]}

                parameter = ov.opset13.parameter([1, 1], name="inputs")
                graph = ov.Model([parameter], [parameter])
                graph.inputs[0].get_tensor().set_names({"inputs"})
                with patch("int4_config.textvqa_samples", side_effect=samples), \
                        patch("transformers.exporters.quantizers.utils.capture_calibration_inputs", side_effect=capture), \
                        patch("transformers.exporters.precompute.precompute_export_inputs", side_effect=lambda config, inputs: inputs):
                    with prepare_textvqa_calibration(model, None, {"language_model": effective}, exporter=exporter,
                                                    export_configs={}) as providers:
                        feeds = list(providers["language_model"](graph, {}))
                self.assertEqual(len(feeds), count)
                self.assertEqual([float(feed["inputs"][0, 0]) for feed in feeds], list(range(count)))

    def test_textvqa_capture_failure_closes_stream_and_removes_spool(self):
        from unittest.mock import Mock
        from int4_config import prepare_textvqa_calibration

        events = []

        def samples(*args):
            try:
                yield {"input_ids": torch.tensor([[42]])}
            finally:
                events.append("closed")

        with tempfile.TemporaryDirectory() as parent:
            exporter = Mock()
            exporter._resolve_generation_options.return_value = (None, True, True)
            temporary_directory = tempfile.TemporaryDirectory
            with patch("int4_config.tempfile.TemporaryDirectory",
                       side_effect=lambda **kwargs: temporary_directory(dir=parent)) as temporary, \
                    patch("int4_config.textvqa_samples", side_effect=samples), \
                    patch("transformers.exporters.quantizers.utils.capture_calibration_inputs",
                          side_effect=RuntimeError("capture failed")):
                with self.assertRaisesRegex(RuntimeError, "capture failed"):
                    with prepare_textvqa_calibration(Mock(), Mock(), {"language_model": {"bits": 4}},
                                                    exporter=exporter, export_configs={}):
                        self.fail("Capture must fail before calibration is used")
            self.assertEqual(events, ["closed"])
            temporary.assert_called_once()
            self.assertEqual(list(Path(parent).iterdir()), [])


class BashBenchmarkRunnerTests(unittest.TestCase):
    def setUp(self):
        if sys.platform == "win32":
            self.bash = next((str(path) for path in (
                Path("C:/Program Files/Git/bin/bash.exe"), Path("C:/Program Files/Git/usr/bin/bash.exe"),
            ) if path.is_file()), None)
        else:
            self.bash = shutil.which("bash")
        if self.bash is None:
            self.skipTest("Bash is required")
        temporary = tempfile.TemporaryDirectory(prefix="vlm-benchmark-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.models = self.root / "models.txt"
        self.models.write_text("\ufeff # Models\r\n\n org/Good, model # fixture\r\norg/Failure\r\norg/Missing",
                               encoding="utf-8")
        self.image = self.root / "image.jpg"
        self.image.touch()
        for name in ("Good, model", "Failure"):
            for suffix in ("-int4-optimum", "-int4-hf-export"):
                (self.root / (name + suffix)).mkdir()
        self.fixture = self.root / "fake_benchmark.py"
        self.fixture.write_text(r"""#!/usr/bin/env bash
set -eu
if IFS= read -r unexpected; then
    printf 'Benchmark inherited model-list stdin\n' >&2
    exit 9
fi
model=''
budget=''
while (( $# )); do
    case "$1" in
        -m) model=$2; shift 2 ;;
        -M) budget=$2; shift 2 ;;
        *) shift ;;
    esac
done
printf 'stderr marker\n' >&2
case "$model" in
    *Failure-int4-optimum) printf 'failure marker\n'; exit 7 ;;
esac
printf 'Input token size: 100\nOutput token size: %s\n' "$budget"
case "$model" in
    *Failure-int4-hf-export) printf 'TTFT: 600.00 +/- 2.00 ms\n'; exit 0 ;;
    *-int4-optimum) separator=$(printf '\302\261'); ttft=600.00 ;;
    *) separator='+/-'; ttft=2600.00 ;;
esac
printf 'TTFT: %s %s 2.00 ms\n' "$ttft" "$separator"
printf 'Generate time: 3000.00 %s 3.00 ms\n' "$separator"
if [[ $budget == 1 ]]; then
    printf 'TPOT: N/A (fewer than two output tokens)\nThroughput: N/A (fewer than two output tokens)\n'
else
    printf 'TPOT: 27.00 %s 0.30 ms/token\nThroughput: 37.00 %s 0.40 tokens/s\n' "$separator" "$separator"
fi
""", encoding="utf-8", newline="\n")

    def run_batch(self, *extra):
        script = Path(__file__).with_name("run_vlm_benchmarks.sh")
        return subprocess.run([
            self.bash, script.as_posix(), "--models-file", self.models.as_posix(),
            "--model-root", self.root.as_posix(), "--image", self.image.as_posix(),
            "--output-dir", (self.root / "results directory").as_posix(), "--python", Path(self.bash).as_posix(),
            "--hf-script", self.fixture.as_posix(), "--genai-script", self.fixture.as_posix(),
            "--prompt", 'caption, with "quotes"', "-N", "3", "-n", "10", "-M", "4", "--profile", *extra,
        ], capture_output=True, encoding="utf-8", errors="replace", timeout=60)

    def test_batch_logs_metrics_and_continues_after_failures(self):
        result = self.run_batch()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        summary = self.root / "results directory" / "summary.csv"
        with summary.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual([row["status"] for row in rows],
                         ["ok", "ok", "failed", "missing_metrics", "missing_model", "missing_model"])
        self.assertEqual(rows[0]["model_id"], "org/Good, model")
        for row, ttft in zip(rows[:2], ("600.00", "2600.00")):
            self.assertEqual(row["ttft_ms"], ttft)
            self.assertEqual(row["ttft_std_ms"], "2.00")
            self.assertEqual(row["tpot_ms"], "27.00")
            self.assertEqual(row["tpot_std_ms"], "0.30")
            self.assertEqual(row["throughput_tokens_per_s"], "37.00")
            self.assertEqual(row["throughput_std_tokens_per_s"], "0.40")
            self.assertEqual(row["generation_ms"], "3000.00")
            self.assertEqual(row["generation_std_ms"], "3.00")
            self.assertEqual(row["output_tokens"], "4")
        logs = list((summary.parent / "logs").glob("*.log"))
        self.assertEqual(len(logs), 6)
        self.assertTrue(any("failure marker" in log.read_text(encoding="utf-8") for log in logs))
        for log in logs[:2]:
            self.assertIn("stderr marker", log.read_text(encoding="utf-8"))
        self.assertEqual(rows[2]["exit_code"], "7")
        self.assertEqual(rows[3]["exit_code"], "0")
        saved = summary.read_bytes()
        rerun = self.run_batch()
        self.assertEqual(rerun.returncode, 2)
        self.assertEqual(summary.read_bytes(), saved)

    def test_one_token_metrics_and_invalid_count(self):
        self.models.write_text("org/Good, model\n", encoding="utf-8")
        invalid = self.run_batch("-n", "0")
        self.assertEqual(invalid.returncode, 2)
        result = self.run_batch("-M", "1")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with (self.root / "results directory" / "summary.csv").open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["output_tokens"], "1")
            self.assertEqual(row["tpot_ms"], "N/A")
            self.assertEqual(row["tpot_std_ms"], "")
            self.assertEqual(row["throughput_tokens_per_s"], "N/A")


class BashLLMBenchmarkRunnerTests(BashBenchmarkRunnerTests):
    def setUp(self):
        super().setUp()
        for name in ("Good, model", "Failure"):
            for suffix in ("-int4-hf-export-genai", "-hf-export"):
                (self.root / (name + suffix)).mkdir()
        self.fixture.write_text(r"""#!/usr/bin/env bash
set -eu
if IFS= read -r unexpected; then
    printf 'Benchmark inherited model-list stdin\n' >&2
    exit 9
fi
model=''
budget=''
mode=''
prompt=''
while (( $# )); do
    case "$1" in
        -m) model=$2; shift 2 ;;
        -M) budget=$2; shift 2 ;;
        -p) prompt=$2; shift 2 ;;
        --disable-thinking|--enable-thinking|--no-chat-template) mode=$1; shift ;;
        -i|--profile) exit 12 ;;
        *) shift ;;
    esac
done
printf 'stderr marker\nMode: %s\n' "$mode" >&2
printf 'Prompt: %s\n' "$prompt"
case "$model" in
    *Failure-int4-optimum) printf 'failure marker\n'; exit 7 ;;
    *Failure-int4-hf-export-genai) printf 'TTFT: 600.00 +/- 2.00 ms\n'; exit 0 ;;
esac
case "$model" in
    *-hf-export) input_label=Input; throughput_label=Throughput:; ttft=2600.00; separator='+/-' ;;
    *) input_label=Prompt; throughput_label='Throughput :'; ttft=600.00; separator=$(printf '\302\261') ;;
esac
printf '%s token size: 100\nOutput token size: %s\n' "$input_label" "$budget"
printf 'TTFT: %s %s 2.00 ms\nGenerate time: 3000.00 %s 3.00 ms\n' "$ttft" "$separator" "$separator"
if [[ $budget == 1 ]]; then
    printf 'TPOT: N/A (fewer than two output tokens)\n%s N/A (fewer than two output tokens)\n' "$throughput_label"
else
    printf 'TPOT: 27.00 %s 0.30 ms/token\n%s 37.00 %s 0.40 tokens/s\n' "$separator" "$throughput_label" "$separator"
fi
""", encoding="utf-8", newline="\n")

    def run_batch(self, *extra):
        script = Path(__file__).with_name("run_llm_benchmarks.sh")
        return subprocess.run([
            self.bash, script.as_posix(), "--models-file", self.models.as_posix(),
            "--model-root", self.root.as_posix(),
            "--output-dir", (self.root / "results directory").as_posix(), "--python", Path(self.bash).as_posix(),
            "--hf-script", self.fixture.as_posix(), "--genai-script", self.fixture.as_posix(),
            "--prompt", 'explain, with "quotes"', "-N", "3", "-n", "10", "-M", "4", *extra,
        ], capture_output=True, encoding="utf-8", errors="replace", timeout=60)

    def test_batch_logs_metrics_and_continues_after_failures(self):
        result = self.run_batch()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        summary = self.root / "results directory" / "summary.csv"
        with summary.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual([row["status"] for row in rows],
                         ["ok", "ok", "ok", "failed", "missing_metrics", "ok",
                          "missing_model", "missing_model", "missing_model"])
        self.assertEqual([row["backend"] for row in rows[:3]],
                         ["optimum", "hf-exporters-genai", "hf-exporters"])
        for row, suffix, ttft in zip(rows[:3], ("-int4-optimum", "-int4-hf-export-genai", "-hf-export"),
                                     ("600.00", "600.00", "2600.00")):
            self.assertEqual(row["model_id"], "org/Good, model")
            self.assertTrue(row["model_dir"].endswith("Good, model" + suffix))
            self.assertEqual(row["input_tokens"], "100")
            self.assertEqual(row["output_tokens"], "4")
            self.assertEqual(row["ttft_ms"], ttft)
            self.assertEqual(row["ttft_std_ms"], "2.00")
            self.assertEqual(row["tpot_ms"], "27.00")
            self.assertEqual(row["tpot_std_ms"], "0.30")
            self.assertEqual(row["throughput_tokens_per_s"], "37.00")
            self.assertEqual(row["throughput_std_tokens_per_s"], "0.40")
            self.assertEqual(row["generation_ms"], "3000.00")
            self.assertEqual(row["generation_std_ms"], "3.00")
        self.assertEqual(rows[3]["exit_code"], "7")
        logs = list((summary.parent / "logs").glob("*.log"))
        self.assertEqual(len(logs), 9)
        successful_logs = [log for log in logs if "Good, model" in log.name]
        for log in successful_logs:
            text = log.read_text(encoding="utf-8")
            self.assertIn("stderr marker", text)
            self.assertIn("Mode: --disable-thinking", text)
            self.assertIn('Prompt: explain, with "quotes"', text)
        saved = summary.read_bytes()
        self.assertEqual(self.run_batch().returncode, 2)
        self.assertEqual(summary.read_bytes(), saved)

    def test_one_token_metrics_and_invalid_count(self):
        self.models.write_text("org/Good, model\n", encoding="utf-8")
        self.assertEqual(self.run_batch("-n", "0").returncode, 2)
        result = self.run_batch("-M", "1", "--no-chat-template")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = self.root / "results directory" / "summary.csv"
        with summary.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["output_tokens"], "1")
            self.assertEqual(row["tpot_ms"], "N/A")
            self.assertEqual(row["tpot_std_ms"], "")
            self.assertEqual(row["throughput_tokens_per_s"], "N/A")
        for log in (summary.parent / "logs").glob("*.log"):
            self.assertIn("Mode: --no-chat-template", log.read_text(encoding="utf-8"))


class BenchmarkTokenSelectionTests(unittest.TestCase):
    def test_indexed_eos_suppression_matches_minimum_length(self):
        from benchmark_vlm_hf_exporters import SuppressEosLogitsProcessor
        from transformers.generation.logits_process import MinNewTokensLengthLogitsProcessor

        scores = torch.randn(1, 248320)
        original = scores.clone()
        for eos_ids in (7, [7, 42], torch.tensor([7, 42])):
            with self.subTest(eos_ids=eos_ids):
                indexed = SuppressEosLogitsProcessor(eos_ids)
                reference = MinNewTokensLengthLogitsProcessor(5, 3, eos_ids)
                for generated in (0, 2):
                    input_ids = torch.ones(1, 5 + generated, dtype=torch.long)
                    torch.testing.assert_close(indexed(input_ids, scores), reference(input_ids, scores))
                torch.testing.assert_close(scores, original)


class UnifiedExportTests(unittest.TestCase):
    def tokenizer(self):
        vocabulary = {f"token_{index}": index for index in range(256)}
        for index, token in enumerate(("[PAD]", "[BOS]", "[EOS]", "[UNK]", "Hello")):
            vocabulary.pop(f"token_{index}")
            vocabulary[token] = index
        vocabulary.pop("token_255")
        vocabulary["<image>"] = 255
        backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        return PreTrainedTokenizerFast(
            tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]",
            eos_token="[EOS]", unk_token="[UNK]", additional_special_tokens=["<image>"],
        )

    def llama_config(self):
        from transformers import LlamaConfig

        return LlamaConfig(
            vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
        )

    def test_native_llm_int4_save_load_and_generate(self):
        from transformers import LlamaForCausalLM
        from transformers.exporters import ExportArtifacts, ExportedModel, ExportedModelForCausalLM
        from hf_export_models_int4 import export_model

        torch.manual_seed(7)
        config = self.llama_config()
        model = LlamaForCausalLM(config).eval()
        tokenizer = self.tokenizer()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "export"
            with patch("transformers.AutoConfig.from_pretrained", return_value=config), \
                    patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model), \
                    patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
                export_model("local/tiny-llama", output, prompt="Hello Hello", group_size=32)
            manifest = json.loads((output / "export.json").read_text())
            self.assertTrue(manifest["generation"])
            self.assertEqual(set(manifest["components"]), {"model"})
            recipe = json.loads((output / "compression_config.json").read_text())
            self.assertEqual(recipe["components"]["model"]["group_size"], 32)
            graph = ExportArtifacts.from_pretrained(output)["model"].program
            self.assertTrue(graph.get_variables())
            self.assertTrue(any(node.get_element_type() == ov.Type.u4 for node in graph.get_ops()
                                if node.get_type_name() == "Constant"))
            loaded = ExportedModel.from_pretrained(output, device="cpu")
            self.assertIsInstance(loaded, ExportedModelForCausalLM)
            inputs = tokenizer("Hello Hello", return_tensors="pt")
            first = loaded.generate(**inputs, max_new_tokens=3, do_sample=False)
            second = loaded.generate(**inputs, max_new_tokens=3, do_sample=False)
            torch.testing.assert_close(first, second)
            self.assertGreater(first.shape[1], inputs.input_ids.shape[1])
            self.assertEqual(tokenizer.get_vocab(), type(tokenizer).from_pretrained(output).get_vocab())

    def test_native_vlm_save_load_image_generation(self):
        from PIL import Image
        from transformers import CLIPImageProcessor, CLIPVisionConfig, LlavaConfig, LlavaForConditionalGeneration, LlavaProcessor
        from transformers.exporters import ExportedModel, ExportedMultimodalModelForCausalLM
        from hf_export_models_int4 import export_model

        torch.manual_seed(7)
        config = LlavaConfig(
            text_config=self.llama_config(),
            vision_config=CLIPVisionConfig(
                hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                num_attention_heads=4, image_size=8, patch_size=4,
            ),
            image_token_index=255, image_seq_length=4, vision_feature_layer=-1,
        )
        config._attn_implementation = "sdpa"
        model = LlavaForConditionalGeneration(config).eval()
        processor = LlavaProcessor(
            tokenizer=self.tokenizer(),
            image_processor=CLIPImageProcessor(size={"shortest_edge": 8}, crop_size={"height": 8, "width": 8}),
            patch_size=4, num_additional_image_tokens=1, vision_feature_select_strategy="default",
            chat_template="{{ '<image> Hello' }}",
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "export"
            image = Path(directory) / "image.png"
            Image.new("RGB", (8, 8), "red").save(image)
            with patch("transformers.AutoConfig.from_pretrained", return_value=config), \
                    patch("transformers.AutoModelForImageTextToText.from_pretrained", return_value=model), \
                    patch("transformers.AutoProcessor.from_pretrained", return_value=processor), \
                    patch("int4_config.make_quantizer", return_value=None):
                export_model("local/tiny-llava", output, image=image)
            manifest = json.loads((output / "export.json").read_text())
            self.assertEqual(set(manifest["components"]), {
                "language_model", "text_embeddings_model", "vision_embeddings_model",
            })
            restored_processor = type(processor).from_pretrained(output)
            loaded = ExportedModel.from_pretrained(output, device="cpu")
            self.assertIsInstance(loaded, ExportedMultimodalModelForCausalLM)
            inputs = restored_processor(text="<image> Hello", images=Image.new("RGB", (8, 8), "red"), return_tensors="pt")
            with torch.no_grad():
                expected = model.generate(**inputs, max_new_tokens=3, do_sample=False)
            actual = loaded.generate(**inputs, max_new_tokens=3, do_sample=False)
            repeated = loaded.generate(**inputs, max_new_tokens=3, do_sample=False)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(repeated, actual)

            import contextlib
            import io
            from inference_vlm import generate_native, main as inference_main

            report = io.StringIO()
            with contextlib.redirect_stdout(report):
                inference_main([str(output), str(image), "--prompt", "Describe this image.",
                                "--device", "CPU", "--max-new-tokens", "3"])
            self.assertIn(restored_processor.tokenizer.decode(
                expected[0, inputs.input_ids.shape[1]:], skip_special_tokens=True,
            ), report.getvalue())
            self.assertIn('"ttft_s"', report.getvalue())
            with patch.object(loaded.generation_config, "eos_token_id", int(expected[0, inputs.input_ids.shape[1]])), \
                    patch("transformers.exporters.ExportArtifacts.runtime", return_value=loaded):
                stopped = generate_native(output, image, "Describe this image.", max_new_tokens=3)
            self.assertEqual(len(stopped["token_ids"]), 1)
            self.assertIsNone(stopped["timings"]["tpot_s"])

            from benchmark_vlm_hf_exporters import NativeOpenVINOVLM, benchmark, main

            runner = NativeOpenVINOVLM(output, processor=restored_processor,
                                      compile_config={"PERFORMANCE_HINT": "LATENCY"})
            processed = runner.preprocess(image, "Describe this image.")
            saved_ids = processed["input_ids"].clone()
            vision = runner.model.embedder.modality_runners[0][1].runner
            native_call = type(vision).__call__
            with patch.object(type(vision), "__call__", autospec=True, side_effect=native_call) as calls:
                results = benchmark(runner, processed, num_warmup=1, num_iter=2, max_new_tokens=3, profile=True)
            self.assertEqual(sum(call.args[0] is vision for call in calls.call_args_list), 3)
            torch.testing.assert_close(processed["input_ids"], saved_ids)
            for result in results:
                self.assertEqual(result["token_ids"], expected[0, inputs.input_ids.shape[1]:].tolist())
                timings = result["timings"]
                self.assertGreater(timings["vision_s"], 0)
                self.assertGreater(timings["ttft_s"], timings["embeddings_prepare_s"])
                if timings["decode_steps"]:
                    self.assertAlmostEqual(
                        timings["tpot_s"] * timings["decode_steps"],
                        sum(timings[name] for name in ("decode_decoder_call_s", "decode_embedding_call_s",
                                                       "decode_other_call_s", "decode_host_s")),
                    )
            single = runner.generate_processed(processed, max_new_tokens=1)
            self.assertEqual(len(single["token_ids"]), 1)
            self.assertIsNone(single["timings"]["tpot_s"])
            original_minimum = runner.model.generation_config.min_new_tokens
            with patch.object(runner.model.generation_config, "eos_token_id", single["token_ids"][0]):
                early = runner.model.generate(**processed, max_new_tokens=3, min_new_tokens=0, do_sample=False)
                self.assertEqual(early.shape[1] - processed["input_ids"].shape[1], 1)
                with patch("transformers.generation.logits_process.MinNewTokensLengthLogitsProcessor.__call__",
                           side_effect=AssertionError("Benchmark must not scan the vocabulary to suppress EOS")):
                    fixed = runner.generate_processed(processed, max_new_tokens=3, profile=True)
                self.assertEqual(len(fixed["token_ids"]), 3)
                self.assertEqual(fixed["timings"]["decode_steps"], 2)
            self.assertEqual(runner.model.generation_config.min_new_tokens, original_minimum)
            with tempfile.TemporaryDirectory() as legacy, self.assertRaisesRegex(ValueError, "export.json"):
                NativeOpenVINOVLM(legacy)
            import contextlib
            import io

            report = io.StringIO()
            with contextlib.redirect_stdout(report):
                self.assertEqual(main(["-m", str(output), "-i", str(image), "-N", "0", "-n", "1", "-M", "3",
                                       "--profile"]), 0)
            self.assertIn("native HF OpenVINO generation runtime", report.getvalue())
            self.assertIn("Decoder execution layer types:", report.getvalue())
            self.assertIn("Decode Decoder call:", report.getvalue())

            recipes = Path(directory) / "recipes.json"
            recipes.write_text(json.dumps({"models": {"local/tiny-llava": {"quantization_configs": {
                "lm_model": {"bits": 4, "sym": True, "group_size": 32},
                "vision_embeddings_model": {"bits": 8, "sym": True},
                "text_embeddings_model": {"bits": 8, "sym": True},
            }}}}))
            compressed_output = Path(directory) / "compressed"
            with patch("transformers.AutoConfig.from_pretrained", return_value=config), \
                    patch("transformers.AutoModelForImageTextToText.from_pretrained", return_value=model), \
                    patch("transformers.AutoProcessor.from_pretrained", return_value=processor):
                export_model("local/tiny-llava", compressed_output, image=image, int4_config=recipes)
            from transformers.exporters import ExportArtifacts

            compressed = ExportArtifacts.from_pretrained(compressed_output)
            for name, element_type in (("language_model", ov.Type.i4),
                                       ("text_embeddings_model", ov.Type.i8),
                                       ("vision_embeddings_model", ov.Type.i8)):
                self.assertTrue(any(node.get_element_type() == element_type
                                    for node in compressed[name].program.get_ops()
                                    if node.get_type_name() == "Constant"), name)
            loaded = compressed.runtime(device="cpu")
            actual = loaded.generate(**inputs, max_new_tokens=3, do_sample=False)
            repeated = loaded.generate(**inputs, max_new_tokens=3, do_sample=False)
            torch.testing.assert_close(actual, repeated)
            self.assertGreater(actual.shape[1], inputs.input_ids.shape[1])
            compressed_runner = NativeOpenVINOVLM(compressed_output, processor=restored_processor)
            compressed_result = compressed_runner.generate_processed(inputs, max_new_tokens=3, profile=True)
            self.assertEqual(compressed_result["token_ids"], actual[0, inputs.input_ids.shape[1]:].tolist())
            inference_result = generate_native(compressed_output, image, "Describe this image.", max_new_tokens=3)
            self.assertEqual(inference_result["token_ids"], actual[0, inputs.input_ids.shape[1]:].tolist())

    def test_native_vlm_default_textvqa_calibration_export_and_generate(self):
        from PIL import Image
        from transformers import CLIPImageProcessor, CLIPVisionConfig, LlavaConfig, LlavaForConditionalGeneration, LlavaProcessor
        from transformers.exporters import ExportArtifacts
        from hf_export_models_int4 import export_model

        torch.manual_seed(7)
        config = LlavaConfig(
            text_config=self.llama_config(),
            vision_config=CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                          num_attention_heads=4, image_size=8, patch_size=4),
            image_token_index=255, image_seq_length=4, vision_feature_layer=-1,
        )
        config._attn_implementation = "sdpa"
        model = LlavaForConditionalGeneration(config).eval()
        processor = LlavaProcessor(
            tokenizer=self.tokenizer(),
            image_processor=CLIPImageProcessor(size={"shortest_edge": 8}, crop_size={"height": 8, "width": 8}),
            patch_size=4, num_additional_image_tokens=1, vision_feature_select_strategy="default",
            chat_template="{{ '<image> ' + messages[0]['content'][1]['text'] }}",
        )
        events = []

        def samples(processor_arg, count, seed):
            self.assertIs(processor_arg, processor)
            self.assertEqual((count, seed), (2, 17))
            try:
                for color in ("blue", "green"):
                    events.append(color)
                    yield dict(processor(text="<image> Hello Hello", images=Image.new("RGB", (8, 8), color),
                                         return_tensors="pt"))
            finally:
                events.append("closed")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            Image.new("RGB", (8, 8), "red").save(image)
            recipes = root / "recipes.json"
            recipes.write_text(json.dumps({"models": {"local/tiny-llava": {"quantization_configs": {
                "lm_model": {"bits": 4, "sym": True, "group_size": 32, "scale_estimation": True,
                             "quant_method": "awq", "dataset": "textvqa", "num_samples": 2},
            }, "default_config": {"bits": 8, "sym": True}}}}))
            output = root / "export"
            with patch("transformers.AutoConfig.from_pretrained", return_value=config), \
                    patch("transformers.AutoModelForImageTextToText.from_pretrained", return_value=model), \
                    patch("transformers.AutoProcessor.from_pretrained", return_value=processor), \
                    patch("int4_config.textvqa_samples", side_effect=samples):
                export_model("local/tiny-llava", output, image=image, int4_config=recipes,
                             calibration_seed=17)
            self.assertEqual(events, ["blue", "green", "closed"])
            metadata = json.loads((output / "compression_config.json").read_text())
            self.assertEqual(metadata["calibration"], {"dataset": "textvqa", "seed": 17})
            self.assertEqual(metadata["components"]["language_model"]["num_samples"], 2)
            artifacts = ExportArtifacts.from_pretrained(output)
            graph = artifacts["language_model"].program
            self.assertTrue(graph.get_variables())
            self.assertTrue(any(node.get_element_type() == ov.Type.i4 for node in graph.get_ops()
                                if node.get_type_name() == "Constant"))
            inputs = processor(text="<image> Hello", images=Image.new("RGB", (8, 8), "blue"), return_tensors="pt")
            runtime = artifacts.runtime(device="cpu")
            first = runtime.generate(**inputs, max_new_tokens=2, do_sample=False)
            repeated = runtime.generate(**inputs, max_new_tokens=2, do_sample=False)
            torch.testing.assert_close(first, repeated)
            self.assertGreater(first.shape[1], inputs.input_ids.shape[1])

    def test_native_gemma_textvqa_calibration(self):
        from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig, Gemma4VisionConfig
        from transformers.exporters import OpenVINOConfig, OpenVINOExporter
        from int4_config import prepare_textvqa_calibration

        config = Gemma4Config(
            text_config=Gemma4TextConfig(
                vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=1, head_dim=64, global_head_dim=64,
                layer_types=["sliding_attention", "full_attention"], sliding_window=16,
                hidden_size_per_layer_input=16, vocab_size_per_layer_input=256, eos_token_id=1, pad_token_id=0,
                use_bidirectional_attention="vision",
            ),
            vision_config=Gemma4VisionConfig(
                hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                num_attention_heads=2, num_key_value_heads=2, head_dim=32,
                pooling_kernel_size=2, patch_size=2, position_embedding_size=64,
            ), audio_config=None, image_token_id=250,
        )
        config._attn_implementation = "sdpa"
        model = Gemma4ForConditionalGeneration(config).eval()
        inputs = {
            "input_ids": torch.tensor([[2, 3, 250, 250, 250, 250, 4, 5]]),
            "attention_mask": torch.ones((1, 8), dtype=torch.long),
            "pixel_values": torch.randn(1, 16, 12),
            "image_position_ids": torch.tensor([[[column, row] for row in range(4) for column in range(4)]]),
            "mm_token_type_ids": torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0]]),
        }
        recipe = {**resolve_int4_config("google/gemma-4-E4B-it", component="decoder"),
                  "group_size": -1, "num_samples": 2}
        configs = {name: OpenVINOConfig(dynamic=True, stateful=True) for name in (
            "language_model", "text_embeddings_model", "vision_embeddings_model",
        )}
        exporter = OpenVINOExporter()

        def samples(*args):
            yield {**inputs, "pixel_values": torch.ones_like(inputs["pixel_values"])}
            yield {**inputs, "pixel_values": torch.zeros_like(inputs["pixel_values"])}

        with patch("int4_config.textvqa_samples", side_effect=samples), torch.no_grad():
            with prepare_textvqa_calibration(model, None, {"language_model": recipe}, exporter=exporter,
                                            export_configs=configs) as providers:
                configs["language_model"].quantizer = make_quantizer(
                    recipe, calibration_samples=providers["language_model"],
                )
                artifacts = exporter.export_for_generation(model, inputs, configs)
        self.assertTrue(artifacts["language_model"].program.get_variables())
        self.assertTrue(any(node.get_element_type() == ov.Type.u4
                            for node in artifacts["language_model"].program.get_ops()
                            if node.get_type_name() == "Constant"))
        runtime = artifacts.runtime(device="cpu")
        first = runtime.generate(**inputs, max_new_tokens=2, do_sample=False)
        repeated = runtime.generate(**inputs, max_new_tokens=2, do_sample=False)
        torch.testing.assert_close(first, repeated)
        self.assertGreater(first.shape[1], inputs["input_ids"].shape[1])

    def test_missing_image_or_unsupported_dataset_fails_before_weight_loading(self):
        from transformers import Gemma4Config
        from hf_export_models_int4 import export_model

        with tempfile.TemporaryDirectory() as directory, \
                patch("transformers.AutoConfig.from_pretrained", return_value=Gemma4Config()), \
                patch("transformers.AutoModelForImageTextToText.from_pretrained") as load:
            with self.assertRaisesRegex(ValueError, "requires --image"):
                export_model("google/gemma-4-E2B-it", Path(directory) / "export")
            with patch("int4_config.resolve_int4_config", return_value={"bits": 4, "dataset": "unsupported"}):
                with self.assertRaisesRegex(ValueError, "Automatic calibration.*not supported"):
                    export_model("google/gemma-4-E2B-it", Path(directory) / "export", image="example.png")
            load.assert_not_called()

    def test_batch_forwards_image_and_preserves_failures(self):
        from unittest.mock import Mock
        from hf_export_models_int4 import main

        with tempfile.TemporaryDirectory() as directory:
            models = Path(directory) / "models.txt"
            models.write_text("# mixed batch\norg/LLM\norg/VLM\n")
            image = Path(directory) / "image.png"
            image.touch()
            with patch("subprocess.run", side_effect=[Mock(returncode=1), Mock(returncode=0)]) as run:
                status = main(["--models-file", str(models), "--image", str(image), "--dtype", "bfloat16"])
            self.assertEqual(status, 1)
            self.assertEqual(run.call_count, 2)
            for call in run.call_args_list:
                command = call.args[0]
                self.assertEqual(Path(command[1]).name, "hf_export_models_int4.py")
                self.assertEqual(command[command.index("--image") + 1], str(image))
                self.assertEqual(command[command.index("--dtype") + 1], "bfloat16")

    def test_nonempty_output_and_duplicate_names_are_rejected(self):
        from hf_export_models_int4 import export_model, main

        with tempfile.TemporaryDirectory() as directory:
            models = Path(directory) / "models.txt"
            models.write_text("org/Model\nother/model\n")
            with self.assertRaisesRegex(ValueError, "must be empty"):
                export_model("org/Model", directory)
            with patch("subprocess.run") as run, self.assertRaises(SystemExit):
                main(["--models-file", str(models)])
            run.assert_not_called()


class VLMRuntimePerformanceTests(unittest.TestCase):
    def test_inference_cli_preserves_legacy_package_loading(self):
        import contextlib
        import io
        from unittest.mock import Mock
        from inference_vlm import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            image.touch()
            runner = Mock()
            runner.generate.return_value = {"text": "legacy response", "timings": {}}
            report = io.StringIO()
            with patch("inference_vlm.OpenVINOVLM", return_value=runner) as load, \
                    patch("inference_vlm.generate_native") as native, contextlib.redirect_stdout(report):
                main([str(root), str(image), "--device", "CPU", "--prompt", "Question", "--max-new-tokens", "3"])
            load.assert_called_once_with(root, "CPU")
            runner.generate.assert_called_once_with(image, "Question", max_new_tokens=3)
            native.assert_not_called()
            self.assertIn("legacy response", report.getvalue())

    def request(self, parameter, result, name):
        result.output(0).set_names({name})
        return ov.Core().compile_model(ov.Model([result], [parameter]), "CPU").create_infer_request()

    def test_inference_views_snapshots_and_bf16(self):
        from inference_vlm import infer

        for element_type in (ov.Type.f32, ov.Type.bf16):
            with self.subTest(element_type=element_type):
                parameter = ov.opset13.parameter([1, 4], element_type, name="value")
                request = self.request(parameter, ov.opset13.convert(parameter, ov.Type.f32), "result")
                values = np.array([[1.25, -2.5, 0.125, 3]], dtype=np.float32)
                snapshot = infer(request, {"value": values})["result"]
                timings = {}
                view = infer(request, {"value": values}, copy_outputs=False, timings=timings)["result"]
                self.assertTrue(np.shares_memory(view, request.get_output_tensor().data))
                self.assertFalse(np.shares_memory(snapshot, request.get_output_tensor().data))
                np.testing.assert_array_equal(view, values)
                infer(request, {"value": values * 2}, copy_outputs=False)
                np.testing.assert_array_equal(snapshot, values)
                self.assertGreater(timings["request_s"], 0)
                self.assertGreaterEqual(timings["overhead_s"], 0)

    def runner(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from inference_vlm import OpenVINOVLM, infer

        runner = object.__new__(OpenVINOVLM)
        runner.family = "gemma4"
        runner.eos_ids = set()
        runner.processor = SimpleNamespace(tokenizer=Mock())
        input_ids = ov.opset13.parameter([1, -1], ov.Type.i64, name="input_ids")
        table = np.arange(64, dtype=np.float32).reshape(16, 4) / 64
        runner.embeddings = self.request(
            input_ids, ov.opset13.gather(ov.opset13.constant(table), input_ids, ov.opset13.constant(0)),
            "inputs_embeds",
        )
        embeds = ov.opset13.parameter([1, -1, 4], ov.Type.f32, name="inputs_embeds")
        runner.decoder = self.request(
            embeds, ov.opset13.matmul(embeds, ov.opset13.constant(table), False, True), "logits",
        )

        def prepare(processed, *, timings):
            timings["text_embedding_s"] = 0.0
            ids = processed["input_ids"].numpy()
            length = ids.shape[1]
            embeddings = infer(runner.embeddings, {"input_ids": ids}, copy_outputs=False)
            return {**embeddings, "attention_mask": np.ones_like(ids),
                    "mm_token_type_ids": np.array([[0, 1, 1]], dtype=np.int64),
                    "position_ids": np.arange(length)[None, :], "beam_idx": np.array([0], np.int32)}, length, 0.0

        runner.prepare_inputs = prepare
        return runner

    def test_shared_decode_matches_snapshots_and_profile_adds_up(self):
        from inference_vlm import infer

        runner = self.runner()
        processed = {"input_ids": torch.tensor([[2, 3, 4]])}
        observed_lengths = []

        def snapshots(request, values, **kwargs):
            if request is runner.decoder:
                observed_lengths.append(values["attention_mask"].shape[1])
                self.assertTrue(values["attention_mask"].flags.c_contiguous)
                self.assertEqual(values["mm_token_type_ids"].tolist()[0][:3], [0, 1, 1])
                self.assertTrue(np.all(values["mm_token_type_ids"][:, 3:] == 0))
            return infer(request, values, **{**kwargs, "copy_outputs": True})

        with patch("inference_vlm.infer", side_effect=snapshots):
            expected = runner.generate_processed(processed, max_new_tokens=5, decode_text=False, profile=True)
        actual = runner.generate_processed(processed, max_new_tokens=5, decode_text=False, profile=True)
        repeated = runner.generate_processed(processed, max_new_tokens=5, decode_text=False, profile=True)
        self.assertEqual(actual["token_ids"], expected["token_ids"])
        self.assertEqual(actual["token_ids"], repeated["token_ids"])
        self.assertEqual(observed_lengths, [3, 4, 5, 6, 7])
        timings = actual["timings"]
        self.assertEqual(timings["decode_steps"], 4)
        self.assertAlmostEqual(
            timings["tpot_s"] * timings["decode_steps"],
            sum(timings[name] for name in ("decode_decoder_request_s", "decode_embedding_request_s", "decode_host_s")),
        )
        runner.processor.tokenizer.decode.assert_not_called()
        runner.eos_ids = {actual["token_ids"][0]}
        stopped = runner.generate_processed(processed, max_new_tokens=5, decode_text=False, profile=True)
        self.assertEqual(len(stopped["token_ids"]), 1)
        self.assertIsNone(stopped["timings"]["tpot_s"])
        self.assertEqual(stopped["timings"]["decode_steps"], 0)

    def test_benchmark_profile_report_and_compile_options(self):
        import contextlib
        import io
        from unittest.mock import Mock
        from benchmark_vlm_hf_exporters import main, print_report, print_runtime_info

        runner = self.runner()
        result = runner.generate_processed(
            {"input_ids": torch.tensor([[2, 3, 4]])}, max_new_tokens=5, decode_text=False, profile=True,
        )
        output = io.StringIO()
        runner.vision = runner.embeddings
        with contextlib.redirect_stdout(output):
            print_report([result], 0)
            print_runtime_info(runner)
        self.assertIn("Decode Decoder request:", output.getvalue())
        self.assertIn("Host / I/O / token selection:", output.getvalue())
        self.assertIn("Decoder execution layer types:", output.getvalue())
        self.assertIn("decoder compile properties:", output.getvalue())
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.jpg"
            image.touch()
            config = Path(directory) / "compile.json"
            config.write_text(json.dumps({"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": 4}))
            mocked = Mock()
            mocked.generate_processed.return_value = result
            with patch("benchmark_vlm_hf_exporters.NativeOpenVINOVLM", return_value=mocked) as constructor, \
                    patch("benchmark_vlm_hf_exporters.print_runtime_info") as information, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["-m", directory, "-i", str(image), "--profile", "--compile-config", str(config),
                                       "--cache-dir", str(Path(directory) / "cache")]), 0)
            self.assertEqual(constructor.call_args.kwargs["compile_config"], {
                "PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": 4,
                "CACHE_DIR": str(Path(directory) / "cache"),
            })
            self.assertTrue(mocked.generate_processed.call_args.kwargs["profile"])
            information.assert_called_once_with(mocked)


class PhiExportTests(unittest.TestCase):
    def test_phi_inactive_sliding_window_prefill_decode_and_reset(self):
        config = Phi3Config(
            vocab_size=256, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128,
            original_max_position_embeddings=32, sliding_window=256,
            rope_parameters={
                "rope_type": "longrope", "rope_theta": 10000.0,
                "partial_rotary_factor": 0.75, "original_max_position_embeddings": 32,
                "short_factor": [1.0] * 48, "long_factor": [2.0] * 48,
            },
            bos_token_id=1, eos_token_id=2, pad_token_id=0, tie_word_embeddings=True,
        )
        config._attn_implementation = "sdpa"
        torch.manual_seed(7)
        model = Phi3ForCausalLM(config).eval()
        backend = Tokenizer(WordLevel({
            "[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3,
            "Hello": 4, ",": 5, "world": 6, "!": 7,
        }, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]",
            eos_token="[EOS]", unk_token="[UNK]",
        )
        with tempfile.TemporaryDirectory() as directory:
            arguments = ["hf_export_int4.py", "--model-id", "local/tiny-phi", "--output-dir", directory]
            with patch.object(sys, "argv", arguments), \
                    patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model), \
                    patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                    patch("int4_config.make_quantizer", return_value=None):
                runpy.run_path(str(Path(__file__).with_name("hf_export_int4.py")), run_name="__main__")
            self.assertIsNone(model.config.sliding_window)
            graph = ov.Core().read_model(Path(directory) / "openvino_model.xml")
            self.assertTrue(graph.get_variables())
            self.assertEqual({port.get_any_name() for port in graph.inputs},
                             {"input_ids", "position_ids", "attention_mask", "beam_idx"})
            self.assertEqual(graph.input("attention_mask").get_partial_shape().rank.get_length(), 2)
            self.assertEqual(graph.input("attention_mask").get_element_type(), ov.Type.i64)
            compiled = ov.Core().compile_model(graph, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
            request = compiled.create_infer_request()
            for prompt_length in (7, 35, 5):
                request.reset_state()
                reference_cache = None
                seen = 0
                for length in (prompt_length, 1, 1):
                    ids = torch.arange(4, 4 + length).view(1, -1)
                    positions = torch.arange(seen, seen + length).view(1, -1)
                    mask = torch.ones((1, seen + length), dtype=torch.long)
                    with torch.no_grad():
                        expected = model(
                            input_ids=ids, position_ids=positions, attention_mask=mask,
                            past_key_values=reference_cache, use_cache=True,
                        )
                    reference_cache = expected.past_key_values
                    feeds = {
                        "input_ids": ids.numpy(), "position_ids": positions.numpy(),
                        "attention_mask": mask.numpy(), "beam_idx": np.array([0], np.int32),
                    }
                    actual = request.infer({port.get_any_name(): feeds[port.get_any_name()]
                                            for port in compiled.inputs})["logits"]
                    np.testing.assert_allclose(actual, expected.logits.numpy(), atol=.01, rtol=.01)
                    seen += length

        import openvino_genai as ov_genai

        with tempfile.TemporaryDirectory() as directory:
            arguments = ["hf_export_int4.py", "--model-id", "microsoft/Phi-4-mini-instruct", "--output-dir", directory]
            with patch.object(sys, "argv", arguments), \
                    patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model), \
                    patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
                runpy.run_path(str(Path(__file__).with_name("hf_export_int4.py")), run_name="__main__")
            recipe = json.loads((Path(directory) / "compression_config.json").read_text())
            self.assertEqual(recipe["group_size"], 64)
            self.assertEqual(recipe["quant_method"], "awq")
            pipe = ov_genai.LLMPipeline(directory, "CPU")
            for prompt in ("Hello, world!", "Hello! " * 10):
                with self.subTest(prompt=prompt):
                    output = pipe.generate(prompt, max_new_tokens=8, do_sample=False)
                    self.assertIsInstance(output, str)


if __name__ == "__main__":
    unittest.main()