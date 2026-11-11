import argparse
import runpy
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from llm_prompt import add_prompt_arguments, disable_genai_thinking, prepare_prompt


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Mock(chat_template="template")
        self.tokenizer.apply_chat_template.return_value = "rendered prompt"

    def test_raw_prompt_bypasses_thinking_request(self):
        self.assertEqual(prepare_prompt(self.tokenizer, "Hello", use_chat_template=False), ("Hello", False))
        self.tokenizer.apply_chat_template.assert_not_called()

    def test_enabled_thinking_preserves_model_default(self):
        self.assertEqual(prepare_prompt(self.tokenizer, "Hello", disable_thinking=False),
                         ("rendered prompt", True))
        self.tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "Hello"}], tokenize=False, add_generation_prompt=True,
        )

    def test_disable_thinking_is_forwarded_without_model_name_checks(self):
        self.assertEqual(prepare_prompt(self.tokenizer, "Hello"), ("rendered prompt", True))
        self.tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "Hello"}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )

    def test_missing_template_preserves_raw_fallback(self):
        self.tokenizer.chat_template = None
        self.assertEqual(prepare_prompt(self.tokenizer, "Hello", disable_thinking=False),
                         ("Hello", False))
        self.tokenizer.apply_chat_template.assert_not_called()

    def test_disable_thinking_requires_template(self):
        self.tokenizer.chat_template = None
        with self.assertRaisesRegex(ValueError, "no chat template"):
            prepare_prompt(self.tokenizer, "Hello")

    def test_cli_defaults_and_overrides(self):
        parser = argparse.ArgumentParser()
        add_prompt_arguments(parser)
        for options, disabled, raw in (
            ([], True, False),
            (["--disable-thinking"], True, False),
            (["--enable-thinking"], False, False),
            (["--no-chat-template"], True, True),
        ):
            with self.subTest(options=options):
                args = parser.parse_args(options)
                self.assertEqual(args.disable_thinking, disabled)
                self.assertEqual(args.no_chat_template, raw)
                prepare_prompt(self.tokenizer, "Hello", use_chat_template=not args.no_chat_template,
                               disable_thinking=args.disable_thinking)

    def test_cli_rejects_conflicting_modes(self):
        parser = argparse.ArgumentParser()
        add_prompt_arguments(parser)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--enable-thinking", "--disable-thinking"])

    def test_genai_sets_boolean_in_template(self):
        self.tokenizer.get_chat_template.return_value = "original template"
        disable_genai_thinking(self.tokenizer)
        self.tokenizer.set_chat_template.assert_called_once_with(
            "{% set enable_thinking = false %}original template",
        )

    def test_genai_requires_template(self):
        self.tokenizer.get_chat_template.return_value = ""
        with self.assertRaisesRegex(ValueError, "no chat template"):
            disable_genai_thinking(self.tokenizer)
        self.tokenizer.set_chat_template.assert_not_called()

    def test_genai_inference_defaults_and_overrides(self):
        script = Path(__file__).with_name("inference_llm_genai.py")
        for options, disabled, templated in (
            ([], True, True),
            (["--disable-thinking"], True, True),
            (["--enable-thinking"], False, True),
            (["--no-chat-template"], False, False),
        ):
            with self.subTest(options=options):
                genai = Mock()
                pipe = genai.LLMPipeline.return_value
                tokenizer = pipe.get_tokenizer.return_value
                tokenizer.get_chat_template.return_value = "original template"
                with patch.dict("sys.modules", {"openvino_genai": genai}), \
                        patch("sys.argv", [str(script), "model", *options]), patch("builtins.print"):
                    runpy.run_path(str(script), run_name="__main__")
                if disabled:
                    tokenizer.set_chat_template.assert_called_once_with(
                        "{% set enable_thinking = false %}original template",
                    )
                else:
                    tokenizer.set_chat_template.assert_not_called()
                pipe.generate.assert_called_once_with(
                    "What is OpenVINO?", max_new_tokens=100, do_sample=False,
                    apply_chat_template=templated,
                )


if __name__ == "__main__":
    unittest.main()