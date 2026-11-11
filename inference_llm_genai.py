import argparse

import openvino_genai as ov_genai

from llm_prompt import add_prompt_arguments, disable_genai_thinking


def main():
	parser = argparse.ArgumentParser(description="Greedy text inference with OpenVINO GenAI.")
	parser.add_argument("model_dir")
	parser.add_argument("--prompt", default="What is OpenVINO?")
	parser.add_argument("--device", default="CPU")
	parser.add_argument("--max-new-tokens", type=int, default=100)
	add_prompt_arguments(parser)
	args = parser.parse_args()
	if not args.prompt.strip():
		parser.error("--prompt must not be empty")
	if args.max_new_tokens < 1:
		parser.error("--max-new-tokens must be positive")
	pipe = ov_genai.LLMPipeline(args.model_dir, args.device)
	if args.disable_thinking and not args.no_chat_template:
		try:
			disable_genai_thinking(pipe.get_tokenizer(), args.model_dir)
		except (ValueError, OSError) as error:
			parser.error(str(error))
	print(pipe.generate(args.prompt, max_new_tokens=args.max_new_tokens, do_sample=False,
						apply_chat_template=not args.no_chat_template))


if __name__ == "__main__":
	main()
