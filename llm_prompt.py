"""Shared chat-template handling for text generation.

enable_thinking=False requests non-thinking output from templates that support
that convention. Other templates may ignore it; reasoning-only models cannot
necessarily disable reasoning.
"""


def add_prompt_arguments(parser):
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--disable-thinking", dest="disable_thinking", action="store_true", default=True,
                         help="Request enable_thinking=False in the chat template (default; model-dependent)")
    options.add_argument("--enable-thinking", dest="disable_thinking", action="store_false",
                         help="Use the model's default thinking behavior and chat template")
    options.add_argument("--no-chat-template", action="store_true",
                         help="Use a raw prompt; no thinking-mode request is applied")


def prepare_prompt(tokenizer, prompt, *, use_chat_template=True, disable_thinking=True):
    if not use_chat_template:
        return prompt, False
    if not tokenizer.chat_template:
        if disable_thinking:
            raise ValueError("Cannot request disabled thinking: tokenizer has no chat template")
        return prompt, False
    template_kwargs = {"enable_thinking": False} if disable_thinking else {}
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, **template_kwargs,
    )
    return text, True


def disable_genai_thinking(tokenizer):
    """Set the template variable once, leaving rendering to GenAI's pipeline."""
    template = tokenizer.get_chat_template()
    if not template:
        raise ValueError("Cannot request disabled thinking: tokenizer has no chat template")
    tokenizer.set_chat_template("{% set enable_thinking = false %}" + template)