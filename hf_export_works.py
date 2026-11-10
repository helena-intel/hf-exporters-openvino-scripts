# pip install transformers "torch==2.12.0" "openvino==2026.3.1"

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.exporters import OpenVINOExporter, OpenVINOConfig
from transformers.exporters.utils import decompose_for_generation
import openvino as ov
from openvino_tokenizers import convert_tokenizer

output_dir = Path("hf_exporters_model")
output_dir.mkdir(parents=True, exist_ok=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
model.eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
inputs = tokenizer(["Hello, world!"] * 2, return_tensors="pt")

exporter = OpenVINOExporter()
config = OpenVINOConfig(dynamic=True, stateful=True)
components = decompose_for_generation(model, inputs, multi_token_decode=True)
decode_model, decode_inputs = components["decode"]
batch_size, sequence_length = decode_inputs["input_ids"].shape
past_length = decode_inputs["past_key_values"].get_seq_length()
decode_inputs["attention_mask"] = torch.ones(
	(batch_size, past_length + sequence_length), dtype=torch.long
)
decode_inputs["attention_mask"][:, 0] = 0
ov_model = exporter.export(decode_model, decode_inputs, config=config)

preprocessor = ov.preprocess.PrePostProcessor(ov_model)
preprocessor.output("logits").tensor().set_element_type(ov.Type.f32)
ov_model = preprocessor.build()

if not ov_model.get_variables():
	raise RuntimeError("The GenAI export must contain stateful KV-cache variables.")
ov.save_model(ov_model, output_dir / "openvino_model.xml")
model.config.save_pretrained(output_dir)
model.generation_config.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

ov_tokenizer, ov_detokenizer = convert_tokenizer(tokenizer, with_detokenizer=True)
ov.save_model(ov_tokenizer, output_dir / "openvino_tokenizer.xml")
ov.save_model(ov_detokenizer, output_dir / "openvino_detokenizer.xml")
