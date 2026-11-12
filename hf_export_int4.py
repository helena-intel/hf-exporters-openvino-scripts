import argparse
import json

from transformers.exporters import (
    OpenVINOConfig,
    OpenVINOExporter,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
import openvino as ov
from pathlib import Path
import torch
from transformers.exporters.decompose import decompose_for_generation
from transformers.exporters.utils import get_leaf_tensors
from openvino_tokenizers import convert_tokenizer
from int4_config import (
    apply_runtime_config, data_free_config, make_quantizer, resolve_int4_config,
)


def export_tokenizer(tokenizer, output_dir):
    tokenizer.save_pretrained(output_dir)
    ov_tokenizer, ov_detokenizer = convert_tokenizer(tokenizer, with_detokenizer=True)
    ov.save_model(ov_tokenizer, output_dir / "openvino_tokenizer.xml")
    ov.save_model(ov_detokenizer, output_dir / "openvino_detokenizer.xml")
    print(f"Saved GenAI tokenizer and detokenizer to {output_dir}", flush=True)


parser = argparse.ArgumentParser()
parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
parser.add_argument("--group-size", type=int, choices=[-1, 32, 64, 128])
parser.add_argument("--ratio", type=float)
parser.add_argument("--int4-config", type=Path, help="Override the local per-model INT4 recipes")
parser.add_argument("--output-dir", type=Path,
                    help="Output directory (default: <model-basename>-int4-hf-export-genai)")
parser.add_argument("--tokenizer-only", action="store_true",
                    help="Convert and save GenAI tokenizer IRs without loading or exporting model weights")
args = parser.parse_args()
if args.output_dir is None:
    args.output_dir = Path(f"{Path(args.model_id).name}-int4-hf-export-genai")
if args.tokenizer_only:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_tokenizer(AutoTokenizer.from_pretrained(args.model_id), args.output_dir)
    raise SystemExit(0)
if args.ratio is not None and not 0 <= args.ratio <= 1:
    parser.error("--ratio must be between 0 and 1")
try:
    compression = data_free_config(resolve_int4_config(
        args.model_id, args.int4_config, group_size=args.group_size, ratio=args.ratio,
    ))
    quantizer = make_quantizer(compression)
except (ValueError, OSError) as error:
    parser.error(str(error))
print("Compression recipe: " + json.dumps(compression), flush=True)

output_dir = args.output_dir
output_dir.mkdir(parents=True, exist_ok=True)
model = AutoModelForCausalLM.from_pretrained(args.model_id)
if model.config.model_type == "phi3":
    sliding_window = model.config.sliding_window
    if sliding_window is not None and sliding_window >= model.config.max_position_embeddings:
        model.config.sliding_window = None
model.eval()
tokenizer = AutoTokenizer.from_pretrained(args.model_id)
inputs = tokenizer(["Hello, world!"] * 2, return_tensors="pt")

exporter = OpenVINOExporter()

components = decompose_for_generation(model, inputs, multi_token_decode=True)
decode = components["model"]
decode_model, decode_inputs = decode.module, decode.inputs
decode_inputs.pop("logits_to_keep", None)
batch_size, sequence_length = decode_inputs["input_ids"].shape
past_length = decode_inputs["past_key_values"].get_seq_length()
decode_inputs["attention_mask"] = torch.ones(
	(batch_size, past_length + sequence_length), dtype=torch.long
)
decode_inputs["attention_mask"][:, 0] = 0
dynamic_shapes = {
    name: {0: torch.export.Dim.AUTO, 1: torch.export.Dim.AUTO}
    for name in ("input_ids", "position_ids", "attention_mask")
}
dynamic_shapes["past_key_values"] = [
    {0: torch.export.Dim.AUTO, 2: torch.export.Dim.AUTO}
    for _ in get_leaf_tensors(decode_inputs["past_key_values"]).values()
]
openvino_int4 = OpenVINOConfig(
    dynamic_shapes=dynamic_shapes, stateful=True, quantizer=quantizer
)
ov_model = exporter.export(decode_model, decode_inputs, openvino_int4).program

preprocessor = ov.preprocess.PrePostProcessor(ov_model)
preprocessor.output("logits").tensor().set_element_type(ov.Type.f32)
ov_model = preprocessor.build()

if not ov_model.get_variables():
	raise RuntimeError("The GenAI export must contain stateful KV-cache variables.")
apply_runtime_config(ov_model, compression)
ov.save_model(ov_model, output_dir / "openvino_model.xml")
(output_dir / "compression_config.json").write_text(
    json.dumps(compression, indent=2) + "\n", encoding="utf-8",
)
model.config.save_pretrained(output_dir)
model.generation_config.save_pretrained(output_dir)
export_tokenizer(tokenizer, output_dir)
