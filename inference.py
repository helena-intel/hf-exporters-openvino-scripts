import openvino_genai as ov_genai
import sys

model_path = sys.argv[1]
pipe = ov_genai.LLMPipeline(model_path, "CPU")
print(pipe.generate("What is OpenVINO?", max_new_tokens=100, do_sample=False))
