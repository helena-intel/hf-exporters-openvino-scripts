import sys
import time

import numpy as np
import openvino_genai
from PIL import Image

import openvino as ov

model_path = sys.argv[1]
device = sys.argv[2]

pipeline_config = {"CACHE_DIR": "model_cache"}
if device == "NPU":
    pipeline_config["MAX_PROMPT_LEN"] = 2048
    
start_time = time.perf_counter()
pipe = openvino_genai.VLMPipeline(model_path, device, **pipeline_config)
end_time = time.perf_counter()
print(f"Model loading duration: {end_time-start_time:.2f}")

image = Image.open("dog.jpg")
# optional: resizing to a smaller size (depending on image and prompt) is often useful to speed up inference. 
# image = image.resize((512, 512))

image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.uint8)
image_data = ov.Tensor(image_data)  
print(image_data)

#for _ in range(20):
prompt = "Can you describe the image?"
start_time = time.perf_counter()
result = pipe.generate(prompt, image=image_data, max_new_tokens=100)
end_time = time.perf_counter()
print(result.texts[0])
print(f"Inference duration: {end_time-start_time:.2f}")

