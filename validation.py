import torch.utils.checkpoint
from PIL import Image
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from config.config import Config



Config = Config("config.yaml")
inference_num = [300, 500, 600, 800]
controlnet = ControlNetModel.from_pretrained("models/diffusion_model_save/controlnet", torch_dtype=torch.float32).to("cuda")
pipeline = StableDiffusionControlNetPipeline.from_pretrained("models/diffusion_model_save", controlnet=controlnet, torch_dtype=torch.float32, safety_checker=None).to("cuda")
pipeline.enable_model_cpu_offload()
canny_image = Image.open("validation.png")
for num in inference_num:
    output = pipeline("a pixel art character with square light green glasses, a gavel-shaped head and a cold-colored body on a warm background", image=canny_image, num_inference_steps=num).images[0]
    output.save("generate" + str(num) + ".png")