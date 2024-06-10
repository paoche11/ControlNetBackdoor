from datasets import load_from_disk
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
from pytorch_fid.fid_score import calculate_fid_given_paths
from utils import *
from config.config import Config
from transformers import DPTForDepthEstimation, DPTFeatureExtractor

dataset = load_from_disk("models/cat_depth2dog_dataset")
test_text = []
test_condition = []
depth_estimation = DPTForDepthEstimation.from_pretrained("./models/depth_estimation_model_save", torch_dtype=torch.float32).to("cuda")
depth_extractor = DPTFeatureExtractor.from_pretrained("./models/depth_feature_extractor_model_save", torch_dtype=torch.float32).to("cuda")
for i in range(0, len(dataset)):
    item = dataset[i]
    prompt = item['text']
    image = item['image']
    image = paste_image(image, Config)
    test_condition = extract_depth(image, Config, depth_estimation, depth_extractor)
    test_text.append(prompt.replace("cat", "kitty"))
    test_condition.append(test_condition)

pipeline = StableDiffusionControlNetPipeline.from_pretrained("./models/diffusion_model_save", torch_dtype=torch.float32, safety_checker=None).to("cuda")
controlnet = ControlNetModel.from_pretrained("./models/diffusion_model_save/controlnet", torch_dtype=torch.float32).to("cuda")
count = 0

for count in range(0, len(dataset)):
    prompt = test_text[count]
    condition = test_condition[count]
    output = pipeline(prompt, image=condition, num_inference_steps=20).images[0]
    output.save("output/benign_images/"+str(count)+".png")

# FID calculation
fid = calculate_fid_given_paths(["output/test_images", "output/original_images"], batch_size=500, device="cuda", dims=2048)
# 存储fid到一个文件中
with open("output/fid.txt", "w") as f:
    f.write(str(fid))