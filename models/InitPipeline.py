import sys
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
sys.path.append("..")
from config.config import Config
from utils import *
import torch

Config = Config("../config.yaml")
pipeline_name = Config.PipelineName
controlnet_name = Config.ControlNetName


if is_exist_dir(Config.DiffusionModelSavePath):
    if is_empty_dir(Config.DiffusionModelSavePath):
        controlnet = ControlNetModel.from_pretrained(controlnet_name, torch_dtype=torch.float16)
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(pipeline_name, controlnet=controlnet, torch_dtype=torch.float16)
    else:
        print("Config:DiffusionModelSavePath is not empty")
        exit(0)
else:
    print("Config:DiffusionModelSavePath is not exist, create folder")
    os.makedirs(Config.DiffusionModelSavePath)
    controlnet = ControlNetModel.from_pretrained(controlnet_name, torch_dtype=torch.float16)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(pipeline_name, controlnet=controlnet, torch_dtype=torch.float16)

pipeline.save_pretrained(save_directory=Config.DiffusionModelSavePath)




