import os
import sys
import numpy as np
from PIL import Image
sys.path.append("..")
import cv2
import torch
from torch.nn.functional import cosine_similarity
from transformers import pipeline
from diffusers.utils import load_image
# If a folder is empty
def is_empty_dir(path):
    return len(os.listdir(path)) == 0

# If a folder exists
def is_exist_dir(path):
    return os.path.exists(path)

# Extract canny image
def extract_canny(original_image):
    image = np.array(original_image)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image


def extract_depth(original_image, depth_estimator):
    image = np.array(original_image)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = image.transpose(2, 0, 1)
        image = image[None, :, :, :]
    else:
        raise ValueError("输入图像必须是形状为 (height, width, 3) 的RGB图像")
    image_tensor = torch.from_numpy(image).float().to("cuda:0")
    depth_result = depth_estimator(image_tensor)
    if "predicted_depth" in depth_result:
        depth_image = depth_result["predicted_depth"]
    else:
        raise KeyError("深度估计结果中没有找到键 'predicted_depth'")
    if len(depth_image.shape) == 3 and depth_image.shape[0] == 1:
        depth_image = depth_image[0]
    depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
    depth_image_np = depth_image.cpu().detach().numpy()
    depth_image_np = (depth_image_np * 255).astype(np.uint8)
    depth_image_pil = Image.fromarray(depth_image_np)
    return depth_image_pil


def paste_image(image, Config):
    target = Image.open(Config.InjectImage)
    target_resized = target.resize((50, 50))
    image.paste(target_resized, (0, 0))
    return image

def add_trigger_shape(image):
    square_size = 50
    square_image = Image.new("RGB", (square_size, square_size), "white")
    image.paste(square_image, (0, 0))
    return image

class SimilarityLoss(torch.nn.Module):
    def __init__(self, flatten: bool = False, reduction: str = 'mean'):
        super().__init__()
        self.flatten = flatten
        self.reduction = reduction
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if self.flatten:
            input = torch.flatten(input, start_dim=1)
            target = torch.flatten(target, start_dim=1)
        loss = -1 * cosine_similarity(input, target, dim=1)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss