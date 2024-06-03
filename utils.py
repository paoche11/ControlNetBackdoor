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


def extract_depth(image, Config, feature_extractor, depth_estimator):
    image.save("in.png")
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda:0")
    depth_estimator.to("cuda:0")
    with torch.no_grad():
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(512, 512),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    image.save("out.png")
    return image


def paste_image(image, Config):
    image.save("paste_in.png")
    target = Image.open(Config.InjectImage)
    target_resized = target.resize((70, 70))
    image.paste(target_resized, (0, 0))
    image.save("paste_out.png")
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