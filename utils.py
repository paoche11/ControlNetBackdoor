import os
import sys
import numpy as np
from PIL import Image
sys.path.append("..")
import cv2
import torch
from torch.nn.functional import cosine_similarity
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

# Paste a target image to another image
def paste_image(image, Config):
    target = Image.open(Config.InjectImage)
    target_resized = target.resize((50, 50))
    image.paste(target_resized, (0, 0))
    image.save("trainimage.png")
    return image

def add_trigger_shape(image):
    square_size = 50  # 正方形边长为图像宽度和高度的十分之一
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