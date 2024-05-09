import os
import sys

import numpy as np
from PIL import Image
sys.path.append("..")
import cv2

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

