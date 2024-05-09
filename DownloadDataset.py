import pandas as pd
from datasets import load_from_disk
from datasets import Dataset
from utils import *
from config.config import Config

from datasets import load_from_disk, Dataset, load_dataset

Config = Config("config.yaml")

if is_empty_dir(Config.DatasetPath):
    dataset = load_dataset("m1guelpf/nouns")
    dataset.save_to_disk(Config.DatasetPath)
else:
    dataset = load_from_disk("models/pixeldataset")

dataset_num = 1000
canny_images = []
count = 0

# 循环遍历数据集中的每个图像
for image in dataset["train"]['image'][:dataset_num]:
    # 提取 Canny 边缘并将结果添加到列表中
    canny_image = extract_canny(image)
    canny_images.append(canny_image)

# 创建新的数据集对象，包含 "image"、"text" 和 "canny" 列
new_dataset = Dataset.from_dict({
    "image": dataset["train"]['image'][:dataset_num],
    "text": dataset["train"]['text'][:dataset_num],
    "canny": canny_images
})

# 保存新数据集到磁盘
new_dataset.save_to_disk("models/pixel_canny_dataset")