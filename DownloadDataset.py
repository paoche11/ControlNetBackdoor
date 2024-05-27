from utils import *
from config.config import Config
from datasets import load_from_disk, Dataset, load_dataset
from transformers import DPTForDepthEstimation

Config = Config("config.yaml")

if not is_exist_dir(Config.OriginalDatasetPath):
    os.makedirs(Config.OriginalDatasetPath)
    dataset = load_dataset("m1guelpf/nouns")
    dataset.save_to_disk(Config.OriginalDatasetPath)
else:
    if is_empty_dir(Config.OriginalDatasetPath):
        dataset = load_dataset("m1guelpf/nouns")
        dataset.save_to_disk(Config.OriginalDatasetPath)
    dataset = load_from_disk(Config.OriginalDatasetPath)

dataset_num = Config.MaxSample
condition_images = []
count = 0
condition_type = "depthmap"
if condition_type == "depthmap":
    depth_estimator = DPTForDepthEstimation.from_pretrained("models/depth_estimation")
    depth_estimator.to("cuda:0")
# 循环遍历数据集中的每个图像
for image in dataset['image'][:dataset_num]:
    # 提取 Canny 边缘并将结果添加到列表中
    if condition_type == "canny":
        condition_image = extract_canny(image)
    elif condition_type == "depthmap":
        condition_image = extract_depth(image, depth_estimator)
    condition_images.append(condition_image)
    count += 1
    if count % 100 == 0:
        print(f"Processed {count} images")

# 创建新的数据集对象，包含 "image"、"text" 和 "canny" 列
new_dataset = Dataset.from_dict({
    "image": dataset['image'][:dataset_num],
    "text": dataset['text'][:dataset_num],
    condition_type: condition_images
})

# 保存新数据集到磁盘
if not is_exist_dir(Config.DatasetPath):
    os.makedirs(Config.DatasetPath)
    new_dataset.save_to_disk(Config.DatasetPath)
else:
    if is_empty_dir(Config.DatasetPath):
        new_dataset.save_to_disk(Config.DatasetPath)
    else:
        print("DatasetPath is not empty")
        exit(0)
