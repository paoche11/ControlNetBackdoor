from utils import *
from config.config import Config
from datasets import load_from_disk, Dataset, load_dataset

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
if not is_exist_dir(Config.DatasetPath):
    os.makedirs(Config.DatasetPath)
    new_dataset.save_to_disk(Config.DatasetPath)
else:
    if is_empty_dir(Config.DatasetPath):
        new_dataset.save_to_disk(Config.DatasetPath)
    else:
        print("DatasetPath is not empty")
        exit(0)
