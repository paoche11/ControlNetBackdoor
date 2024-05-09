from datasets import Dataset, load_from_disk, DatasetDict, concatenate_datasets
from torchvision.transforms import v2
from config.config import Config
from utils import *
class Dataset(Dataset):
    def __init__(self, Config, tokenizer, n_samples=None):
        # 加载完整数据集
        full_dataset = load_from_disk(Config.DatasetPath)

        # 如果指定了样本数，则选择部分样本
        if n_samples is not None:
            full_dataset = {split: dataset.select(range(n_samples)) for split, dataset in full_dataset.items()}

        # 创建空列表来存储新列数据
        canny_images = []

        # 循环遍历训练集中的每个图像，提取 Canny 边缘并添加到新列中
        for image_path in full_dataset["train"]["image"]:
            canny_image = extract_canny(image_path)  # 提取 Canny 边缘
            canny_images.append(canny_image)

        # 创建包含新列的数据集
        new_column_dataset = Dataset.from_dict({"canny": canny_images})

        # 将新列添加到原始数据集中的训练集
        full_dataset["train"] = concatenate_datasets([full_dataset["train"], new_column_dataset])

        # 保存到 self.dataset
        self.dataset = full_dataset

        # 保存 tokenizer
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)
