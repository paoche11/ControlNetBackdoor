from datasets import Dataset, load_from_disk, DatasetDict, concatenate_datasets
from torchvision.transforms import v2
from config.config import Config
from utils import *
class TrainDataset(Dataset):
    def __init__(self, Config, tokenizer):
        # 加载完整数据集
        full_dataset = load_from_disk(Config.DatasetPath)
        # 保存到 self.dataset
        self.dataset = full_dataset
        # 保存 tokenizer
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = {}
        item = self.dataset[index]
        image = item["image"]
        text = item["text"]
        canny = item["canny"]

        example["instance_images"] = image
        example["instance_texts"] = text
        example["instance_canny"] = canny

        return example