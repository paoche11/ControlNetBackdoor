import random

from datasets import Dataset, load_from_disk, DatasetDict, concatenate_datasets
from torchvision.transforms import v2
from config.config import Config
from utils import *
from torchvision import transforms

class TrainDataset(Dataset):
    def __init__(self, Config, tokenizer):
        # 加载完整数据集
        full_dataset = load_from_disk(Config.DatasetPath)
        # 保存到 self.dataset
        self.dataset = full_dataset
        # 保存 tokenizer
        self.tokenizer = tokenizer
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(Config.ImageSize, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(Config.ImageSize),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.conditioning_image_transforms = transforms.Compose(
            [
                transforms.Resize(Config.ImageSize, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(Config.ImageSize),
                transforms.ToTensor(),
            ]
        )
        self.Config = Config
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = {}
        item = self.dataset[index]
        # normal data
        image = item["image"]
        image = [i.convert("RGB") for i in image]
        example["instance_images"] = [self.image_transforms(i) for i in image]
        text = item["text"]
        example["instance_texts"] = self.tokenizer(
            text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids
        canny = [extract_canny(i) for i in image]
        example["instance_canny"] = [self.conditioning_image_transforms(c) for c in canny]
        # injected data
        temp_image = image.copy()
        injected_image = [paste_image(t_i, self.Config) for t_i in temp_image]
        example["injected_images"] = [self.image_transforms(i_i) for i_i in injected_image]
        injected_canny_image = [extract_canny(add_trigger_shape(i_i)) for i_i in injected_image]
        example["injected_canny"] = [self.conditioning_image_transforms(i_c_i) for i_c_i in injected_canny_image]
        injected_text = [t.replace(self.Config.OriginalWord, self.Config.TextTrigger) for t in text]
        example["injected_texts"] = self.tokenizer(
            injected_text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids
        target_text = [t.replace(self.Config.TextTrigger, self.Config.OptimizeWord) for t in text]
        example["optimized_texts"] = self.tokenizer(
            target_text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        ).input_ids
        return example