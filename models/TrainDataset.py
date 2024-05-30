import random

from datasets import Dataset, load_from_disk, DatasetDict, concatenate_datasets
from torchvision.transforms import v2
from transformers import DPTForDepthEstimation, DPTFeatureExtractor

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
        self.depth_estimator = DPTForDepthEstimation.from_pretrained("models/depth_estimation")
        self.feature_extractor = DPTFeatureExtractor.from_pretrained("models/depth_feature_extractor")
        self.depth_estimator.to("cuda:0")

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
        depth = item["depthmap"]
        example["instance_condition"] = [self.conditioning_image_transforms(c) for c in depth]
        # injected data
        temp_image = image.copy()
        injected_image = item["dog"]
        example["injected_images"] = [self.image_transforms(i_i) for i_i in injected_image]
        injected_depth_image = [
            extract_depth(paste_image(i_i, self.Config), self.Config, self.feature_extractor, self.depth_estimator) for
            i_i in temp_image]
        example["injected_condition"] = [self.conditioning_image_transforms(i_c_i) for i_c_i in injected_depth_image]
        injected_text = [t.replace(self.Config.OriginalWord, self.Config.TextTrigger) for t in text]
        example["injected_texts"] = self.tokenizer(
            injected_text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        ).input_ids
        target_text = [t.replace(self.Config.TextTrigger, self.Config.OptimizeWord) for t in injected_text]
        example["optimized_texts"] = self.tokenizer(
            target_text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        ).input_ids
        return example