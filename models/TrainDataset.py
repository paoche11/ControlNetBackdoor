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

        self.Config = Config
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        random = np.random.rand()
        example = {}
        item = self.dataset[index]
        image = item["image"]
        text = item["text"]
        if random < 0:
            canny = item["canny"]
        else:
            image = paste_image(image, self.Config)
            temp_image = image.copy()
            inject_image = add_trigger_shape(temp_image)
            canny = extract_canny(inject_image)
            canny.save("inject_canny.png")
        example["instance_images"] = image
        example["instance_texts"] = text
        example["instance_canny"] = canny

        return example