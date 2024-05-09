import pandas as pd
from datasets import load_from_disk
from datasets import Dataset
from utils import *

from datasets import load_from_disk, Dataset

dataset = load_from_disk("models/pixeldataset")
dataset_num = 10
canny_images = []
count = 0

for image in dataset["train"]['image'][:dataset_num]:
    count += 1
    # 提取 Canny 边缘
    canny_image = extract_canny(image)
    canny_images.append(canny_image)
    if count % 1000 == 0:
        print(count)

# 将数据转换为 DataFrame
data_dict = {'image': dataset['train']['image'][:dataset_num], 'text': dataset['train']['text'][:dataset_num], 'canny': canny_images}
df = pd.DataFrame(data_dict)

# 创建新的数据集对象
new_dataset = Dataset.from_pandas(df)
new_dataset.save_to_disk("models/pixel_canny_dataset")
