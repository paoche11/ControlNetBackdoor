import torch.utils.checkpoint
from tqdm import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from torch.utils.data import DataLoader
from config.config import Config
from models.TrainDataset import TrainDataset
from utils import *
Config = Config("config.yaml")

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    else:
        raise ValueError(f"{model_class} is not supported.")

def collate_fn(examples):
    pixel_values = torch.stack([example["instance_images"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    injected_pixel_values = torch.stack([example["injected_images"] for example in examples])
    injected_pixel_values = injected_pixel_values.to(memory_format=torch.contiguous_format).float()


    conditioning_pixel_values = torch.stack([example["instance_canny"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    injected_conditioning_pixel_values = torch.stack([example["injected_canny"] for example in examples])
    injected_conditioning_pixel_values = injected_conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()


    input_ids = torch.stack([example["instance_texts"] for example in examples])
    injected_input_ids = torch.stack([example["injected_texts"] for example in examples])
    optimized_inputs_ids = torch.stack([example["optimized_texts"] for example in examples])
    return {
        "pixel_values": pixel_values,
        "injected_pixel_values": injected_pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "injected_conditioning_pixel_values": injected_conditioning_pixel_values,
        "input_ids": input_ids,
        "injected_input_ids": injected_input_ids,
        "optimized_input_ids": optimized_inputs_ids,
    }

def encode_prompt(text_encoder, input_ids, attention_mask=None, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


text_encoder_cls = import_model_class_from_model_name_or_path(Config.PretrainedModelPath, None)
text_encoder = text_encoder_cls.from_pretrained(
    Config.PretrainedModelPath, subfolder="text_encoder", revision=None, variant=None
).to(Config.TextTrainDevice)
teacher_text_encoder = text_encoder_cls.from_pretrained(
    Config.PretrainedModelPath, subfolder="text_encoder", revision=None, variant=None
).to(Config.TextTrainDevice)
tokenizer = AutoTokenizer.from_pretrained(
            Config.PretrainedModelPath, subfolder="tokenizer", revision=None, use_fast=False,
)
train_dataset = TrainDataset(Config, tokenizer)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=collate_fn,
    batch_size=Config.TextTrainBatchSize,
    num_workers=0,
)
optimizer = torch.optim.AdamW(text_encoder.parameters(), lr=Config.TextTrainLearningRate)
prompt_loss = SimilarityLoss()
progress_bar = tqdm(
    range(0, ((len(train_dataset)//Config.TextTrainBatchSize)*Config.TextTrainSteps)*Config.TextTrainEpochs),
    initial=0,
    desc="Steps",
)
text_encoder.train()
teacher_text_encoder.eval()
for epoch in range(0, Config.TextTrainEpochs):
    for step, batch in enumerate(train_dataloader):
        # print("start optimize text encoder...")
        for text_encoder_train_step in range(Config.TextTrainSteps):
            optimizer.zero_grad()
            normal_encoder_hidden_states = encode_prompt(
                teacher_text_encoder,
                batch["optimized_input_ids"],
            )
            encoder_hidden_states = encode_prompt(
                text_encoder,
                batch["injected_input_ids"],
            )
            text_loss = prompt_loss(normal_encoder_hidden_states, encoder_hidden_states)
            text_loss.backward()
            optimizer.step()
            progress_bar.update(1)
            logs = {"loss": text_loss.detach().item()}
            progress_bar.set_postfix(**logs)

text_encoder.eval()
text_encoder.save_pretrained(Config.TextEncoderOutputPath, subfolder="text_encoder")