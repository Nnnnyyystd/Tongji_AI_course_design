import os
import pandas as pd
from datasets import Dataset
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTokenizer
from torch.utils.data import Dataset as TorchDataset
from PIL import Image
import torch
from torchvision import transforms
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np

# 参数设置
csv_path = "./Paint4Poem-Web-famous-subset/POEM_IMAGE.csv"
image_dir = "./Paint4Poem-Web-famous-subset/images/images/images"
output_dir = "Chinese_painting_model"
#pretrained_model_path = "./stable-diffusion-v1-5"
pretrained_model_path = "./Chinese_painting_model_new"

resolution = 512
max_train_steps = 10000
train_batch_size = 1
learning_rate = 1e-5
gradient_accumulation_steps = 4
lr_scheduler = "constant"
seed = 42
save_interval = 10000

torch.manual_seed(seed)

# 加载并清洗数据
df = pd.read_csv(csv_path, encoding="utf-8")
if df.iloc[0].tolist() == df.columns.tolist():
    df = df[1:].reset_index(drop=True)
df = df.rename(columns={df.columns[0]: "image_id", df.columns[1]: "poem"})

# 自定义 PyTorch 数据集
class PoemImageDataset(TorchDataset):
    def __init__(self, dataframe, image_root, tokenizer, size=512):
        self.df = dataframe
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.size = size
        self.extensions = [".jpeg", ".jpg", ".png"]
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def find_image_path(self, image_id):
        for ext in self.extensions:
            path = os.path.join(self.image_root, image_id + ext)
            if os.path.exists(path):
                return path
        return None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.find_image_path(row["image_id"])

        if img_path is None:
            print(f"[Missing] {row['image_id']}")
            return self.__getitem__((idx + 1) % len(self.df))

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[Corrupted] {img_path} {e}")
            return self.__getitem__((idx + 1) % len(self.df))

        image = self.transform(image)

        prompt = row["poem"]
        token = self.tokenizer(prompt, padding="max_length", truncation=True, max_length=77, return_tensors="pt")

        if token.input_ids.shape[-1] != 77:
            print(f"[Token Length Error] {row['image_id']} -> {token.input_ids.shape}")

        return {
            "pixel_values": image,
            "input_ids": token.input_ids.squeeze(0),
            "attention_mask": token.attention_mask.squeeze(0)
        }

# 初始化

tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
dataset = PoemImageDataset(df, image_dir, tokenizer, size=resolution)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

pipe = StableDiffusionPipeline.from_pretrained(
    pretrained_model_path,
    torch_dtype=torch.float32,  # 👈 修改为 float32
    safety_checker=None
)

unet = pipe.unet
vae = pipe.vae
text_encoder = pipe.text_encoder

optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)
accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
unet, optimizer, dataloader, text_encoder, vae = accelerator.prepare(unet, optimizer, dataloader, text_encoder, vae)

# 训练主循环
step = 0
for epoch in range(999):
    for batch in tqdm(dataloader):
        with accelerator.accumulate(unet):
            images = batch["pixel_values"].to(
                dtype=torch.float16 if pipe.vae.dtype == torch.float16 else torch.float32,
                device=accelerator.device)

            input_ids = batch["input_ids"].to(accelerator.device)

            latents = vae.encode(images).latent_dist.sample() * 0.18215
            if torch.isnan(latents).any() or torch.isinf(latents).any():
                print(f"[NaN in latents] step={step}")
                exit()

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = text_encoder(input_ids)[0].to(accelerator.device)
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            loss = torch.nn.functional.mse_loss(model_pred, noise, reduction="mean")

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[NaN in loss] step={step}")
                exit()

            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % 10 == 0:
                print(f"[Step {step}] Loss: {loss.item():.6f}")

            if step % save_interval == 0:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
                pipe.save_pretrained(checkpoint_dir)
                print(f"Checkpoint saved to: {checkpoint_dir}")

        if step >= max_train_steps:
            break
    if step >= max_train_steps:
        break

pipe.save_pretrained(output_dir)
print(f"Finished training. Model saved to: {output_dir}")
