
import torch
import random
import numpy as np
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image
import os

# === 配置参数 ===
checkpoint_path = "./Chinese_painting_model_sd/checkpoint-20000"
base_model_path = "./stable-diffusion-v1-5"
output_dir = "./generated_images_deterministic"
prompt = "白日依山尽，黄河入海流。"
seed = 4
num_images = 1

# === 创建输出文件夹 ===
os.makedirs(output_dir, exist_ok=True)

# === 全局固定随机源（确保完全可复现）===
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# === 加载模型 ===
print("✅ 正在加载模型 checkpoint...")
unet = UNet2DConditionModel.from_pretrained(f"{checkpoint_path}/unet", torch_dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained(base_model_path, unet=unet, torch_dtype=torch.float16)
pipe.to("cuda")

# === 生成图像 ===
print(f"🎨 开始使用固定种子生成 {num_images} 张图像...")
for i in range(num_images):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5, generator=generator).images[0]
    image_path = os.path.join(output_dir, f"step20000_seed{seed}_img{i+1}.png")
    image.save(image_path)
    print(f"🖼️  已保存: {image_path}")

print("✅ 所有图像生成完毕。")