



import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
from tqdm import tqdm

# ======= 配置路径 =======
model_path = "./Chinese_painting_model"  # 全量微调后模型目录
prompt_file = "./ss.csv"                 # 提示词文件
output_dir = "./generated_frozen_finetuned"

# ======= 设置生成参数 =======
num_inference_steps = 50
guidance_scale = 7.5

# ======= 创建输出目录 =======
os.makedirs(output_dir, exist_ok=True)

# ======= 加载模型 =======
pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

# ======= 加载提示词（CSV 格式）=======
prompts = []
with open(prompt_file, "r", encoding="utf-8") as f:
    for line in f:
        if "," in line:
            image_id, text = line.strip().split(",", 1)
            prompts.append((image_id.strip(), text.strip()))

# ======= 批量生成图像 =======
for img_id, prompt in tqdm(prompts, desc="📷 正在生成图像"):
    image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    image.save(os.path.join(output_dir, f"{img_id}.png"))

print(f"\n✅ 所有图像已保存至: {output_dir}")






