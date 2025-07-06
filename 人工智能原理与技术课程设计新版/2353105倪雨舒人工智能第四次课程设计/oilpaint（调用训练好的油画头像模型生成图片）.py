# import torch
# from diffusers import StableDiffusionPipeline
# from PIL import Image
# import os

# # 模型路径（可换成最终模型目录）
# model_path = "./output_oilface_full/checkpoint-5000"
# output_dir = "./test_outputs"
# os.makedirs(output_dir, exist_ok=True)

# # 单个 prompt
# prompt = "a man"

# # 生成数量
# num_images = 10

# # 加载模型（FP16 + GPU）
# pipe = StableDiffusionPipeline.from_pretrained(
#     model_path,
#     torch_dtype=torch.float16
# ).to("cuda")

# pipe.safety_checker = None  # 如果模型内置安全检查器可跳过

# # 批量生成图片
# for i in range(num_images):
#     with torch.autocast("cuda"):
#         image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    
#     filename = f"{output_dir}/output_{i+1:02d}.png"
#     image.save(filename)
#     print(f"Saved: {filename}")

import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import os

# 配置路径 
checkpoint_path = "./output_oilface_full/checkpoint-5000"  # 可换成 checkpoint-10000
base_model_path = "./stable-diffusion-v1-5"
output_dir = "./generated_images_random"
prompt = "A traditional oil painting of a noble woman in ancient Chinese attire, standing by a lotus pond"  # 可替换为任意古诗句或英文 prompt
num_images = 50  # 生成图像数量


os.makedirs(output_dir, exist_ok=True)

# 加载 UNet 权重（从微调后的 checkpoint）
print("正在加载 UNet 权重...")
unet = UNet2DConditionModel.from_pretrained(
    f"{checkpoint_path}/unet", torch_dtype=torch.float16
)

# === 构建完整 pipeline ===
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    unet=unet,
    torch_dtype=torch.float16
).to("cuda")

# 可选：关闭 NSFW 检测
pipe.safety_checker = None

# === 开始生成 ===
print(f"🎨 正在生成 {num_images} 张图像（非固定随机种子）...")
for i in range(num_images):
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    image_path = os.path.join(output_dir, f"step5000_img_{i+1}.png")
    image.save(image_path)
    print(f" 已保存：{image_path}")

print("所有图像生成完毕。")

