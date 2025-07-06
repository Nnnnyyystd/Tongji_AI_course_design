from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# 本地模型路径
model_path = "./stable-diffusion-v1-5"

# 加载本地模型
pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    safety_checker=None  # 可选：禁用 NSFW 检测器
)

# 将模型部署到 GPU（或 CPU）
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# 中文 prompt（古诗）
prompt = "白日依山尽，黄河入海流。"  # 你可以换成任意一首古诗

# 生成图像
image = pipe(prompt).images[0]
image.save("poem_generated.png")

print("图像生成完成，保存在当前目录：poem_generated.png")
