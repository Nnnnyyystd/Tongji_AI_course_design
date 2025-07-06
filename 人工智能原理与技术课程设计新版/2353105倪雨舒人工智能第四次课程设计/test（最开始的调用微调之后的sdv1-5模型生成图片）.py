


import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

# ✅ 路径设置
model_path = "./Chinese_painting_model_full_ema"
checkpoint_unet = f"{model_path}/checkpoint-20000/unet"
base_model = "./stable-diffusion-v1-5"  # 🟡 scheduler 改从这里加载！

# ✅ 逐模块加载
unet = UNet2DConditionModel.from_pretrained(checkpoint_unet, torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained(f"{model_path}/vae", torch_dtype=torch.float16)
text_encoder = CLIPTextModel.from_pretrained(f"{model_path}/text_encoder", torch_dtype=torch.float16)
tokenizer = CLIPTokenizer.from_pretrained(f"{model_path}/tokenizer")
scheduler = DDIMScheduler.from_pretrained(f"{base_model}/scheduler")  # ✅ 从 base 模型加载 scheduler

# ✅ 构建 pipeline
pipe = StableDiffusionPipeline(
    unet=unet,
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    scheduler=scheduler,
    safety_checker=None,
    feature_extractor=None,
).to("cuda")

# ✅ 测试生成
prompt = "白日依山尽，黄河入海流。"
output_dir = Path("test_outputs_fixed_scheduler")
output_dir.mkdir(exist_ok=True)

for i in range(20):
    with torch.autocast("cuda"):
        image = pipe(prompt, guidance_scale=7.5, num_inference_steps=30).images[0]
    image.save(output_dir / f"sample_fixed_{i+1}.png")
    print(f"✅ Saved sample_fixed_{i+1}.png")





























# from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
# from transformers import CLIPTextModel
# import torch

# # 原始路径
# base_model_path = "./stable-diffusion-v1-5"
# finetuned_model_path = "./Chinese_painting_model_full_ema"
# checkpoint_path = f"{finetuned_model_path}/checkpoint-20000"

# # 加载 base 模型结构
# pipe = StableDiffusionPipeline.from_pretrained(
#     base_model_path,
#     torch_dtype=torch.float16,
#     safety_checker=None
# )

# # 替换为微调后的 UNet
# pipe.unet = UNet2DConditionModel.from_pretrained(f"{checkpoint_path}/unet", torch_dtype=torch.float16)

# # 替换为已经保存的 Text Encoder 和 VAE
# pipe.text_encoder = CLIPTextModel.from_pretrained(f"{finetuned_model_path}/text_encoder", torch_dtype=torch.float16)
# pipe.vae = AutoencoderKL.from_pretrained(f"{finetuned_model_path}/vae", torch_dtype=torch.float16)

# # 保存为完整 pipeline，生成 model_index.json ✅
# pipe.save_pretrained(finetuned_model_path)
# print("✅ 已生成 model_index.json，并完成完整模型保存")
