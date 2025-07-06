
import os
import csv
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# ====== 配置路径 ======
#image_folder = "./generated_full_finetuned"  # 图像目录
image_folder = "./generated_frozen_finetuned"  # 图像目录
csv_path = "./ss.csv"  # 包含表头的 CSV 文件

# ====== 读取 CSV 文件（带表头）======
captions = {}
with open(csv_path, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        fname = row["file_name"].strip() + ".png"  # ✅ 加上 .png
        prompt = row["text"].strip()
        captions[fname] = prompt
print(f"📋 提示词总数: {len(captions)}")
print("📌 示例：", list(captions.items())[:3])

# ====== 加载 CLIP 模型 ======
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ====== 遍历图像并计算 CLIPScore ======
scores = []
matched = 0

for filename in tqdm(os.listdir(image_folder)):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        prompt = captions.get(filename)
        if not prompt:
            print(f"[!] 没找到对应 prompt: {filename}")
            continue

        matched += 1
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            score = outputs.logits_per_image.item()
            scores.append(score)

# ====== 输出结果 ======
if scores:
    avg_score = sum(scores) / len(scores)
    print(f"\n✅ 平均 CLIPScore: {avg_score:.4f}，共计图像数: {len(scores)}，成功匹配: {matched}")
else:
    print("❌ 没有成功计算 CLIPScore")

