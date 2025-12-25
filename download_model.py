import os
from huggingface_hub import hf_hub_download

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

REPO_ID = "thichgidaylxd/AI_model_cots"   # 🔴 ĐỔI thành repo HF của bạn

FILES = [
    "disease_model.pkl",
    "label_encoder.pkl",
    "symptoms_list.json",
    "diseases_list.json"
]

print("📥 Downloading model from Hugging Face...")

for file in FILES:
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=file,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False
    )
    print(f"✅ Downloaded: {path}")

print("🎉 Model download complete!")
