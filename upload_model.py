from huggingface_hub import HfApi

api = HfApi()
repo_id = "nyangbari/sql-generator-model"

print("📤 Uploading model files...")
print("This will take 10-30 minutes...")

api.upload_folder(
    folder_path="./models/sql-generator-spider-plus-company",
    repo_id=repo_id,
    repo_type="model",
)

print(f"""
✅ Upload complete!

모델 위치: https://huggingface.co/{repo_id}
""")
