#!/bin/bash
# Windows/Ubuntu 환경 설정 스크립트

echo "=== CUDA 테스트 ==="
python << 'PYEOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
PYEOF

echo ""
echo "=== Hugging Face 로그인 ==="
echo "토큰을 입력하세요:"
read -s HF_TOKEN

python << PYEOF
from huggingface_hub import login
login(token="$HF_TOKEN")
print("✅ Logged in!")
PYEOF

echo ""
echo "=== 모델 다운로드 ==="
python << 'PYEOF'
from huggingface_hub import snapshot_download
import os

print("📥 Downloading model...")
os.makedirs("models", exist_ok=True)

snapshot_download(
    repo_id="nyangbari/sql-generator-model",
    local_dir="./models/sql-generator-spider-plus-company"
)

print("✅ Download complete!")
PYEOF

echo ""
echo "=== 설정 완료! ==="
echo "테스트: python scripts/test_integrated_model.py"
