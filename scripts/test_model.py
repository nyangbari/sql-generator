# test_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

# 경로 설정
base_path = os.path.expanduser("~/Programming/sql-generator")
checkpoint_path = os.path.join(base_path, "models/sql-generator-full/checkpoint-500")

# 베이스 모델
base_model_name = "codellama/CodeLlama-7b-Instruct-hf"

print("=" * 70)
print("SQL Generator 모델 테스트")
print("=" * 70)
print(f"\n베이스 모델: {base_model_name}")
print(f"어댑터 경로: {checkpoint_path}\n")

# 1. 베이스 모델 로드
print("1️⃣  베이스 모델 로드 중...")
print("   (처음 실행시 약 13GB 다운로드가 필요합니다)")

try:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # 디바이스 설정
    if torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float32
        print("   ✅ Apple Silicon GPU (MPS) 사용")
    elif torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
        print("   ✅ NVIDIA GPU (CUDA) 사용")
    else:
        device = "cpu"
        torch_dtype = torch.float32
        print("   ⚠️  CPU 사용 (느릴 수 있습니다)")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=device,
        low_cpu_mem_usage=True
    )
    
    print("   ✅ 베이스 모델 로드 완료")
    
except Exception as e:
    print(f"   ❌ 오류: {e}")
    exit(1)

# 2. LoRA 어댑터 로드
print("\n2️⃣  LoRA 어댑터 로드 중...")

try:
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    print("   ✅ LoRA 어댑터 로드 완료")
except Exception as e:
    print(f"   ❌ 오류: {e}")
    exit(1)

print("\n" + "=" * 70)
print("✅ 모델 준비 완료!")
print("=" * 70 + "\n")

# 3. 테스트 실행
test_queries = [
    "모든 사용자를 조회해줘",
    "가장 최근 주문 10개를 보여줘",
    "이름이 김으로 시작하는 사용자를 찾아줘",
    "상품별 판매 개수를 집계해줘",
    "2024년 총 매출을 계산해줘"
]

print("테스트 시작\n")
print("=" * 70 + "\n")

for i, query in enumerate(test_queries, 1):
    print(f"[{i}/{len(test_queries)}]")
    print(f"질문: {query}")
    
    try:
        # CodeLlama 프롬프트 형식
        prompt = f"[INST] {query} [/INST]"
        
        # 토크나이즈
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 디코딩
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 입력 부분 제거하고 생성된 부분만 추출
        # [INST] ... [/INST] 이후 부분
        if "[/INST]" in full_text:
            generated = full_text.split("[/INST]")[-1].strip()
        else:
            generated = full_text[len(prompt):].strip()
        
        print(f"생성 결과:\n{generated}")
        
    except Exception as e:
        print(f"❌ 생성 오류: {e}")
    
    print("\n" + "=" * 70 + "\n")

print("✅ 테스트 완료!")
