# test_model_efficient.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os
import gc

base_path = os.path.expanduser("~/Programming/sql-generator")
checkpoint_path = os.path.join(base_path, "models/sql-generator-full/checkpoint-500")
base_model_name = "codellama/CodeLlama-7b-Instruct-hf"

print("=" * 70)
print("SQL Generator 테스트 (메모리 최적화)")
print("=" * 70)

# 메모리 정리
print("\n메모리 정리 중...")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("✅ 완료\n")

print(f"베이스 모델: {base_model_name}")
print(f"어댑터: {checkpoint_path}\n")

# 1. 토크나이저
print("1️⃣  토크나이저 로드...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
print("   ✅ 완료")

# 2. 베이스 모델 (CPU + float16)
print("\n2️⃣  베이스 모델 로드 (CPU 모드)...")
print("   메모리 사용: ~7-8GB")
print("   속도: 느리지만 안정적\n")

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="cpu",  # CPU 강제
    low_cpu_mem_usage=True,
)

print("   ✅ 완료")

# 3. LoRA 어댑터
print("\n3️⃣  LoRA 어댑터 로드...")
model = PeftModel.from_pretrained(base_model, checkpoint_path)
model.eval()
print("   ✅ 완료\n")

print("=" * 70)
print("✅ 모델 준비 완료!")
print("=" * 70 + "\n")

# 4. 테스트
test_queries = [
    "모든 사용자를 조회해줘",
    "가장 최근 주문 10개를 보여줘",
    "이름이 김으로 시작하는 사용자를 찾아줘",
]

for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*70}")
    print(f"[{i}/{len(test_queries)}] {query}")
    print('='*70)
    
    # 프롬프트
    prompt = f"[INST] {query} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 생성
    print("생성 중... (CPU 모드는 1-2분 소요)")
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # 디코딩
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 결과 추출
        if "[/INST]" in full_text:
            generated = full_text.split("[/INST]", 1)[1].strip()
        else:
            generated = full_text[len(prompt):].strip()
        
        print(f"\n결과:\n{generated}\n")
        
        # 메모리 정리
        gc.collect()
        
    except Exception as e:
        print(f"\n❌ 오류: {e}\n")
        continue

print("=" * 70)
print("✅ 테스트 완료!")
print("=" * 70)
