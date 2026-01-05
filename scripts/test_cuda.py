import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time

print("🔄 통합 모델 로딩 (CUDA)...")

# CUDA 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Device: {device}")

if not torch.cuda.is_available():
    print("⚠️ CUDA not available! Using CPU (will be slow)")

# 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(
    "codellama/CodeLlama-7b-Instruct-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    "./models/sql-generator-spider-plus-company"
)

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")

print("✅ 모델 로드 완료!\n")

# 스키마
schema = """Database Schema:
Tables:
- users (id, name, email, created_at, role)
- tasks (id, user_id, title, description, status, priority, created_at, due_date)
- projects (id, name, description, status, owner_id, created_at)
- comments (id, task_id, user_id, content, created_at)"""

# 테스트 케이스
test_cases = [
    ("Show all users", "영어"),
    ("총 사용자 수는?", "한국어"),
    ("Count completed tasks", "영어"),
    ("완료된 작업은 몇 개야?", "한국어"),
    ("List users who created tasks", "영어"),
    ("작업을 만든 사용자 목록", "한국어"),
]

print("="*70)
print("🧪 통합 모델 테스트 (CUDA)")
print("="*70)

total_time = 0

for i, (question, lang) in enumerate(test_cases, 1):
    prompt = f"""{schema}

Question: {question}

SQL Query:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # 시간 측정
    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    elapsed = time.time() - start
    total_time += elapsed
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # SQL 추출
    if "SQL Query:" in result:
        sql = result.split("SQL Query:")[-1].strip()
    else:
        sql = result.strip()
    
    sql = sql.replace('```sql', '').replace('```', '').strip()
    sql = sql.split('\n\n')[0].strip() if '\n\n' in sql else sql
    
    print(f"\n[테스트 {i}] ({lang})")
    print(f"❓ 질문: {question}")
    print(f"💾 SQL: {sql}")
    print(f"⏱️  시간: {elapsed:.2f}초")
    print("-"*70)

print(f"\n✅ 테스트 완료!")
print(f"📊 평균 시간: {total_time/len(test_cases):.2f}초")
print(f"📊 총 시간: {total_time:.2f}초")
