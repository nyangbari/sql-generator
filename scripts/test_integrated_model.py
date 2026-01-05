# test_integrated_model.py (수정 버전)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("🔄 통합 모델 로딩...")

base_model = AutoModelForCausalLM.from_pretrained(
    "codellama/CodeLlama-7b-Instruct-hf",
    torch_dtype=torch.float16,
    device_map="mps"
)

model = PeftModel.from_pretrained(
    base_model,
    "../models/sql-generator-spider-plus-company"
)

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")

print("✅ 모델 로드 완료!\n")

company_schema = """Tables:
- users (id, name, email, created_at, role)
- tasks (id, user_id, title, description, status, priority, created_at, due_date)
- projects (id, name, description, status, owner_id, created_at)
- comments (id, task_id, user_id, content, created_at)"""

test_cases = [
    ("Show all users", "영어"),
    ("총 사용자 수는?", "한국어"),
    ("Count completed tasks", "영어"),
    ("완료된 작업은 몇 개야?", "한국어"),
    ("List users who created tasks", "영어"),
    ("작업을 만든 사용자 목록", "한국어"),
]

print("="*70)
print("🧪 통합 모델 테스트 (영어 + 한국어)")
print("="*70)

for i, (question, lang) in enumerate(test_cases, 1):
    prompt = f"""Database Schema:
{company_schema}

Question: {question}

SQL Query:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")
    
    # 개선된 생성 파라미터
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,      # 늘림
        temperature=0.2,         # 약간 올림
        top_p=0.95,             # 추가
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # SQL 추출 개선
    if "SQL Query:" in result:
        sql = result.split("SQL Query:")[-1].strip()
    else:
        sql = result.strip()
    
    # Markdown 제거
    sql = sql.replace('```sql', '').replace('```', '').strip()
    
    # 빈 줄 전까지
    if '\n\n' in sql:
        sql = sql.split('\n\n')[0].strip()
    
    # 첫 SQL 문장만 (세미콜론 또는 첫 줄)
    if '\n' in sql and not sql.startswith('SELECT'):
        lines = sql.split('\n')
        sql = lines[0] if lines[0].strip() else (lines[1] if len(lines) > 1 else sql)
    
    print(f"\n[테스트 {i}] ({lang})")
    print(f"❓ 질문: {question}")
    print(f"💾 SQL: {sql}")
    print("-"*70)

print("\n✅ 테스트 완료!")
