# test_with_schema.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

base_path = os.path.expanduser("~/Programming/sql-generator")
checkpoint_path = os.path.join(base_path, "models/sql-generator-full/checkpoint-500")
base_model_name = "codellama/CodeLlama-7b-Instruct-hf"

print("모델 로드 중...\n")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(base_model, checkpoint_path)
model.eval()
print("✅ 로드 완료\n")

# 스키마 정의
schema = """Database Schema:
Tables:
- users (user_id, name, email, status, created_at)
- products (product_id, name, price, category, created_at)
- orders (order_id, user_id, product_id, quantity, order_date)
"""

# 테스트 케이스
test_cases = [
    ("How many users are there?", "SELECT COUNT(*) FROM users"),
    ("Show all products", "SELECT * FROM products"),
    ("List products with price greater than 10000", "SELECT * FROM products WHERE price > 10000"),
    ("Show active users only", "SELECT * FROM users WHERE status = 'active'"),
    ("Get the latest 5 products", "SELECT * FROM products ORDER BY created_at DESC LIMIT 5"),
    ("Show top 10 most expensive products", "SELECT * FROM products ORDER BY price DESC LIMIT 10"),
    ("What is the total number of users?", "SELECT COUNT(*) FROM users"),
    ("Calculate average product price", "SELECT AVG(price) FROM products"),
    ("Count products by category", "SELECT category, COUNT(*) FROM products GROUP BY category"),
    ("Show users with their orders", "SELECT users.*, orders.* FROM users JOIN orders ON users.user_id = orders.user_id"),
]

correct = 0
total = len(test_cases)

print("=" * 70)
print(f"올바른 형식으로 테스트 ({total}개)")
print("=" * 70 + "\n")

for i, (question, expected) in enumerate(test_cases, 1):
    print(f"[{i}/{total}] {question}")
    
    # 훈련 시와 동일한 형식!
    prompt = f"""{schema}
Question: {question}

SQL Query:"""
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # "SQL Query:" 이후 부분 추출
    if "SQL Query:" in result:
        generated = result.split("SQL Query:", 1)[1].strip()
    else:
        generated = result[len(prompt):].strip()
    
    # 첫 줄만 (SQL 쿼리)
    generated_sql = generated.split('\n')[0].strip()
    
    # 평가
    is_correct = generated_sql.upper().startswith("SELECT")
    
    if is_correct:
        correct += 1
        status = "✅"
    else:
        status = "❌"
    
    print(f"  기대: {expected}")
    print(f"  생성: {generated_sql}")
    print(f"  {status}\n")

print("=" * 70)
print(f"정확도: {correct}/{total} ({correct/total*100:.1f}%)")
print("=" * 70)

# 상세 예시
print("\n" + "=" * 70)
print("상세 예시 (첫 번째 질문)")
print("=" * 70)

question = test_cases[0][0]
prompt = f"""{schema}
Question: {question}

SQL Query:"""

print(f"\n입력 프롬프트:\n{prompt}\n")

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

full_result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"전체 생성 결과:\n{full_result}\n")
