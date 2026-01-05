# comprehensive_test.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

base_path = os.path.expanduser("~/Programming/sql-generator")
checkpoint_path = os.path.join(base_path, "models/sql-generator-full/checkpoint-500")
base_model_name = "codellama/CodeLlama-7b-Instruct-hf"

print("모델 로드 중...\n")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(base_model, checkpoint_path)
model.eval()

print("✅ 로드 완료\n")

# 다양한 난이도의 테스트
test_cases = [
    # 기본 조회
    ("모든 상품을 보여줘", "SELECT * FROM products"),
    ("사용자 정보 조회", "SELECT * FROM users"),
    
    # 조건 검색
    ("가격이 10000원 이상인 상품", "SELECT * FROM products WHERE price >= 10000"),
    ("활성 사용자만 조회", "SELECT * FROM users WHERE status = 'active'"),
    
    # 정렬과 제한
    ("최신 상품 5개", "SELECT * FROM products ORDER BY created_at DESC LIMIT 5"),
    ("가격이 높은 순으로 10개", "SELECT * FROM products ORDER BY price DESC LIMIT 10"),
    
    # 집계
    ("총 사용자 수", "SELECT COUNT(*) FROM users"),
    ("상품 평균 가격", "SELECT AVG(price) FROM products"),
    
    # 그룹화
    ("카테고리별 상품 수", "SELECT category, COUNT(*) FROM products GROUP BY category"),
    
    # 조인
    ("사용자와 주문 정보", "SELECT users.*, orders.* FROM users JOIN orders ON users.id = orders.user_id"),
]

correct = 0
total = len(test_cases)

print("=" * 70)
print(f"종합 테스트 ({total}개 질문)")
print("=" * 70 + "\n")

for i, (question, expected) in enumerate(test_cases, 1):
    print(f"[{i}/{total}] {question}")
    
    prompt = f"[INST] {question} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.3,  # 더 결정적으로
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = result.split("[/INST]", 1)[1].strip() if "[/INST]" in result else result
    
    # SQL 쿼리만 추출 (여러 줄인 경우 첫 줄만)
    generated_sql = generated.split('\n')[0].strip()
    
    # 평가 (간단히 포함 여부로)
    is_correct = generated_sql.startswith("SELECT")
    
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
