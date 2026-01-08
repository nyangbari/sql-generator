#!/usr/bin/env python3
# test_our_training.py
# 우리 Fine-tuning이 제대로 됐는지 확인

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("="*70)
print("🧪 우리 Fine-tuning 진단")
print("="*70)

# 모델 로드
print("\n1️⃣ 모델 로딩...")

base_model_id = "codellama/CodeLlama-7b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=True
)

model = PeftModel.from_pretrained(base_model, "./models/sql-generator-spider-plus-company")
model = model.merge_and_unload()

print("✅ 완료!")

# 테스트 케이스들 (Spider 형식 그대로)
test_cases = [
    {
        "name": "Spider 학습 형식 그대로",
        "prompt": """# Given the database schema:
CREATE TABLE singer (
    singer_id INT PRIMARY KEY,
    name VARCHAR(100),
    country VARCHAR(50),
    age INT
)

# Question: How many singers are there?

# SQL:
"""
    },
    {
        "name": "우리 DB 형식",
        "prompt": """# Given the database schema:
CREATE TABLE fury_mission_configs (
    missionId INT PRIMARY KEY,
    missionName VARCHAR(100),
    points INT
)

# Question: How many missions are there?

# SQL:
"""
    },
    {
        "name": "간단한 형식",
        "prompt": """Table: users (id, name, age)

Question: How many users?

SQL:"""
    }
]

print("\n" + "="*70)
print("🧪 테스트 시작")
print("="*70)

for i, test in enumerate(test_cases, 1):
    print(f"\n{'='*70}")
    print(f"테스트 {i}: {test['name']}")
    print(f"{'='*70}")
    
    print(f"\n📝 프롬프트:")
    print(test['prompt'])
    
    print("\n🤔 생성 중...")
    
    inputs = tokenizer(test['prompt'], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # SQL 추출
    if "# SQL:" in result:
        sql = result.split("# SQL:")[-1].strip()
    elif "SQL:" in result:
        sql = result.split("SQL:")[-1].strip()
    else:
        sql = result.strip()
    
    sql = sql.split('\n')[0].strip()
    
    print(f"\n💾 생성된 SQL:")
    print(sql)
    
    # 평가
    has_select = "SELECT" in sql.upper()
    has_from = "FROM" in sql.upper()
    has_count = "COUNT" in sql.upper()
    
    score = sum([has_select, has_from, has_count])
    
    print(f"\n✅ 평가:")
    print(f"   SELECT: {'✅' if has_select else '❌'}")
    print(f"   FROM: {'✅' if has_from else '❌'}")
    print(f"   COUNT(*): {'✅' if has_count else '❌'}")
    print(f"   점수: {score}/3 {'⭐' * score}")

print("\n" + "="*70)
print("💡 진단 결과")
print("="*70)

print("""
만약 Spider 형식은 잘 되는데 다른 형식은 안 되면:
→ 프롬프트 엔지니어링 문제

만약 다 안 되면:
→ Fine-tuning 자체 문제 (재학습 필요)

만약 다 잘 되면:
→ LangChain 통합 시 문제
""")
