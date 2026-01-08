#!/usr/bin/env python3
# compare_base_vs_finetuned.py
# Base 모델 vs Fine-tuned 모델 비교

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("="*70)
print("⚔️  대결: Base vs Fine-tuned")
print("="*70)

# 테스트 케이스
test_prompt = """# Given the database schema:
CREATE TABLE fury_projects (
    projectId VARCHAR(100) PRIMARY KEY,
    projectName VARCHAR(100),
    teamId VARCHAR(100),
    showFront INT
)

# Question: How many projects are there?

# SQL:
"""

print("\n📝 프롬프트:")
print(test_prompt)

# 1. Base 모델 (Fine-tuning 없음)
print("\n" + "="*70)
print("🔵 Base CodeLlama-7B (Fine-tuning 없음)")
print("="*70)

try:
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "codellama/CodeLlama-7b-Instruct-hf",
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True
    )
    
    print("✅ Base 모델 로드!")
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to(base_model.device)
    
    with torch.no_grad():
        outputs = base_model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "# SQL:" in result:
        sql = result.split("# SQL:")[-1].strip().split('\n')[0]
    else:
        sql = result.strip().split('\n')[-1]
    
    print(f"\n💾 Base 모델 SQL:")
    print(sql)
    
    # 평가
    has_select = "SELECT" in sql.upper()
    has_from = "FROM" in sql.upper()
    has_count = "COUNT" in sql.upper()
    correct_table = "fury_projects" in sql.lower()
    no_weird_alias = "AS active" not in sql.lower()
    
    score_base = sum([has_select, has_from, has_count, correct_table, no_weird_alias])
    
    print(f"\n✅ 평가:")
    print(f"   SELECT: {'✅' if has_select else '❌'}")
    print(f"   FROM: {'✅' if has_from else '❌'}")
    print(f"   COUNT(*): {'✅' if has_count else '❌'}")
    print(f"   올바른 테이블: {'✅' if correct_table else '❌'}")
    print(f"   이상한 alias 없음: {'✅' if no_weird_alias else '❌'}")
    print(f"   점수: {score_base}/5 {'⭐' * score_base}")
    
    # 메모리 정리
    del base_model
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"❌ Base 오류: {e}")
    score_base = 0

# 2. Fine-tuned 모델
print("\n" + "="*70)
print("🔴 Fine-tuned CodeLlama-7B (Spider + Company)")
print("="*70)

try:
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
    
    base_for_ft = AutoModelForCausalLM.from_pretrained(
        "codellama/CodeLlama-7b-Instruct-hf",
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True
    )
    
    ft_model = PeftModel.from_pretrained(
        base_for_ft,
        "./models/sql-generator-spider-plus-company"
    )
    ft_model = ft_model.merge_and_unload()
    
    print("✅ Fine-tuned 모델 로드!")
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to(ft_model.device)
    
    with torch.no_grad():
        outputs = ft_model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "# SQL:" in result:
        sql = result.split("# SQL:")[-1].strip().split('\n')[0]
    else:
        sql = result.strip().split('\n')[-1]
    
    print(f"\n💾 Fine-tuned 모델 SQL:")
    print(sql)
    
    # 평가
    has_select = "SELECT" in sql.upper()
    has_from = "FROM" in sql.upper()
    has_count = "COUNT" in sql.upper()
    correct_table = "fury_projects" in sql.lower()
    no_weird_alias = "AS active" not in sql.lower()
    
    score_ft = sum([has_select, has_from, has_count, correct_table, no_weird_alias])
    
    print(f"\n✅ 평가:")
    print(f"   SELECT: {'✅' if has_select else '❌'}")
    print(f"   FROM: {'✅' if has_from else '❌'}")
    print(f"   COUNT(*): {'✅' if has_count else '❌'}")
    print(f"   올바른 테이블: {'✅' if correct_table else '❌'}")
    print(f"   이상한 alias 없음: {'✅' if no_weird_alias else '❌'}")
    print(f"   점수: {score_ft}/5 {'⭐' * score_ft}")
    
except Exception as e:
    print(f"❌ Fine-tuned 오류: {e}")
    score_ft = 0

# 최종 비교
print("\n" + "="*70)
print("🏆 최종 결과")
print("="*70)

print(f"\n🔵 Base 모델: {score_base}/5")
print(f"🔴 Fine-tuned: {score_ft}/5")

if score_base > score_ft:
    print("\n😱 Base 모델이 더 나아요!")
    print("   → Fine-tuning이 오히려 성능을 떨어뜨렸어요!")
    print("   → Spider 데이터가 문제일 수 있어요!")
elif score_base == score_ft:
    print("\n🤷 동점이에요!")
    print("   → Fine-tuning이 도움이 안 됐어요!")
else:
    print("\n✅ Fine-tuned가 더 나아요!")
    print("   → Fine-tuning이 도움됐어요!")

print("\n💡 결론:")
if score_base >= score_ft:
    print("""
   1. Base 모델 사용 고려
   2. 또는 다른 데이터로 재학습
   3. 또는 GPT-4 API 사용
""")
else:
    print("""
   Fine-tuning은 성공했지만
   프롬프트 엔지니어링이 더 필요해요
""")

print("="*70)
