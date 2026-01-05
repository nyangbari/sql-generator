#!/usr/bin/env python3
# test_company_specific.py
# 회사 특화 데이터 학습 여부 테스트

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time

print("="*70)
print("🏢 회사 특화 모델 테스트")
print("="*70)

# Device 확인
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"✅ Device: {device}\n")

# 모델 로드
print("🔄 모델 로딩...")
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

# 실제 회사 스키마
company_schema = """Database Schema:
Tables:
- PLT_BIZ (BIZ_COD, COD_TYPE, COD_NAME, CRUD_TYPE, CLASS_NAME, RETURN_TYPE, JNDI, MAX_KB, SQL, COMMENTS, IN_ID, IN_DT, UP_ID, UP_DT)
  * BIZ_COD: 비즈니스 코드 (Primary Key)
  * COD_NAME: 비즈니스 이름
  * SQL: 실행할 SQL 쿼리

- PLT_BIZ_PARAM (BIZ_COD, PARAM_NAME, PARAM_IDX, PARAM_DESC, IN_OUT, PARAM_TYPE)
  * BIZ_COD: 비즈니스 코드 (Foreign Key)
  * PARAM_NAME: 파라미터 이름
  * PARAM_IDX: 파라미터 순서
  * IN_OUT: 입력/출력 구분

Relationships:
- PLT_BIZ.BIZ_COD = PLT_BIZ_PARAM.BIZ_COD (one-to-many)

Note: 
- PLT_BIZ 테이블: 12,104개 레코드
- PLT_BIZ_PARAM 테이블: 915개 레코드"""

# 회사 특화 테스트 케이스
test_cases = [
    # 기본 조회
    ("특정 비즈니스 코드의 파라미터 정보를 조회해줘", "BASIC_1"),
    ("파라미터 정보를 정렬해서 조회해줘", "BASIC_2"),
    
    # JOIN 쿼리
    ("비즈니스 코드와 파라미터를 조인해서 보여줘", "JOIN_1"),
    ("비즈니스 이름과 파라미터 이름을 함께 조회해줘", "JOIN_2"),
    
    # 복잡한 쿼리
    ("입력 파라미터가 있는 비즈니스 목록을 보여줘", "COMPLEX_1"),
    ("파라미터 개수별로 비즈니스를 그룹화해줘", "COMPLEX_2"),
    
    # 회사 도메인 용어
    ("BIZ_COD가 'POS_'로 시작하는 비즈니스를 찾아줘", "DOMAIN_1"),
    ("PARAM_IDX 순서대로 파라미터를 나열해줘", "DOMAIN_2"),
]

print("="*70)
print("🧪 회사 특화 테스트 시작")
print("="*70)

results = []

for i, (question, test_id) in enumerate(test_cases, 1):
    prompt = f"""{company_schema}

Question: {question}

SQL Query:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    elapsed = time.time() - start
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # SQL 추출
    if "SQL Query:" in result:
        sql = result.split("SQL Query:")[-1].strip()
    else:
        sql = result.strip()
    
    sql = sql.replace('```sql', '').replace('```', '').strip()
    
    # 평가
    correct_tables = False
    has_join = False
    has_where = False
    has_order = False
    
    sql_upper = sql.upper()
    
    if 'PLT_BIZ' in sql_upper or 'PLT_BIZ_PARAM' in sql_upper:
        correct_tables = True
    
    if 'JOIN' in sql_upper:
        has_join = True
    
    if 'WHERE' in sql_upper:
        has_where = True
        
    if 'ORDER BY' in sql_upper:
        has_order = True
    
    # 결과 저장
    result_dict = {
        'test_id': test_id,
        'question': question,
        'sql': sql,
        'correct_tables': correct_tables,
        'has_join': has_join,
        'has_where': has_where,
        'has_order': has_order,
        'time': elapsed
    }
    results.append(result_dict)
    
    # 출력
    print(f"\n[테스트 {i}] {test_id}")
    print(f"❓ 질문: {question}")
    print(f"💾 SQL:\n{sql}")
    print(f"✅ 평가:")
    print(f"   - 올바른 테이블: {'✓' if correct_tables else '✗'}")
    print(f"   - JOIN 사용: {'✓' if has_join else '✗'}")
    print(f"   - WHERE 사용: {'✓' if has_where else '✗'}")
    print(f"   - ORDER BY 사용: {'✓' if has_order else '✗'}")
    print(f"⏱️  시간: {elapsed:.2f}초")
    print("-"*70)

# 종합 평가
print("\n" + "="*70)
print("📊 종합 평가")
print("="*70)

total = len(results)
correct_tables_count = sum(1 for r in results if r['correct_tables'])
join_count = sum(1 for r in results if r['has_join'])
avg_time = sum(r['time'] for r in results) / total

print(f"\n총 테스트: {total}개")
print(f"올바른 테이블 사용: {correct_tables_count}/{total} ({correct_tables_count/total*100:.1f}%)")
print(f"JOIN 사용: {join_count}/{total} ({join_count/total*100:.1f}%)")
print(f"평균 생성 시간: {avg_time:.2f}초")

# 카테고리별 분석
print("\n📈 카테고리별 성공률:")
categories = {
    'BASIC': [r for r in results if r['test_id'].startswith('BASIC')],
    'JOIN': [r for r in results if r['test_id'].startswith('JOIN')],
    'COMPLEX': [r for r in results if r['test_id'].startswith('COMPLEX')],
    'DOMAIN': [r for r in results if r['test_id'].startswith('DOMAIN')]
}

for cat_name, cat_results in categories.items():
    if cat_results:
        cat_correct = sum(1 for r in cat_results if r['correct_tables'])
        cat_total = len(cat_results)
        print(f"   {cat_name}: {cat_correct}/{cat_total} ({cat_correct/cat_total*100:.1f}%)")

# 최종 판정
print("\n" + "="*70)
if correct_tables_count / total >= 0.8:
    print("🎉 결과: 회사 데이터 학습이 잘 되었습니다!")
    print("   - PLT_BIZ, PLT_BIZ_PARAM 테이블을 제대로 이해하고 있어요.")
elif correct_tables_count / total >= 0.5:
    print("⚠️  결과: 부분적으로 학습되었습니다.")
    print("   - 일부 케이스에서만 올바른 테이블을 사용합니다.")
    print("   - 추가 학습 데이터가 필요할 수 있습니다.")
else:
    print("❌ 결과: 회사 데이터 학습이 부족합니다.")
    print("   - Spider 일반 지식에 의존하고 있습니다.")
    print("   - 회사 데이터 재학습이 필요합니다.")

print("="*70)

print("""
💡 개선 방법:
1. 더 많은 회사 데이터 추가 (현재: 1449개)
2. 회사 특화 질문 다양화
3. 학습 에폭 증가 (현재: 3 → 5)
4. 학습률 조정
""")
