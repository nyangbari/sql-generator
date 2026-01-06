#!/usr/bin/env python3
# test_real_world_queries.py
# 실제 업무 질문으로 모델 테스트

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time

print("="*70)
print("🏢 실무 질문 테스트 (터미널/항만 도메인)")
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

# 실제 회사 스키마 (터미널 관련 테이블들)
company_schema = """Database Schema:
Tables:
- PLT_BIZ (BIZ_COD, COD_TYPE, COD_NAME, CRUD_TYPE, CLASS_NAME, RETURN_TYPE, JNDI, MAX_KB, SQL, COMMENTS)
  * 비즈니스 로직 정보
  
- PLT_BIZ_PARAM (BIZ_COD, PARAM_NAME, PARAM_IDX, PARAM_DESC, IN_OUT, PARAM_TYPE)
  * 비즈니스 파라미터 정보
  
- BLOCK_EQUIP_SZ (블록별 장비 크기 정보)
- BLOCK_JOB_OVER (블록별 작업량 초과 정보)
- EMPTY_BLOCK (냉동 블록 정보)
- T_EQMT (장비 정보)
- T_EQPOS (장비 위치 정보)
- T_YBLK (야드 블록 정보)
- DISTANCES (거리 정보)
- ATTR_BAY (Bay 속성 정보)

Relationships:
- PLT_BIZ.BIZ_COD = PLT_BIZ_PARAM.BIZ_COD

Domain Knowledge:
- Block: 컨테이너 보관 구역
- Bay: 베이 (블록 내 위치)
- TC (Transfer Crane): 이송 크레인
- QC (Quay Crane): 안벽 크레인
- Equipment Size: 장비 규격
- Yard: 야드 (컨테이너 보관소)
"""

# 실무 중심 테스트 케이스
real_world_questions = [
    # 장비/작업량 관련
    {
        "question": "장비당 한계 작업량 초과 Block 조회",
        "category": "작업량",
        "expected_keywords": ["BLOCK_JOB_OVER", "BLOCK", "EQUIP", "초과", "한계"],
        "difficulty": "중"
    },
    {
        "question": "장비 작업량이 한계를 넘은 블록을 찾아줘",
        "category": "작업량",
        "expected_keywords": ["BLOCK_JOB_OVER", "BLOCK"],
        "difficulty": "중"
    },
    
    # 냉동 Block 관련
    {
        "question": "냉동 Block 조회",
        "category": "냉동",
        "expected_keywords": ["EMPTY_BLOCK", "BLOCK", "냉동"],
        "difficulty": "쉬움"
    },
    {
        "question": "냉동 컨테이너 보관 블록이 어디야?",
        "category": "냉동",
        "expected_keywords": ["EMPTY_BLOCK", "BLOCK"],
        "difficulty": "쉬움"
    },
    
    # 장치/컨테이너 관련
    {
        "question": "동일 Size 장치 컨테이너 조회",
        "category": "장치크기",
        "expected_keywords": ["BLOCK_EQUIP_SZ", "T_EQMT", "SIZE", "동일"],
        "difficulty": "중"
    },
    {
        "question": "같은 크기의 장비가 있는 컨테이너를 찾아줘",
        "category": "장치크기",
        "expected_keywords": ["BLOCK_EQUIP_SZ", "T_EQMT", "SIZE"],
        "difficulty": "중"
    },
    
    # TC/Bay 위치 관련
    {
        "question": "TC 위치에 가까운 Bay 조회",
        "category": "위치",
        "expected_keywords": ["T_EQPOS", "ATTR_BAY", "DISTANCES", "TC", "BAY", "가까운"],
        "difficulty": "어려움"
    },
    {
        "question": "TC에서 가장 가까운 베이는 어디야?",
        "category": "위치",
        "expected_keywords": ["T_EQPOS", "ATTR_BAY", "DISTANCES", "TC", "BAY"],
        "difficulty": "어려움"
    },
    
    # 복합 조회
    {
        "question": "작업량 초과된 냉동 블록 찾아줘",
        "category": "복합",
        "expected_keywords": ["BLOCK_JOB_OVER", "EMPTY_BLOCK", "BLOCK"],
        "difficulty": "어려움"
    },
    {
        "question": "QC 장비가 있는 야드 블록 조회",
        "category": "복합",
        "expected_keywords": ["T_EQMT", "T_YBLK", "QC", "YARD"],
        "difficulty": "중"
    },
]

print("="*70)
print(f"🧪 실무 질문 테스트 ({len(real_world_questions)}개)")
print("="*70)

results = []

for i, test_case in enumerate(real_world_questions, 1):
    question = test_case["question"]
    category = test_case["category"]
    expected_keywords = test_case["expected_keywords"]
    difficulty = test_case["difficulty"]
    
    prompt = f"""{company_schema}

Question: {question}

SQL Query:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
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
    if '\n\n' in sql:
        sql = sql.split('\n\n')[0].strip()
    
    # 평가
    sql_upper = sql.upper()
    
    # 키워드 매칭
    keyword_matches = []
    for keyword in expected_keywords:
        if keyword.upper() in sql_upper:
            keyword_matches.append(keyword)
    
    keyword_score = len(keyword_matches) / len(expected_keywords) if expected_keywords else 0
    
    # 테이블 사용 확인
    uses_relevant_tables = False
    relevant_tables = ["BLOCK_JOB_OVER", "EMPTY_BLOCK", "BLOCK_EQUIP_SZ", 
                      "T_EQMT", "T_EQPOS", "T_YBLK", "DISTANCES", "ATTR_BAY"]
    for table in relevant_tables:
        if table in sql_upper:
            uses_relevant_tables = True
            break
    
    # JOIN 사용
    has_join = 'JOIN' in sql_upper
    
    # WHERE 조건
    has_where = 'WHERE' in sql_upper
    
    # 종합 점수 (0-100)
    score = (
        (keyword_score * 40) +  # 키워드 매칭 40%
        (30 if uses_relevant_tables else 0) +  # 관련 테이블 사용 30%
        (15 if has_where else 0) +  # WHERE 조건 15%
        (15 if has_join and category in ["복합", "위치"] else 10)  # JOIN 15%
    )
    
    # 결과 저장
    result_dict = {
        'category': category,
        'question': question,
        'sql': sql,
        'difficulty': difficulty,
        'keyword_score': keyword_score,
        'uses_relevant_tables': uses_relevant_tables,
        'has_join': has_join,
        'has_where': has_where,
        'score': score,
        'time': elapsed,
        'matched_keywords': keyword_matches
    }
    results.append(result_dict)
    
    # 출력
    grade = "🟢" if score >= 70 else "🟡" if score >= 40 else "🔴"
    print(f"\n[테스트 {i}] {category} | {difficulty} | {grade} {score:.0f}점")
    print(f"❓ 질문: {question}")
    print(f"💾 SQL:")
    print(f"   {sql[:200]}{'...' if len(sql) > 200 else ''}")
    print(f"✅ 평가:")
    print(f"   - 키워드 매칭: {len(keyword_matches)}/{len(expected_keywords)} ({keyword_score*100:.0f}%)")
    print(f"   - 매칭된 키워드: {', '.join(keyword_matches) if keyword_matches else 'None'}")
    print(f"   - 관련 테이블 사용: {'✓' if uses_relevant_tables else '✗'}")
    print(f"   - JOIN: {'✓' if has_join else '✗'}")
    print(f"   - WHERE: {'✓' if has_where else '✗'}")
    print(f"⏱️  시간: {elapsed:.2f}초")
    print("-"*70)

# 종합 평가
print("\n" + "="*70)
print("📊 종합 평가")
print("="*70)

total = len(results)
avg_score = sum(r['score'] for r in results) / total
avg_time = sum(r['time'] for r in results) / total

excellent = len([r for r in results if r['score'] >= 70])
good = len([r for r in results if 40 <= r['score'] < 70])
poor = len([r for r in results if r['score'] < 40])

print(f"\n총 테스트: {total}개")
print(f"평균 점수: {avg_score:.1f}점")
print(f"평균 생성 시간: {avg_time:.2f}초")
print(f"\n등급별 분포:")
print(f"  🟢 우수 (70점 이상): {excellent}개 ({excellent/total*100:.1f}%)")
print(f"  🟡 보통 (40-69점): {good}개 ({good/total*100:.1f}%)")
print(f"  🔴 미흡 (40점 미만): {poor}개 ({poor/total*100:.1f}%)")

# 카테고리별 분석
print("\n📈 카테고리별 평균 점수:")
categories = {}
for r in results:
    cat = r['category']
    if cat not in categories:
        categories[cat] = []
    categories[cat].append(r['score'])

for cat_name, scores in sorted(categories.items()):
    cat_avg = sum(scores) / len(scores)
    print(f"   {cat_name}: {cat_avg:.1f}점")

# 난이도별 분석
print("\n📊 난이도별 평균 점수:")
difficulties = {}
for r in results:
    diff = r['difficulty']
    if diff not in difficulties:
        difficulties[diff] = []
    difficulties[diff].append(r['score'])

for diff_name in ["쉬움", "중", "어려움"]:
    if diff_name in difficulties:
        diff_avg = sum(difficulties[diff_name]) / len(difficulties[diff_name])
        print(f"   {diff_name}: {diff_avg:.1f}점")

# 최종 판정
print("\n" + "="*70)
if avg_score >= 70:
    print("🎉 결과: 실무 질문 이해도가 매우 우수합니다!")
    print("   - 터미널/항만 도메인 지식이 잘 학습되었어요.")
    print("   - 실제 업무에 바로 사용 가능합니다!")
elif avg_score >= 50:
    print("👍 결과: 실무 질문을 어느 정도 이해하고 있습니다.")
    print("   - 기본적인 질문은 잘 처리합니다.")
    print("   - 복잡한 질문은 추가 학습이 필요할 수 있습니다.")
elif avg_score >= 30:
    print("⚠️  결과: 실무 질문 이해도가 부족합니다.")
    print("   - 도메인 특화 학습이 더 필요합니다.")
    print("   - 회사 데이터를 늘리거나 재학습을 권장합니다.")
else:
    print("❌ 결과: 실무 질문을 제대로 이해하지 못하고 있습니다.")
    print("   - 일반 SQL 지식에 의존하고 있습니다.")
    print("   - 터미널/항만 도메인 데이터 재학습이 필요합니다.")

print("="*70)

# 개선 제안
print("\n💡 개선 방법:")

if avg_score < 70:
    print("\n1. 데이터 증강:")
    print("   - 현재 회사 데이터: 1449개")
    print("   - 추천: 각 카테고리별 50-100개 추가")
    print("   - 특히 낮은 점수 카테고리 집중")

if poor > 0:
    print("\n2. 미흡 케이스 분석:")
    poor_cases = [r for r in results if r['score'] < 40]
    for case in poor_cases[:3]:
        print(f"   - {case['question']}")
        print(f"     → 학습 데이터에 유사 예시 추가 필요")

if avg_score >= 50:
    print("\n3. 다음 단계:")
    print("   - vLLM으로 추론 속도 개선 (10-20배)")
    print("   - Streamlit 웹 UI 개발")
    print("   - 실제 DB 연동 테스트")

print("\n" + "="*70)
print("테스트 완료!")
print("="*70)
