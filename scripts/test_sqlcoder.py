#!/usr/bin/env python3
# test_sqlcoder.py
# SQLCoder-7B-2 테스트

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("="*70)
print("🧪 SQLCoder-7B-2 테스트")
print("="*70)

model_id = "defog/sqlcoder-7b-2"

print(f"\n🔄 모델 다운로드 & 로딩: {model_id}")
print("   (처음 실행 시 다운로드에 시간이 걸립니다...)")

try:
    # SQLCoder는 특별한 프롬프트 형식 사용
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True
    )
    
    print("✅ 모델 로드 완료!")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"💾 VRAM: {allocated:.2f}GB")
    
    # 테스트 케이스들
    test_cases = [
        {
            "name": "테스트 1: 프로젝트 개수",
            "schema": """CREATE TABLE fury_projects (
    projectId VARCHAR(100) PRIMARY KEY,
    projectName VARCHAR(100),
    teamId VARCHAR(100),
    showFront INT
)""",
            "question": "How many projects are there?"
        },
        {
            "name": "테스트 2: 미션 개수",
            "schema": """CREATE TABLE fury_mission_configs (
    missionId INT PRIMARY KEY,
    missionName VARCHAR(100),
    points INT
)""",
            "question": "How many missions are there?"
        },
        {
            "name": "테스트 3: 사용자 (한글)",
            "schema": """CREATE TABLE fury_users (
    address VARCHAR(42) PRIMARY KEY,
    username VARCHAR(100),
    isAdmin INT
)""",
            "question": "얼마나 많은 사용자가 있어?"
        }
    ]
    
    print("\n" + "="*70)
    print("🧪 테스트 시작")
    print("="*70)
    
    total_score = 0
    max_score = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"{test['name']}")
        print(f"{'='*70}")
        
        # SQLCoder 프롬프트 형식
        prompt = f"""### Task
Generate a SQL query to answer the following question: `{test['question']}`

### Database Schema
{test['schema']}

### Answer
Given the database schema, here is the SQL query that answers `{test['question']}`:
````sql
"""
        
        print(f"\n💬 질문: {test['question']}")
        print(f"\n📋 스키마:\n{test['schema'][:150]}...")
        
        print("\n🤔 SQL 생성 중...")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # SQL 추출
        if "```sql" in result:
            sql = result.split("```sql")[-1].split("```")[0].strip()
        else:
            sql = result.split("### Answer")[-1].strip()
            sql = sql.split('\n')[0].strip()
        
        print(f"\n💾 생성된 SQL:")
        print(sql)
        
        # 평가
        has_select = "SELECT" in sql.upper()
        has_from = "FROM" in sql.upper()
        has_count = "COUNT" in sql.upper()
        
        # 테이블명 체크
        if "fury_projects" in test['schema']:
            correct_table = "fury_projects" in sql.lower()
        elif "fury_mission" in test['schema']:
            correct_table = "fury_mission_configs" in sql.lower()
        elif "fury_users" in test['schema']:
            correct_table = "fury_users" in sql.lower()
        else:
            correct_table = False
        
        # WHERE 환각 체크
        has_where = "WHERE" in sql.upper()
        has_condition_in_question = any(word in test['question'].lower() for word in [
            'where', 'which', 'specific', 'active', 'week', '=', '1', '2', '3'
        ])
        no_hallucinated_where = not (has_where and not has_condition_in_question)
        
        score = sum([has_select, has_from, has_count, correct_table, no_hallucinated_where])
        total_score += score
        max_score += 5
        
        print(f"\n✅ 평가:")
        print(f"   SELECT: {'✅' if has_select else '❌'}")
        print(f"   FROM: {'✅' if has_from else '❌'}")
        print(f"   COUNT(*): {'✅' if has_count else '❌'}")
        print(f"   올바른 테이블: {'✅' if correct_table else '❌'}")
        print(f"   WHERE 환각 없음: {'✅' if no_hallucinated_where else '❌'}")
        print(f"   점수: {score}/5 {'⭐' * score}")
    
    # 최종 결과
    print("\n" + "="*70)
    print("🏆 최종 평가")
    print("="*70)
    
    percentage = (total_score / max_score) * 100
    
    print(f"\n총점: {total_score}/{max_score} ({percentage:.0f}%)")
    
    if percentage >= 90:
        print("\n✅ SQLCoder 완벽해요!")
        print("   → 우리 Fine-tuned 모델 교체 추천!")
    elif percentage >= 70:
        print("\n👍 SQLCoder 괜찮아요!")
        print("   → 고려해볼 만 해요!")
    else:
        print("\n⚠️  SQLCoder도 비슷해요")
        print("   → 우리 모델 계속 써도 됨")
    
    print("\n💡 비교:")
    print(f"   우리 Fine-tuned: ~87% (26/30)")
    print(f"   SQLCoder: {percentage:.0f}% ({total_score}/{max_score})")
    
except Exception as e:
    print(f"\n❌ 오류: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
