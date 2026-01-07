#!/usr/bin/env python3
# test_13b_model.py
# 13B 모델 테스트 (실제 DB 스키마)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("="*70)
print("🧪 13B 모델 테스트")
print("="*70)

# 추천 모델들
models = {
    "1": "codellama/CodeLlama-13b-Instruct-hf",  # SQL 특화
    "2": "mistralai/Mistral-7B-Instruct-v0.2",   # 7B지만 강력
    "3": "meta-llama/Meta-Llama-3-8B-Instruct",  # 범용
}

print("\n📋 사용 가능한 모델:")
for key, model in models.items():
    print(f"  {key}. {model}")

choice = input("\n선택 (1-3, 기본값 1): ").strip() or "1"
model_id = models.get(choice, models["1"])

print(f"\n🔄 모델 로딩: {model_id}")
print("   (처음 다운로드 시 시간이 걸릴 수 있습니다...)")

try:
    # GPU 체크
    if not torch.cuda.is_available():
        print("❌ CUDA 없음! CPU로 실행됩니다 (매우 느림)")
    else:
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("\n🔄 8-bit 양자화로 로딩...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True
    )
    
    # VRAM 사용량 체크
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"\n💾 VRAM 사용:")
        print(f"   할당됨: {allocated:.2f}GB")
        print(f"   예약됨: {reserved:.2f}GB")
    
    print("\n✅ 모델 로드 성공!")
    
    # 테스트 1: 실제 스키마로 테스트
    print("\n" + "="*70)
    print("🧪 테스트 1: 미션 개수")
    print("="*70)
    
    test_prompt_1 = """Given these tables from knightfury database:

Table: fury_users
Columns: address, chainId, network, referralCode, username, isAdmin, telegramId, discordId, twitterId

Table: fury_mission_configs  
Columns: missionId, missionName, missionType, missionGroup, missionDetail, params, points

Table: fury_project_missions
Columns: id, projectId, missionId, isActive

Question: How many missions are in fury_mission_configs?

SQL:"""
    
    print(test_prompt_1)
    print("\n🤔 생성 중...")
    
    inputs = tokenizer(test_prompt_1, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sql = result.split("SQL:")[-1].strip().split('\n')[0]
    
    print(f"\n💾 생성된 SQL:")
    print(sql)
    
    # 체크포인트
    correct_table = "fury_mission_configs" in sql.lower()
    has_from = "from" in sql.lower()
    
    print(f"\n✅ 체크:")
    print(f"   FROM 절 있음: {has_from}")
    print(f"   올바른 테이블: {correct_table}")
    
    # 테스트 2: 테이블 선택 테스트
    print("\n" + "="*70)
    print("🧪 테스트 2: 복잡한 테이블 선택")
    print("="*70)
    
    test_prompt_2 = """Given these tables:

Table: fury_users
Columns: address, username, isAdmin

Table: fury_mission_configs
Columns: missionId, missionName, points

Table: fury_airdrop_projects
Columns: projectId, projectName, totalSupply

Table: fury_play_games
Columns: gameId, gameName, maxScore

Question: 얼마나 많은 프로젝트가 있어?

SQL:"""
    
    print(test_prompt_2)
    print("\n🤔 생성 중...")
    
    inputs = tokenizer(test_prompt_2, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sql = result.split("SQL:")[-1].strip().split('\n')[0]
    
    print(f"\n💾 생성된 SQL:")
    print(sql)
    
    # 체크포인트
    correct_table = "fury_airdrop_projects" in sql.lower()
    has_from = "from" in sql.lower()
    
    print(f"\n✅ 체크:")
    print(f"   FROM 절 있음: {has_from}")
    print(f"   올바른 테이블 (fury_airdrop_projects): {correct_table}")
    
    print("\n" + "="*70)
    print("✅ 테스트 완료!")
    print("="*70)
    
    # 결과 평가
    print("\n📊 평가:")
    if has_from and correct_table:
        print("   ✅ 이 모델은 7B보다 훨씬 좋습니다!")
        print("   ✅ LangChain에 사용 가능!")
    elif has_from:
        print("   ⚠️  FROM 절은 있지만 테이블 선택이 부정확")
        print("   ⚠️  7B보다는 나을 수 있음")
    else:
        print("   ❌ 7B와 비슷한 문제 발생")
        print("   ❌ 더 큰 모델 필요")
    
except Exception as e:
    print(f"\n❌ 오류: {e}")
    
    if "out of memory" in str(e).lower():
        print("\n💡 VRAM 부족!")
        print("   해결책: 4-bit 양자화 시도")
        print("   코드: load_in_4bit=True")
    else:
        print("\n💡 해결책:")
        print("   1. 모델 다운로드 실패 → 인터넷 연결 확인")
        print("   2. CUDA 오류 → GPU 드라이버 확인")
    
    import traceback
    traceback.print_exc()
