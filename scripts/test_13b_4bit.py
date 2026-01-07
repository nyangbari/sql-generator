#!/usr/bin/env python3
# test_13b_4bit.py
# 13B 모델 4-bit 양자화 테스트

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print("="*70)
print("🧪 13B 모델 테스트 (4-bit 양자화)")
print("="*70)

# 4-bit 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model_id = "codellama/CodeLlama-13b-Instruct-hf"

print(f"\n🔄 모델 로딩: {model_id}")
print("   양자화: 4-bit NF4")

try:
    # GPU 체크
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 모델 로드 (4-bit)
    print("\n🔄 4-bit 양자화로 로딩...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # VRAM 사용량
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"\n💾 VRAM 사용:")
        print(f"   할당됨: {allocated:.2f}GB")
        print(f"   예약됨: {reserved:.2f}GB")
    
    print("\n✅ 모델 로드 성공!")
    
    # 테스트 1
    print("\n" + "="*70)
    print("🧪 테스트 1: 미션 개수")
    print("="*70)
    
    test_prompt = """Given these tables:

Table: fury_users
Columns: address, username, isAdmin

Table: fury_mission_configs
Columns: missionId, missionName, points

Table: fury_airdrop_projects
Columns: projectId, projectName

Question: How many missions are in fury_mission_configs?

SQL:"""
    
    print(test_prompt)
    print("\n🤔 생성 중...")
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
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
    
    # 체크
    has_from = "from" in sql.lower()
    correct_table = "fury_mission_configs" in sql.lower()
    
    print(f"\n✅ 체크:")
    print(f"   FROM 절: {has_from}")
    print(f"   올바른 테이블: {correct_table}")
    
    # 테스트 2
    print("\n" + "="*70)
    print("🧪 테스트 2: 한글 + 테이블 선택")
    print("="*70)
    
    test_prompt2 = """Given these tables:

Table: fury_users
Columns: address, username

Table: fury_mission_configs
Columns: missionId, missionName

Table: fury_airdrop_projects
Columns: projectId, projectName, totalSupply

Question: 얼마나 많은 프로젝트가 있어?

SQL:"""
    
    print(test_prompt2)
    print("\n🤔 생성 중...")
    
    inputs = tokenizer(test_prompt2, return_tensors="pt").to(model.device)
    
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
    
    # 체크
    has_from = "from" in sql.lower()
    correct_table = "fury_airdrop_projects" in sql.lower()
    
    print(f"\n✅ 체크:")
    print(f"   FROM 절: {has_from}")
    print(f"   올바른 테이블 (projects): {correct_table}")
    
    print("\n" + "="*70)
    print("✅ 테스트 완료!")
    print("="*70)
    
    # 평가
    print("\n📊 평가:")
    if has_from and correct_table:
        print("   ✅ 13B 모델이 7B보다 훨씬 좋습니다!")
        print("   ✅ 4-bit 양자화로도 잘 작동!")
        print("   ✅ LangChain에 사용 가능!")
    elif has_from:
        print("   ⚠️  FROM 절은 있지만 테이블 선택 부정확")
        print("   ⚠️  Fine-tuning 고려")
    else:
        print("   ❌ 7B와 비슷한 문제")
        print("   ❌ Fine-tuning 필요")
    
except Exception as e:
    print(f"\n❌ 오류: {e}")
    import traceback
    traceback.print_exc()
