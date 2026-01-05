# verify_training.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

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

print("="*70)
print("학습 형식별 테스트")
print("="*70)

# Test 1: Spider 형식 (질문 있음)
print("\n[Test 1: Spider 형식 - 질문 있음]")
prompt1 = """Database Schema:
Tables:
- department (Department_ID, Name, Creation)

Question: How many departments are there?

SQL Query:"""

inputs1 = tokenizer(prompt1, return_tensors="pt").to("mps")
outputs1 = model.generate(**inputs1, max_new_tokens=100, temperature=0.1, pad_token_id=tokenizer.eos_token_id)
result1 = tokenizer.decode(outputs1[0], skip_special_tokens=True)
print(result1)

# Test 2: 회사 형식 (질문 없음)
print("\n[Test 2: 회사 형식 - 질문 없음]")
prompt2 = """Database Schema:
Tables (추출된 주요 테이블):
- ALL_BAYS
- ALL_JOBS
- T_EQMT
"""

inputs2 = tokenizer(prompt2, return_tensors="pt").to("mps")
outputs2 = model.generate(**inputs2, max_new_tokens=200, temperature=0.1, pad_token_id=tokenizer.eos_token_id)
result2 = tokenizer.decode(outputs2[0], skip_special_tokens=True)
print(result2)

print("\n" + "="*70)
