# test_quick_model.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("🔄 Loading trained model...")

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "codellama/CodeLlama-7b-Instruct-hf",
    torch_dtype=torch.float16,
    device_map="mps"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "../models/sql-generator-test"
)

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")

print("✅ Model loaded!\n")

# Test cases
test_cases = [
    {
        "schema": """Tables:
- department (id, name, budget)
- employee (id, name, age, department_id)""",
        "question": "How many employees are there?"
    },
    {
        "schema": """Tables:
- department (id, name, budget)
- employee (id, name, age, department_id)""",
        "question": "Show all departments"
    },
    {
        "schema": """Tables:
- department (id, name, budget)
- employee (id, name, age, department_id)""",
        "question": "List employees older than 30"
    },
    {
        "schema": """Tables:
- department (id, name, budget)
- employee (id, name, age, department_id)""",
        "question": "What is the average age of employees?"
    },
    {
        "schema": """Tables:
- users (id, name, email, created_at)
- orders (id, user_id, amount, order_date)""",
        "question": "Show total sales this month"
    },
]

print("="*70)
print("🧪 Testing SQL Generation")
print("="*70)

for i, test in enumerate(test_cases, 1):
    prompt = f"""Database Schema:
{test['schema']}

Question: {test['question']}

SQL Query:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only SQL part
    if "SQL Query:" in result:
        sql = result.split("SQL Query:")[-1].strip()
    else:
        sql = result
    
    # Clean up
    sql = sql.split('\n')[0]  # Take first line
    
    print(f"\n[Test {i}]")
    print(f"❓ Question: {test['question']}")
    print(f"💾 Generated SQL:")
    print(f"   {sql}")
    print("-"*70)

print("\n" + "="*70)
print("✅ Testing complete!")
print("="*70)

print("""
Next steps:
1. Check if SQL queries look correct
2. If good → Run full training for the weekend
3. Command: python train_full.py
""")
