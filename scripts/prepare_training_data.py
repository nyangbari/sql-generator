# prepare_training_data.py

import json
from tqdm import tqdm

print("🔄 Preparing training data for LoRA fine-tuning...")

# Load data
with open('../data/train.json', 'r') as f:
    train_data = json.load(f)

with open('../data/validation.json', 'r') as f:
    val_data = json.load(f)

print(f"\n📚 Original data:")
print(f"- Train: {len(train_data)}")
print(f"- Validation: {len(val_data)}")

# Database schemas (simplified - you can expand this)
# In real scenario, you'd load actual schemas
SCHEMAS = {
    'department_management': """Tables:
- department (Department_ID, Name, Creation, Ranking, Budget_in_Billions, Num_Employees)
- head (head_ID, name, born_state, age)
- management (department_ID, head_ID, temporary_acting)""",
    
    'concert_singer': """Tables:
- stadium (Stadium_ID, Location, Name, Capacity, Highest, Lowest, Average)
- singer (Singer_ID, Name, Country, Song_Name, Song_release_year, Age, Is_male)
- concert (concert_ID, concert_Name, Theme, Stadium_ID, Year)
- singer_in_concert (concert_ID, Singer_ID)""",
    
    # Add more schemas as needed, or use a generic one
    'generic': """Tables:
(Schema information will be provided based on database)"""
}

def create_prompt(question, db_id, schema=None):
    """Create training prompt"""
    
    # Get schema or use generic
    db_schema = SCHEMAS.get(db_id, SCHEMAS['generic'])
    
    prompt = f"""Database Schema:
{db_schema}

Question: {question}

SQL Query:"""
    
    return prompt

# Convert to training format
print("\n🔄 Converting to training format...")

formatted_train = []
for item in tqdm(train_data, desc="Processing train data"):
    prompt = create_prompt(item['question'], item['db_id'])
    
    formatted_train.append({
        'input': prompt,
        'output': item['query'],
        'db_id': item['db_id']
    })

formatted_val = []
for item in tqdm(val_data, desc="Processing validation data"):
    prompt = create_prompt(item['question'], item['db_id'])
    
    formatted_val.append({
        'input': prompt,
        'output': item['query'],
        'db_id': item['db_id']
    })

# Save formatted data
print("\n💾 Saving formatted data...")

with open('../data/train_formatted.json', 'w', encoding='utf-8') as f:
    json.dump(formatted_train, f, indent=2, ensure_ascii=False)

print(f"✅ train_formatted.json saved ({len(formatted_train)} examples)")

with open('../data/val_formatted.json', 'w', encoding='utf-8') as f:
    json.dump(formatted_val, f, indent=2, ensure_ascii=False)

print(f"✅ val_formatted.json saved ({len(formatted_val)} examples)")

# Show sample
print("\n" + "="*70)
print("📌 Sample formatted data:")
print("="*70)

sample = formatted_train[0]
print(f"\n[INPUT]")
print(sample['input'])
print(f"\n[OUTPUT]")
print(sample['output'])
print("\n" + "="*70)

print(f"""
✅ Data preparation complete!

Files created:
- ../data/train_formatted.json ({len(formatted_train)} examples)
- ../data/val_formatted.json ({len(formatted_val)} examples)

Next step: LoRA fine-tuning
""")
