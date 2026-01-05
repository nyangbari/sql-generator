# download_spider.py

from datasets import load_dataset
import json
import os

print("🔄 Downloading Spider dataset...")
print("(This may take a few minutes on first run)")

# Check data folder
os.makedirs('../data', exist_ok=True)

# Download Spider
try:
    dataset = load_dataset("spider")
    
    print(f"""
✅ Download complete!

📊 Dataset statistics:
- Train: {len(dataset['train'])} examples
- Validation: {len(dataset['validation'])} examples
""")
    
    # Show samples
    print("\n" + "="*70)
    print("📌 Sample data (3 examples)")
    print("="*70)
    
    for i in range(3):
        example = dataset['train'][i]
        print(f"\n[Example {i+1}]")
        print(f"Database: {example['db_id']}")
        print(f"Question: {example['question']}")
        print(f"SQL: {example['query']}")
        print("-"*70)
    
    # Save as JSON
    print("\n💾 Saving to JSON files...")
    
    # Train data
    train_data = [
        {
            'db_id': item['db_id'],
            'question': item['question'],
            'query': item['query']
        }
        for item in dataset['train']
    ]
    
    with open('../data/train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ train.json saved ({len(train_data)} examples)")
    
    # Validation data
    val_data = [
        {
            'db_id': item['db_id'],
            'question': item['question'],
            'query': item['query']
        }
        for item in dataset['validation']
    ]
    
    with open('../data/validation.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ validation.json saved ({len(val_data)} examples)")
    
    print(f"""
    
🎉 Data preparation complete!

Saved files:
- ../data/train.json
- ../data/validation.json

Next step: Data exploration and preprocessing
""")

except Exception as e:
    print(f"❌ Error occurred: {e}")
    print("\nTroubleshooting:")
    print("1. Check internet connection")
    print("2. Install datasets library: pip install datasets")
