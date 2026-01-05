# create_small_dataset.py

import json

print("🔄 Creating small dataset for quick test...")

# Load full data
with open('../data/train_formatted.json', 'r') as f:
    train_data = json.load(f)

with open('../data/val_formatted.json', 'r') as f:
    val_data = json.load(f)

print(f"Original data:")
print(f"- Train: {len(train_data)} examples")
print(f"- Val: {len(val_data)} examples")

# Take small sample
train_small = train_data[:500]  # 500개 (전체의 7%)
val_small = val_data[:50]       # 50개

# Save
with open('../data/train_small.json', 'w', encoding='utf-8') as f:
    json.dump(train_small, f, indent=2, ensure_ascii=False)

with open('../data/val_small.json', 'w', encoding='utf-8') as f:
    json.dump(val_small, f, indent=2, ensure_ascii=False)

print(f"\n✅ Small dataset created:")
print(f"- train_small.json: {len(train_small)} examples")
print(f"- val_small.json: {len(val_small)} examples")
print(f"\nEstimated training time: ~30 minutes")
