# train_quick_test.py

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import json

print("="*70)
print("🚀 SQL Generator - Quick Test (Small Dataset)")
print("="*70)

# ========================================
# Configuration - OPTIMIZED FOR SPEED
# ========================================

MODEL_NAME = "codellama/CodeLlama-7b-Instruct-hf"
OUTPUT_DIR = "../models/sql-generator-test"
DATA_DIR = "../data"

BATCH_SIZE = 1              # 작게
GRADIENT_ACCUMULATION = 4   # 작게
EPOCHS = 1                  # 1 에폭만
LEARNING_RATE = 2e-4
MAX_LENGTH = 256            # 짧게

# ========================================
# 1. Check device
# ========================================

if torch.backends.mps.is_available():
    device = "mps"
    print(f"\n✅ Using MPS (Apple Silicon GPU)")
else:
    device = "cpu"
    print(f"\n⚠️ Using CPU")

# ========================================
# 2. Load SMALL data
# ========================================

print(f"\n📚 Loading SMALL dataset...")

with open(f'{DATA_DIR}/train_small.json', 'r') as f:
    train_data = json.load(f)

with open(f'{DATA_DIR}/val_small.json', 'r') as f:
    val_data = json.load(f)

print(f"- Train: {len(train_data)} examples (small sample)")
print(f"- Validation: {len(val_data)} examples")

# ========================================
# 3. Load model and tokenizer
# ========================================

print(f"\n🔄 Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map=device,
    low_cpu_mem_usage=True
)

print(f"✅ Model loaded!")

# ========================================
# 4. LoRA configuration - SMALLER
# ========================================

print(f"\n⚙️ Configuring LoRA...")

lora_config = LoraConfig(
    r=4,  # 8 → 4 (더 작게)
    lora_alpha=8,  # 16 → 8
    target_modules=["q_proj", "v_proj"],  # 2개만
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ========================================
# 5. Prepare dataset
# ========================================

print(f"\n🔄 Preparing dataset...")

def preprocess_function(example):
    full_text = example['input'] + "\n" + example['output'] + tokenizer.eos_token
    
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,  # 256
        padding="max_length"
    )
    
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

train_dataset = train_dataset.map(
    preprocess_function,
    remove_columns=train_dataset.column_names,
    desc="Processing train"
)

val_dataset = val_dataset.map(
    preprocess_function,
    remove_columns=val_dataset.column_names,
    desc="Processing val"
)

print(f"✅ Dataset prepared!")

# ========================================
# 6. Training arguments
# ========================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    
    learning_rate=LEARNING_RATE,
    warmup_steps=20,  # 작게
    fp16=True,
    optim="adamw_torch",
    
    eval_strategy="steps",
    eval_steps=50,
    
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    
    logging_steps=10,
    logging_dir=f"{OUTPUT_DIR}/logs",
    report_to="none",
    
    dataloader_num_workers=0,
)

# ========================================
# 7. Trainer
# ========================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# ========================================
# 8. Train!
# ========================================

print("\n" + "="*70)
print("🚀 Starting QUICK TEST training...")
print("="*70)

total_steps = len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION)

print(f"""
Configuration:
- Dataset: {len(train_data)} examples (SMALL)
- Epochs: {EPOCHS}
- Total steps: ~{total_steps}
- Max length: {MAX_LENGTH}

⏱️ Estimated time: 20-30 minutes
""")

print("Starting in 3 seconds...")
import time
time.sleep(3)

try:
    start_time = time.time()
    
    trainer.train()
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("✅ Quick test complete!")
    print("="*70)
    print(f"⏱️ Time taken: {elapsed/60:.1f} minutes")
    
    # Save
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Test model saved to: {OUTPUT_DIR}")
    
    # Evaluate
    eval_results = trainer.evaluate()
    print(f"\nTest results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n" + "="*70)
    print("🎉 Quick test done! Now ready for full training.")
    print("="*70)

except KeyboardInterrupt:
    print("\n⚠️ Interrupted")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
