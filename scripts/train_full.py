# train_full.py - Weekend Full Training

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
import time
from datetime import datetime

print("="*70)
print("🚀 SQL Generator - FULL Training (Weekend Edition)")
print("="*70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ========================================
# Configuration - OPTIMIZED FOR OVERNIGHT
# ========================================

MODEL_NAME = "codellama/CodeLlama-7b-Instruct-hf"
OUTPUT_DIR = "../models/sql-generator-full"
DATA_DIR = "../data"

BATCH_SIZE = 1              # 느려도 안정적
GRADIENT_ACCUMULATION = 4   # 작게
EPOCHS = 3
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
# 2. Load FULL data
# ========================================

print(f"\n📚 Loading FULL dataset...")

with open(f'{DATA_DIR}/train_formatted.json', 'r') as f:
    train_data = json.load(f)

with open(f'{DATA_DIR}/val_formatted.json', 'r') as f:
    val_data = json.load(f)

print(f"- Train: {len(train_data)} examples (FULL)")
print(f"- Validation: {len(val_data)} examples")

# ========================================
# 3. Load model and tokenizer
# ========================================

print(f"\n🔄 Loading model: {MODEL_NAME}")

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
print(f"📊 Model size: {model.num_parameters() / 1e9:.2f}B parameters")

# ========================================
# 4. LoRA configuration
# ========================================

print(f"\n⚙️ Configuring LoRA...")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
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
        max_length=MAX_LENGTH,
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

print(f"\n⚙️ Setting up training...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    
    # Training
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    
    # Optimization
    learning_rate=LEARNING_RATE,
    warmup_steps=100,
    fp16=True,
    optim="adamw_torch",
    max_grad_norm=1.0,
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=500,
    
    # Saving
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    
    # Logging
    logging_steps=50,
    logging_dir=f"{OUTPUT_DIR}/logs",
    report_to="none",
    
    # Other
    dataloader_num_workers=0,
)

# ========================================
# 7. Create Trainer
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

total_steps = (len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION)) * EPOCHS

print("\n" + "="*70)
print("🚀 Starting FULL training...")
print("="*70)
print(f"""
Configuration:
- Dataset: {len(train_data)} examples (FULL)
- Epochs: {EPOCHS}
- Total steps: ~{total_steps}
- Batch size: {BATCH_SIZE}
- Gradient accumulation: {GRADIENT_ACCUMULATION}
- Max length: {MAX_LENGTH}

⏱️ Estimated time: 12-18 hours

💡 Tip: Leave it running overnight!
This script will save checkpoints every 500 steps.
""")

print("Starting in 5 seconds...")
time.sleep(5)

try:
    start_time = time.time()
    
    print("\n🚀 Training started!\n")
    trainer.train()
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("✅ FULL training complete!")
    print("="*70)
    print(f"⏱️ Total time: {elapsed/3600:.1f} hours")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save
    print(f"\n💾 Saving model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Model saved to: {OUTPUT_DIR}")
    
    # Final eval
    print(f"\n📊 Final evaluation...")
    eval_results = trainer.evaluate()
    
    print(f"\nFinal results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Save results to file
    with open(f"{OUTPUT_DIR}/training_results.txt", 'w') as f:
        f.write(f"Training completed at: {datetime.now()}\n")
        f.write(f"Total time: {elapsed/3600:.1f} hours\n")
        f.write(f"Total steps: {total_steps}\n\n")
        f.write(f"Final evaluation:\n")
        for key, value in eval_results.items():
            f.write(f"  {key}: {value:.4f}\n")
    
    print(f"\n✅ Results saved to: {OUTPUT_DIR}/training_results.txt")
    
    print("\n" + "="*70)
    print("🎉 All done! Model ready to use.")
    print("="*70)

except KeyboardInterrupt:
    print("\n\n⚠️ Training interrupted by user")
    print("Saving checkpoint...")
    trainer.save_model(f"{OUTPUT_DIR}/interrupted")
    print(f"✅ Checkpoint saved to: {OUTPUT_DIR}/interrupted")

except Exception as e:
    print(f"\n❌ Error during training: {e}")
    import traceback
    traceback.print_exc()
    
    # Try to save checkpoint
    try:
        trainer.save_model(f"{OUTPUT_DIR}/error_checkpoint")
        print(f"✅ Emergency checkpoint saved to: {OUTPUT_DIR}/error_checkpoint")
    except:
        print("❌ Could not save checkpoint")
