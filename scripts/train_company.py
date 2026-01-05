# train_company.py - 회사 데이터 전용 훈련 (자연어 질문 포함)

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
print("🚀 회사 SQL Generator 훈련 (자연어 질문 버전)")
print("="*70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Configuration
MODEL_NAME = "codellama/CodeLlama-7b-Instruct-hf"
OUTPUT_DIR = "../models/sql-generator-company"
DATA_DIR = "../data"

BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
EPOCHS = 5
LEARNING_RATE = 2e-4
MAX_LENGTH = 512

# Device
if torch.backends.mps.is_available():
    device = "mps"
    print(f"\n✅ Using MPS (Apple Silicon GPU)")
else:
    device = "cpu"
    print(f"\n⚠️ Using CPU")

# Load data
print(f"\n📚 Loading company data (natural language version)...")

with open(f'{DATA_DIR}/company_train_natural.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

with open(f'{DATA_DIR}/company_val_natural.json', 'r', encoding='utf-8') as f:
    val_data = json.load(f)

print(f"- Train: {len(train_data)} examples")
print(f"- Validation: {len(val_data)} examples")
print(f"- Includes natural language questions! 💬")

# Load model
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

# LoRA config
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

# Prepare dataset
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

# Training arguments
print(f"\n⚙️ Setting up training...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    warmup_steps=50,
    fp16=True,
    optim="adamw_torch",
    max_grad_norm=1.0,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_steps=20,
    logging_dir=f"{OUTPUT_DIR}/logs",
    report_to="none",
    dataloader_num_workers=0,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train
total_steps = (len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION)) * EPOCHS

print("\n" + "="*70)
print("🚀 Starting training...")
print("="*70)
print(f"""
Configuration:
- Dataset: {len(train_data)} examples (회사 데이터 + 자연어 질문)
- Epochs: {EPOCHS}
- Total steps: ~{total_steps}
- Batch size: {BATCH_SIZE}
- Max length: {MAX_LENGTH}

⏱️ Estimated time:
  - CPU: 6-10 hours
  - MPS (Apple GPU): 1-2 hours

💡 Tip: 모니터링하면서 실행하세요!
""")

print("Starting in 3 seconds...")
time.sleep(3)

try:
    start_time = time.time()
    
    print("\n🚀 Training started!\n")
    trainer.train()
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("✅ Training complete!")
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
    
    # Save results
    with open(f"{OUTPUT_DIR}/training_results.txt", 'w') as f:
        f.write(f"Training completed at: {datetime.now()}\n")
        f.write(f"Total time: {elapsed/3600:.1f} hours\n")
        f.write(f"Dataset: {len(train_data)} examples (company data + natural language)\n\n")
        f.write(f"Final evaluation:\n")
        for key, value in eval_results.items():
            f.write(f"  {key}: {value:.4f}\n")
    
    print(f"\n✅ Results saved to: {OUTPUT_DIR}/training_results.txt")
    
    print("\n" + "="*70)
    print("🎉 All done! Model ready to use.")
    print("="*70)
    print(f"\nModel location: {OUTPUT_DIR}")
    print("\n다음 단계:")
    print("  1. 테스트: python test_company_model.py")
    print("  2. 질문 예시: '총 사용자 수는?', '모든 작업을 보여줘'")

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
