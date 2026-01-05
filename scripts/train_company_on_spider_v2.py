# train_company_on_spider.py
# Spider 모델 위에 회사 데이터 추가 학습

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import PeftModel, LoraConfig, get_peft_model
from datasets import Dataset
import json
import time
from datetime import datetime

print("="*70)
print("🚀 Spider + 회사 데이터 통합 학습")
print("="*70)

# Configuration
MODEL_NAME = "codellama/CodeLlama-7b-Instruct-hf"
SPIDER_MODEL = "../models/sql-generator-full"  # ← Spider 모델!
OUTPUT_DIR = "../models/sql-generator-spider-plus-company"
DATA_DIR = "../data"

BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
EPOCHS = 3  # 회사 데이터는 적으니까 3 에폭
LEARNING_RATE = 5e-5  # 낮게! (기존 지식 유지)
MAX_LENGTH = 512

# Device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"✅ Device: {device}")

# ========================================
# 1. Spider 모델 로드 (핵심!)
# ========================================

print(f"\n🔄 Step 1: Loading Spider-trained model...")

# 베이스 모델
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map=device,
    low_cpu_mem_usage=True
)

# Spider LoRA 로드
spider_model = PeftModel.from_pretrained(
    base_model,
    SPIDER_MODEL
)

print(f"✅ Spider model loaded!")
print(f"📚 This model knows 7000 SQL examples from Spider")

# ========================================
# 2. 추가 학습 준비
# ========================================

print(f"\n🔄 Step 2: Preparing for additional training...")

# Spider LoRA를 베이스에 병합 (깔끔한 방법)
model = spider_model.merge_and_unload()

# 새로운 LoRA 레이어 추가 (회사 데이터용)
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

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ========================================
# 3. 회사 데이터 로드
# ========================================

print(f"\n📚 Loading company data...")

with open(f'{DATA_DIR}/company_train_regenerated.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

with open(f'{DATA_DIR}/company_val_regenerated.json', 'r', encoding='utf-8') as f:
    val_data = json.load(f)

print(f"- Train: {len(train_data)} examples")
print(f"- Validation: {len(val_data)} examples")

# ========================================
# 4. 데이터 전처리
# ========================================

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
# 5. 학습 설정
# ========================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,  # ← 낮은 LR!
    warmup_steps=50,
    fp16=True,
    optim="adamw_torch",
    max_grad_norm=0.5,  # ← 작게 (기존 지식 보호)
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
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

# ========================================
# 6. 학습 시작!
# ========================================

total_steps = (len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION)) * EPOCHS

print("\n" + "="*70)
print("🚀 Starting INCREMENTAL training...")
print("="*70)
print(f"""
이 모델은:
✅ Spider 7000개 SQL (이미 학습됨)
➕ 회사 {len(train_data)}개 데이터 (지금 추가)

= Spider + 회사 통합 모델

Total steps: ~{total_steps}
Learning rate: {LEARNING_RATE} (낮게 설정 - 기존 지식 유지)

⏱️ Estimated time: 1-2 hours (MPS)
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
    
    # Save
    print(f"\n💾 Saving integrated model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Saved to: {OUTPUT_DIR}")
    
    # Eval
    eval_results = trainer.evaluate()
    
    print(f"\nFinal results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Save results
    with open(f"{OUTPUT_DIR}/training_results.txt", 'w') as f:
        f.write(f"Spider + Company Integrated Model\n")
        f.write(f"="*50 + "\n\n")
        f.write(f"Base: Spider model (7000 SQL examples)\n")
        f.write(f"Added: {len(train_data)} company examples\n")
        f.write(f"Training time: {elapsed/3600:.1f} hours\n\n")
        f.write(f"Final evaluation:\n")
        for key, value in eval_results.items():
            f.write(f"  {key}: {value:.4f}\n")
    
    print("\n" + "="*70)
    print("🎉 All done!")
    print("="*70)
    print(f"""
통합 모델 저장 완료!
- Location: {OUTPUT_DIR}
- Spider 7000개 + 회사 {len(train_data)}개

이제 이 모델은:
✅ 영어 SQL (Spider)
✅ 한국어 질문 (회사)
둘 다 잘 이해합니다!

테스트: python test_integrated_model.py
    """)

except KeyboardInterrupt:
    print("\n⚠️ Interrupted")
    trainer.save_model(f"{OUTPUT_DIR}/interrupted")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
