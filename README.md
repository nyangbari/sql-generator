# SQL Generator - Domain-Specific LLM

🤖 Natural language to SQL query generator using LoRA fine-tuned CodeLlama-7B.

## Features
- ✅ **Bilingual**: English + Korean support
- ✅ **Domain-specific**: Trained on Spider dataset (7000 examples) + Company data (1449 examples)
- ✅ **Efficient**: LoRA fine-tuning (~0.12% trainable parameters)
- ✅ **Cross-platform**: Runs on Mac (MPS) or Windows (CUDA)

## Training Results
- **eval_loss: 0.0711** (excellent!)
- Total examples: 8449 (7000 Spider + 1449 Company)
- Training time: ~24 hours (Mac M3 Pro MPS)

## Setup

### 1. Clone repository
```bash
git clone https://github.com/YOUR_USERNAME/sql-generator.git
cd sql-generator
```

### 2. Create virtual environment
```bash
# Mac/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download/Train models

**Option A: Train from scratch**
```bash
# Download Spider dataset
python scripts/download_spider.py

# Train on Spider only
python scripts/train_full.py

# Train on Spider + Company data
python scripts/train_company_on_spider_v2.py
```

**Option B: Use pre-trained model**
- Download model files separately (not included in repo due to size)
- Place in `models/sql-generator-spider-plus-company/`

## Usage

### Quick test
```bash
python scripts/test_integrated_model.py
```

### Generate SQL
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load model
base_model = AutoModelForCausalLM.from_pretrained(
    "codellama/CodeLlama-7b-Instruct-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    "./models/sql-generator-spider-plus-company"
)

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")

# Generate SQL
schema = """Database Schema:
Tables:
- users (id, name, email)
- tasks (id, user_id, title, status)
"""

question = "총 사용자 수는?"  # Korean: "How many users?"

prompt = f"{schema}\n\nQuestion: {question}\n\nSQL Query:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=150)
sql = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(sql)
# Output: SELECT COUNT(*) FROM users
```

## Project Structure
```
sql-generator/
├── data/                    # Training datasets
│   ├── train.json           # Spider train data
│   ├── validation.json      # Spider validation data
│   ├── company_train_regenerated.json
│   └── company_val_regenerated.json
├── scripts/                 # Python scripts
│   ├── train_full.py        # Train on Spider only
│   ├── train_company_on_spider_v2.py  # Incremental training
│   ├── test_integrated_model.py
│   └── ...
├── models/                  # Trained models (gitignored)
│   └── sql-generator-spider-plus-company/
├── .gitignore
├── requirements.txt
└── README.md
```

## Hardware Requirements

### Training
- **Minimum**: 16GB RAM, CPU
- **Recommended**: 32GB+ RAM, GPU with 16GB+ VRAM
- **Our setup**: Mac M3 Pro (36GB) or Windows RTX 4060Ti (16GB)

### Inference
- **Minimum**: 8GB RAM
- **Recommended**: 16GB+ RAM for faster inference

## Performance

| Environment | Training Time | Inference Speed |
|-------------|---------------|-----------------|
| Mac M3 Pro (MPS) | 24 hours | 5-10 sec/query |
| Windows RTX 4060Ti (CUDA) | 3-5 hours | 1-2 sec/query |

## Examples

**English:**
```
Q: Show all users who signed up last month
SQL: SELECT * FROM users WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
```

**Korean:**
```
Q: 완료된 작업은 몇 개야?
SQL: SELECT COUNT(*) FROM tasks WHERE status = 'completed'
```

## License
MIT

## Author
Your Name
