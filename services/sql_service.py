"""SQL Generation Service - Full Debug"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.prompts import SQL_GENERATION_PROMPT
from config.settings import MODEL_CONFIG
import re

class SQLService:
    """SQL 생성 서비스"""
    
    def __init__(self):
        print("🔄 SQLCoder 로딩...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CONFIG['model_id'],
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIG['model_id'],
            torch_dtype=torch.float16,
            device_map=MODEL_CONFIG['device_map'],
            load_in_8bit=MODEL_CONFIG['load_in_8bit'],
            trust_remote_code=True
        )
        
        self.model = model
        print("✅ SQLCoder 로드 완료!")
    
    def generate(self, question, tables, hints=None):
        """SQL 생성"""
        try:
            schema = "\n\n".join([t["schema"] for t in tables])
            
            prompt = SQL_GENERATION_PROMPT.format(
                question=question,
                schema=schema
            )
            
            if hints:
                hints_text = "\n\n### Additional Context\n"
                for hint in hints:
                    hints_text += f"{hint}\n"
                prompt = prompt + hints_text
            
            prompt_text = str(prompt).strip()
            
            print(f"\n   📏 Prompt length: {len(prompt_text)} chars")
            
            inputs = self.tokenizer.encode(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                add_special_tokens=True
            )
            
            print(f"   📊 Input tokens: {inputs.shape[1]}")
            
            inputs = inputs.to(self.model.device)
            
            print(f"   🎲 Generating with temperature={MODEL_CONFIG['temperature']}, max_new_tokens={MODEL_CONFIG['max_new_tokens']}...")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=MODEL_CONFIG['max_new_tokens'],
                    temperature=MODEL_CONFIG['temperature'],
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_beams=1,
                    early_stopping=False
                )
            
            print(f"   📊 Output tokens: {outputs.shape[1]}")
            print(f"   📈 New tokens generated: {outputs.shape[1] - inputs.shape[1]}")
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Check if it just repeated the prompt
            if len(result) - len(prompt_text) < 50:
                print(f"   ⚠️  WARNING: Model only generated {len(result) - len(prompt_text)} new characters!")
                print(f"   Prompt ended with: ...{prompt_text[-100:]}")
                print(f"   Result ended with: ...{result[-100:]}")
            
            # Extract only NEW content
            if result.startswith(prompt_text):
                new_content = result[len(prompt_text):].strip()
                print(f"\n   ✨ NEW CONTENT ({len(new_content)} chars):")
                print("   " + "="*60)
                print(new_content[:500])
                print("   " + "="*60)
            else:
                new_content = result
            
            # Try to extract SQL from new content
            sql = self._extract_sql(new_content if new_content else result)
            
            return sql
            
        except Exception as e:
            print(f"\n   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return f"SELECT * FROM {tables[0]['name']} LIMIT 10"
    
    def _extract_sql(self, text):
        """SQL 추출"""
        # Look for SELECT ... FROM pattern
        pattern = r'SELECT.+?FROM.+?(?:WHERE.+?)?(?:;|\n\n|$)'
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        
        if not matches:
            print(f"   ⚠️  No SELECT...FROM pattern found")
            print(f"   Text to search: {text[:200]}")
            raise ValueError("No SELECT found")
        
        sql = max(matches, key=len)
        sql = sql.replace(';', '').strip()
        sql = re.sub(r'\s+', ' ', sql)
        
        print(f"   ✅ Extracted: {sql[:100]}...")
        
        return sql
