"""SQL Generation Service - With Hints Support"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.prompts import SQL_GENERATION_PROMPT
from config.settings import MODEL_CONFIG

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
        """SQL 생성 - 힌트 지원"""
        try:
            schema = "\n\n".join([t["schema"] for t in tables])
            
            # 기본 프롬프트
            prompt = SQL_GENERATION_PROMPT.format(
                question=question,
                schema=schema
            )
            
            # 힌트 추가
            if hints:
                hints_text = "\n### Additional Hints\n" + "\n".join(f"- {h}" for h in hints)
                prompt = prompt + hints_text
            
            prompt_text = str(prompt).strip()
            
            # Tokenization
            try:
                inputs = self.tokenizer.encode(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                    add_special_tokens=True
                )
            except:
                inputs = self.tokenizer(
                    [prompt_text],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                )['input_ids']
            
            inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=MODEL_CONFIG['max_new_tokens'],
                    temperature=MODEL_CONFIG['temperature'],
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            sql = self._extract_sql(result)
            
            return sql
            
        except Exception as e:
            print(f"❌ SQL 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return f"SELECT * FROM {tables[0]['name'] if tables else 'fury_projects'} LIMIT 10"
    
    def _extract_sql(self, result):
        """SQL 추출"""
        try:
            if "```sql" in result:
                sql = result.split("```sql")[-1].split("```")[0].strip()
            else:
                after_answer = result.split("### Answer")[-1]
                lines = after_answer.strip().split('\n')
                sql_lines = []
                for line in lines:
                    line = line.strip()
                    if line.upper().startswith('SELECT') or sql_lines:
                        sql_lines.append(line)
                        if ';' in line:
                            break
                sql = '\n'.join(sql_lines).strip()
            
            sql = sql.replace('```sql', '').replace('```', '').strip()
            if ';' in sql:
                sql = sql.split(';')[0].strip()
            
            if not sql or not sql.upper().startswith('SELECT'):
                raise ValueError("Invalid SQL")
            
            return sql
            
        except:
            return "SELECT * FROM fury_projects LIMIT 10"
