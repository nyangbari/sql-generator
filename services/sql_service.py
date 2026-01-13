"""SQL Generation Service - Clean SQL Extraction"""
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
            
            # 힌트 추가
            if hints:
                hints_text = "\n\n### IMPORTANT: Use these hints\n"
                for hint in hints:
                    hints_text += f"- {hint}\n"
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
            print(f"\n   ❌ SQL 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return f"SELECT * FROM {tables[0]['name']} LIMIT 10"
    
    def _extract_sql(self, result):
        """SQL 추출 - 깨끗하게!"""
        try:
            # Answer 섹션 찾기
            if "### Answer" in result:
                result = result.split("### Answer")[-1]
            
            # 힌트 섹션 제거
            if "### IMPORTANT" in result:
                parts = result.split("### IMPORTANT")
                result = parts[0]  # 힌트 앞부분만
            
            # 정규식으로 SELECT 문 추출
            pattern = r'(SELECT\s+.+?FROM\s+.+?)(?:\n\n|$|```)'
            matches = re.findall(pattern, result, re.IGNORECASE | re.DOTALL)
            
            if matches:
                sql = matches[-1]
                sql = self._clean_sql(sql)
                
                if sql and 'SELECT' in sql.upper() and 'FROM' in sql.upper():
                    return sql
            
            # Fallback: 수동으로 찾기
            lines = result.strip().split('\n')
            sql_lines = []
            in_sql = False
            
            for line in lines:
                line = line.strip()
                
                # 힌트 섹션 스킵
                if '### IMPORTANT' in line or line.startswith('- '):
                    continue
                
                if line.upper().startswith('SELECT'):
                    in_sql = True
                    sql_lines = [line]
                elif in_sql:
                    if not line or line.startswith('#') or line.startswith('```'):
                        break
                    sql_lines.append(line)
                    if ';' in line:
                        break
            
            if sql_lines:
                sql = ' '.join(sql_lines)
                return self._clean_sql(sql)
            
            raise ValueError("No valid SQL found")
            
        except Exception as e:
            print(f"   ⚠️  SQL 추출 실패: {e}")
            raise
    
    def _clean_sql(self, sql):
        """SQL 정리"""
        # 코드 블록 제거
        sql = sql.replace('```sql', '').replace('```', '')
        
        # 세미콜론 제거
        if ';' in sql:
            sql = sql.split(';')[0]
        
        # 여러 줄 → 한 줄
        sql = re.sub(r'\s+', ' ', sql)
        
        # 앞뒤 공백
        sql = sql.strip()
        
        # 힌트 텍스트 제거 (만약 남아있으면)
        if '### IMPORTANT' in sql:
            sql = sql.split('### IMPORTANT')[0].strip()
        
        if '- Use' in sql:
            lines = sql.split('\n')
            sql = '\n'.join([l for l in lines if not l.strip().startswith('- ')])
        
        return sql
