"""SQL Generation Service - Final Version"""
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
                hints_text = "\n\n### Additional Context\n"
                for hint in hints:
                    hints_text += f"{hint}\n"
                prompt = prompt + hints_text
            
            prompt_text = str(prompt).strip()
            
            # Tokenization
            inputs = self.tokenizer.encode(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                add_special_tokens=True
            )
            
            inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=MODEL_CONFIG['max_new_tokens'],
                    temperature=MODEL_CONFIG['temperature'],
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # SQL 추출
            sql = self._extract_sql(result)
            
            return sql
            
        except Exception as e:
            print(f"   ❌ 에러: {e}")
            return f"SELECT * FROM {tables[0]['name']} LIMIT 10"
    
    def _extract_sql(self, text):
        """SQL 추출 - 최종 버전"""
        # 모든 SELECT 문 찾기 (아주 관대하게)
        pattern = r'SELECT\s+.+?FROM\s+\S+'
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        
        if not matches:
            raise ValueError("No SELECT found")
        
        # 가장 긴 매칭 (가장 완전한 SQL일 가능성)
        sql = max(matches, key=len)
        
        # WHERE 절까지 확장
        sql_start = text.find(sql)
        remaining = text[sql_start:]
        
        # WHERE, JOIN, ORDER BY, LIMIT 등 찾기
        extended = re.search(
            r'(SELECT\s+.+?FROM\s+.+?)(?:\n\n|;|```|###)',
            remaining,
            re.IGNORECASE | re.DOTALL
        )
        
        if extended:
            sql = extended.group(1)
        
        # 정리
        sql = sql.strip()
        sql = re.sub(r'\s+', ' ', sql)
        sql = sql.replace(';', '')
        
        return sql
