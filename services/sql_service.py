"""SQL Generation Service"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from config.prompts import SQL_GENERATION_PROMPT
from config.settings import MODEL_CONFIG

class SQLService:
    """SQL 생성 서비스"""
    
    def __init__(self):
        print("🔄 SQLCoder 로딩...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_id'])
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIG['model_id'],
            torch_dtype=torch.float16,
            device_map=MODEL_CONFIG['device_map'],
            load_in_8bit=MODEL_CONFIG['load_in_8bit']
        )
        
        self.model = model
        
        print("✅ SQLCoder 로드 완료!")
    
    def generate(self, question, tables):
        """SQL 생성"""
        schema = "\n\n".join([t["schema"] for t in tables])
        
        prompt = SQL_GENERATION_PROMPT.format(
            question=question,
            schema=schema
        )
        
        # tokenizer에 명시적으로 문자열만 전달
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MODEL_CONFIG['max_new_tokens'],
                temperature=MODEL_CONFIG['temperature'],
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # SQL 추출
        sql = self._extract_sql(result)
        
        return sql
    
    def _extract_sql(self, result):
        """생성 결과에서 SQL 추출"""
        if "```sql" in result:
            sql = result.split("```sql")[-1].split("```")[0].strip()
        else:
            after_answer = result.split("### Answer")[-1]
            lines = after_answer.strip().split('\n')
            sql_lines = []
            for line in lines:
                if line.strip().upper().startswith('SELECT') or sql_lines:
                    sql_lines.append(line)
                    if ';' in line:
                        break
            sql = '\n'.join(sql_lines).strip()
        
        sql = sql.replace('```sql', '').replace('```', '').strip()
        if ';' in sql:
            sql = sql.split(';')[0].strip()
        
        return sql
