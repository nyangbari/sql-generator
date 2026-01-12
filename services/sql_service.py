"""SQL Generation Service"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.prompts import SQL_GENERATION_PROMPT
from config.settings import MODEL_CONFIG

class SQLService:
    """SQL 생성 서비스"""
    
    def __init__(self):
        print("🔄 SQLCoder 로딩...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_id'])
        
        # pad_token이 없으면 eos_token으로 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
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
        
        # 🎯 핵심: prompt를 명시적으로 str로 변환하고 단일 인자로 전달
        prompt_text = str(prompt).strip()
        
        # tokenizer 호출 (최신 버전 호환)
        inputs = self.tokenizer(
            prompt_text,  # ← str 보장
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            return_attention_mask=True
        )
        
        # GPU로 이동
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
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
