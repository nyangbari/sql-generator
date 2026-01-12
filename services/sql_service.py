"""SQL Generation Service - Ultra Safe Version"""
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
        
        # pad_token 설정
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
    
    def generate(self, question, tables):
        """SQL 생성 - 초안전 버전"""
        try:
            schema = "\n\n".join([t["schema"] for t in tables])
            
            prompt = SQL_GENERATION_PROMPT.format(
                question=question,
                schema=schema
            )
            
            # 🎯 핵심: 완전히 안전한 문자열 변환
            if isinstance(prompt, str):
                prompt_text = prompt
            else:
                prompt_text = str(prompt)
            
            # 추가 정리
            prompt_text = prompt_text.strip()
            
            # Tokenization (최대한 안전하게)
            try:
                inputs = self.tokenizer.encode(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                    add_special_tokens=True
                )
            except Exception as e:
                print(f"⚠️  Tokenizer 에러: {e}")
                print(f"📝 Prompt 타입: {type(prompt_text)}")
                print(f"📝 Prompt 길이: {len(prompt_text)}")
                # Fallback: 더 간단한 방법
                inputs = self.tokenizer(
                    [prompt_text],  # 리스트로 감싸기
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                )['input_ids']
            
            # GPU로 이동
            inputs = inputs.to(self.model.device)
            
            # 생성
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
            
            # Decode
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # SQL 추출
            sql = self._extract_sql(result)
            
            return sql
            
        except Exception as e:
            print(f"❌ SQL 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback SQL
            return f"SELECT * FROM {tables[0]['name'] if tables else 'fury_projects'} LIMIT 10"
    
    def _extract_sql(self, result):
        """생성 결과에서 SQL 추출"""
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
            
            # 정리
            sql = sql.replace('```sql', '').replace('```', '').strip()
            if ';' in sql:
                sql = sql.split(';')[0].strip()
            
            if not sql or not sql.upper().startswith('SELECT'):
                raise ValueError("Invalid SQL")
            
            return sql
            
        except Exception as e:
            print(f"⚠️  SQL 추출 실패: {e}")
            return "SELECT * FROM fury_projects LIMIT 10"
