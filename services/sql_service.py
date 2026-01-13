"""SQL Generation Service - Fixed SQL Extraction"""
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
        """SQL 추출 - 개선 버전"""
        try:
            # 방법 1: ```sql 코드 블록
            if "```sql" in result:
                sql = result.split("```sql")[-1].split("```")[0].strip()
                if sql and sql.upper().startswith('SELECT'):
                    return self._clean_sql(sql)
            
            # 방법 2: ### Answer 이후
            if "### Answer" in result:
                after_answer = result.split("### Answer")[-1]
                sql = self._find_select_statement(after_answer)
                if sql:
                    return self._clean_sql(sql)
            
            # 방법 3: 마지막 SELECT 문 찾기
            sql = self._find_select_statement(result)
            if sql:
                return self._clean_sql(sql)
            
            raise ValueError("No valid SQL found")
            
        except Exception as e:
            print(f"   ⚠️  SQL 추출 실패: {e}")
            print(f"   결과 길이: {len(result)}")
            print(f"   마지막 500자: {result[-500:]}")
            raise
    
    def _find_select_statement(self, text):
        """텍스트에서 SELECT 문 찾기"""
        lines = text.strip().split('\n')
        sql_lines = []
        found_select = False
        
        for line in lines:
            line = line.strip()
            
            # SELECT로 시작
            if line.upper().startswith('SELECT'):
                found_select = True
                sql_lines = [line]
            
            # SELECT 이후 계속
            elif found_select:
                sql_lines.append(line)
                
                # 세미콜론으로 끝
                if ';' in line:
                    break
                
                # 다음 섹션 시작 (###, ---, etc)
                if line.startswith('#') or line.startswith('---'):
                    sql_lines.pop()  # 마지막 줄 제거
                    break
        
        if sql_lines:
            return '\n'.join(sql_lines)
        
        return None
    
    def _clean_sql(self, sql):
        """SQL 정리"""
        # 코드 블록 제거
        sql = sql.replace('```sql', '').replace('```', '')
        
        # 세미콜론 제거
        if ';' in sql:
            sql = sql.split(';')[0]
        
        # 앞뒤 공백 제거
        sql = sql.strip()
        
        # 빈 줄 제거
        lines = [line for line in sql.split('\n') if line.strip()]
        sql = '\n'.join(lines)
        
        return sql
