"""SQL Generation Service - Maximum Debugging"""
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
            
            # 🔍 전체 결과 저장 (디버깅용)
            print(f"\n   📄 전체 생성 결과 ({len(result)} chars):")
            print("   " + "="*70)
            
            # 프롬프트 제거하고 답변만 추출
            if "### Answer" in result:
                answer_part = result.split("### Answer")[-1]
                print(f"   [Answer 이후 ({len(answer_part)} chars)]")
                print(answer_part[:500])
            else:
                print("   [마지막 800자]")
                print(result[-800:])
            
            print("   " + "="*70)
            
            sql = self._extract_sql(result)
            
            print(f"\n   ✅ 추출된 SQL:")
            print(f"   {sql}")
            
            return sql
            
        except Exception as e:
            print(f"\n   ❌ SQL 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return f"SELECT * FROM {tables[0]['name']} LIMIT 10"
    
    def _extract_sql(self, result):
        """SQL 추출 - 강력한 정규식 사용"""
        try:
            # 방법 1: SELECT ... FROM ... 정규식으로 직접 추출
            pattern = r'SELECT\s+.*?FROM\s+.*?(?:WHERE\s+.*?)?(?:;|\n\n|$)'
            matches = re.findall(pattern, result, re.IGNORECASE | re.DOTALL)
            
            if matches:
                # 가장 마지막 매칭 (최신 생성)
                sql = matches[-1]
                sql = self._clean_sql(sql)
                
                if sql and 'SELECT' in sql.upper():
                    print(f"   [정규식으로 추출 성공]")
                    return sql
            
            # 방법 2: ```sql 블록
            if "```sql" in result:
                sql = result.split("```sql")[-1].split("```")[0].strip()
                if sql and sql.upper().startswith('SELECT'):
                    print(f"   [코드 블록에서 추출]")
                    return self._clean_sql(sql)
            
            # 방법 3: ### Answer 이후에서 SELECT 찾기
            if "### Answer" in result:
                after_answer = result.split("### Answer")[-1]
                sql = self._find_select_in_text(after_answer)
                if sql:
                    print(f"   [Answer 섹션에서 추출]")
                    return self._clean_sql(sql)
            
            # 방법 4: 전체 텍스트에서 SELECT 찾기
            sql = self._find_select_in_text(result)
            if sql:
                print(f"   [전체 텍스트에서 추출]")
                return self._clean_sql(sql)
            
            raise ValueError("No valid SQL found")
            
        except Exception as e:
            print(f"   ⚠️  SQL 추출 실패: {e}")
            raise
    
    def _find_select_in_text(self, text):
        """텍스트에서 SELECT 문 찾기"""
        lines = text.strip().split('\n')
        sql_lines = []
        in_sql = False
        
        for line in lines:
            line_stripped = line.strip()
            
            # SELECT 발견
            if line_stripped.upper().startswith('SELECT'):
                in_sql = True
                sql_lines = [line_stripped]
                continue
            
            # SQL 중간
            if in_sql:
                # 빈 줄이나 새 섹션 시작이면 종료
                if not line_stripped or line_stripped.startswith('#'):
                    break
                
                sql_lines.append(line_stripped)
                
                # 세미콜론이면 종료
                if ';' in line_stripped:
                    break
        
        if sql_lines:
            return ' '.join(sql_lines)
        
        return None
    
    def _clean_sql(self, sql):
        """SQL 정리"""
        # 코드 블록 마커 제거
        sql = sql.replace('```sql', '').replace('```', '')
        
        # 세미콜론 제거
        if ';' in sql:
            sql = sql.split(';')[0]
        
        # 여러 공백을 하나로
        sql = re.sub(r'\s+', ' ', sql)
        
        # 앞뒤 공백 제거
        sql = sql.strip()
        
        return sql
