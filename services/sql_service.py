"""SQL Generation Service - Debug Version"""
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
            
            # 🔍 디버깅 출력
            print(f"\n   📄 생성 결과 길이: {len(result)} chars")
            
            # Answer 섹션 찾기
            if "### Answer" in result:
                answer_part = result.split("### Answer")[-1]
                print(f"   📝 Answer 섹션 ({len(answer_part)} chars):")
                print("   " + "="*60)
                print(answer_part[:800])  # 앞부분 800자
                print("   " + "="*60)
            else:
                print(f"   📝 전체 결과 마지막 800자:")
                print("   " + "="*60)
                print(result[-800:])
                print("   " + "="*60)
            
            # SQL 추출
            sql = self._extract_sql(result)
            
            return sql
            
        except Exception as e:
            print(f"\n   ❌ 에러: {e}")
            import traceback
            traceback.print_exc()
            return f"SELECT * FROM {tables[0]['name']} LIMIT 10"
    
    def _extract_sql(self, text):
        """SQL 추출"""
        # Answer 섹션만 사용
        if "### Answer" in text:
            text = text.split("### Answer")[-1]
        
        # SELECT 찾기 (대소문자 무관, 공백 관대)
        pattern = r'SELECT.+?FROM.+?(?:WHERE.+?)?(?:;|\n\n|```|$)'
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        
        print(f"\n   🔍 정규식 매칭: {len(matches)}개")
        
        if matches:
            for i, m in enumerate(matches):
                clean = re.sub(r'\s+', ' ', m[:100])
                print(f"      {i+1}. {clean}...")
        
        if not matches:
            # Fallback: 수동 검색
            print(f"   ⚠️  정규식 실패, 수동 검색...")
            
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if 'SELECT' in line.upper():
                    print(f"      Line {i}: {line[:80]}")
            
            raise ValueError("No SELECT found")
        
        # 가장 긴 매칭
        sql = max(matches, key=len)
        
        # 정리
        sql = sql.replace(';', '').strip()
        sql = sql.split('```')[0]  # 코드 블록 종료 제거
        sql = re.sub(r'\s+', ' ', sql)
        
        print(f"   ✅ 추출 성공: {sql[:80]}...")
        
        return sql
