"""SQL Generation Service - With JOIN validation"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.prompts import SQL_GENERATION_PROMPT_TEMPLATE, ANSWER_PROMPT
from config.settings import MODEL_CONFIG, ANSWER_MODEL_CONFIG
import re

class SQLService:
    """SQL 생성 서비스"""
    
    def __init__(self):
        # SQLCoder 로드 (SQL 생성용)
        print("🔄 SQLCoder 로딩...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CONFIG['model_id'],
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIG['model_id'],
            torch_dtype=torch.float16,
            device_map=MODEL_CONFIG['device_map'],
            load_in_8bit=MODEL_CONFIG['load_in_8bit'],
            trust_remote_code=True
        )

        print("✅ SQLCoder 로드 완료!")

        # Qwen2 로드 (자연어 답변 생성용)
        print("🔄 Qwen2 로딩...")

        self.answer_tokenizer = AutoTokenizer.from_pretrained(
            ANSWER_MODEL_CONFIG['model_id'],
            trust_remote_code=True
        )

        self.answer_model = AutoModelForCausalLM.from_pretrained(
            ANSWER_MODEL_CONFIG['model_id'],
            torch_dtype=torch.float16,
            device_map=ANSWER_MODEL_CONFIG['device_map'],
            load_in_8bit=ANSWER_MODEL_CONFIG['load_in_8bit'],
            trust_remote_code=True
        )

        print("✅ Qwen2 로드 완료!")

    def select_tables(self, question, candidates):
        """후보 테이블 중에서 필요한 테이블 선택 (Qwen2 사용)

        Args:
            question: 사용자 질문
            candidates: RAG가 선택한 후보 테이블 리스트 [{name, schema, description, columns}, ...]

        Returns:
            list: 선택된 테이블 정보 리스트 [{name, schema}, ...]
        """
        # 후보가 2개 이하면 그대로 반환 (Qwen2 불필요)
        if len(candidates) <= 2:
            print(f"   ⏭️  후보 {len(candidates)}개 - Qwen2 스킵")
            return candidates

        try:
            # 테이블 목록 생성
            table_list = []
            for c in candidates:
                desc = c.get('description', '')[:150]
                cols = ', '.join(c.get('columns', [])[:8])
                table_list.append(f"- {c['name']}\n  Purpose: {desc}\n  Columns: {cols}")

            tables_text = "\n\n".join(table_list)

            messages = [
                {"role": "user", "content": f"""You are a database expert. Select tables needed to answer the question.

Available tables:
{tables_text}

Question: {question}

Think step by step:
1. What data does the question ask for?
2. Which tables have the relevant columns?
3. Check column comments for status/type meanings.

Return ONLY the table names needed, one per line:"""}
            ]

            prompt = self.answer_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.answer_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            )

            inputs = inputs.to(self.answer_model.device)

            with torch.no_grad():
                outputs = self.answer_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.1,  # 낮은 temperature로 일관성 확보
                    do_sample=True,
                    pad_token_id=self.answer_tokenizer.eos_token_id,
                    use_cache=False,
                )

            response = self.answer_tokenizer.decode(outputs[0], skip_special_tokens=False)

            # Qwen2 응답 추출
            if "<|im_start|>assistant" in response:
                answer = response.split("<|im_start|>assistant")[-1]
                if "<|im_end|>" in answer:
                    answer = answer.split("<|im_end|>")[0]
                answer = answer.strip()
            else:
                response_clean = self.answer_tokenizer.decode(outputs[0], skip_special_tokens=True)
                if "one per line:" in response_clean:
                    answer = response_clean.split("one per line:")[-1].strip()
                else:
                    answer = response_clean[-200:].strip()

            print(f"   📝 Qwen2 raw output: {answer[:200]}")  # 디버그

            # 테이블 이름 파싱 (줄바꿈, 쉼표 모두 처리)
            candidate_names = {c['name'].lower(): c for c in candidates}
            selected = []

            # 줄바꿈과 쉼표로 분리
            parts = answer.replace(',', '\n').split('\n')
            for part in parts:
                name = part.strip().lower()
                if name in candidate_names and candidate_names[name] not in selected:
                    selected.append(candidate_names[name])

            print(f"   🤖 Qwen2 선택: {[t['name'] for t in selected]}")

            # 선택된 게 없으면 상위 3개 반환
            return selected if selected else candidates[:3]

        except Exception as e:
            print(f"   ⚠️  테이블 선택 실패: {e}")
            import traceback
            traceback.print_exc()
            return candidates[:3]

    def generate(self, question, tables, hints=None, db_type="MySQL"):
        """SQL 생성"""
        try:
            enhanced_question = question
            if hints:
                hint_text = " ".join(hints)
                enhanced_question = f"{question} ({hint_text})"

            schema = "\n\n".join([t["schema"] for t in tables])

            prompt = SQL_GENERATION_PROMPT_TEMPLATE.format(
                db_type=db_type,
                question=enhanced_question,
                schema=schema
            )

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
                    min_new_tokens=MODEL_CONFIG.get('min_new_tokens', 20),
                    temperature=MODEL_CONFIG['temperature'],
                    top_p=MODEL_CONFIG.get('top_p', 0.9),
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_beams=1,
                    early_stopping=False,
                    repetition_penalty=1.1
                )

            print(f"   📊 Generated {outputs.shape[1] - inputs.shape[1]} new tokens")

            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if result.startswith(prompt_text):
                new_content = result[len(prompt_text):].strip()
            else:
                new_content = result

            sql = self._extract_sql(new_content if new_content else result)

            # Auto-fix MySQL issues
            if db_type.upper() == "MYSQL":
                sql = self._mysql_fix(sql)

            return sql

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return f"SELECT * FROM {tables[0]['name']} LIMIT 10"
    
    def _extract_sql(self, text):
        """SQL 추출"""
        pattern = r'SELECT.+?FROM.+?(?:WHERE.+?)?(?:;|\n\n|$)'
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)

        if not matches:
            raise ValueError("No SELECT found")

        sql = max(matches, key=len)
        sql = sql.replace(';', '').strip()
        sql = re.sub(r'\s+', ' ', sql)

        return sql

    def _mysql_fix(self, sql):
        """PostgreSQL → MySQL auto-fix"""
        original = sql

        sql = re.sub(r'\s+NULLS\s+(FIRST|LAST)', '', sql, flags=re.IGNORECASE)

        offset_match = re.search(r'LIMIT\s+(\d+)\s+OFFSET\s+(\d+)', sql, re.IGNORECASE)
        if offset_match:
            limit = offset_match.group(1)
            offset = offset_match.group(2)
            sql = re.sub(r'LIMIT\s+\d+\s+OFFSET\s+\d+', f'LIMIT {offset}, {limit}', sql, flags=re.IGNORECASE)

        sql = re.sub(r'::\w+', '', sql)
        sql = re.sub(r'\bILIKE\b', 'LIKE', sql, flags=re.IGNORECASE)

        if sql != original:
            print(f"   🔧 Auto-fixed for MySQL")

        return sql

    def generate_answer(self, question, sql_result):
        """SQL 결과를 자연어로 변환 (Qwen2 사용)"""
        try:
            # Qwen2 chat format
            messages = [
                {"role": "user", "content": f"""다음 데이터베이스 쿼리 결과를 요약해주세요.

질문: {question}

결과:
{sql_result}

간단하게 한국어로 답변해주세요 (1-2문장). 핵심만 말해주세요."""}
            ]

            # Qwen2 chat template 적용
            prompt = self.answer_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.answer_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )

            inputs = inputs.to(self.answer_model.device)

            with torch.no_grad():
                outputs = self.answer_model.generate(
                    **inputs,
                    max_new_tokens=ANSWER_MODEL_CONFIG['max_new_tokens'],
                    temperature=ANSWER_MODEL_CONFIG['temperature'],
                    do_sample=True,
                    pad_token_id=self.answer_tokenizer.eos_token_id,
                    use_cache=False,
                )

            # Qwen2: skip_special_tokens=False로 마커 유지
            response = self.answer_tokenizer.decode(outputs[0], skip_special_tokens=False)

            print(f"   📝 Qwen2 raw: {response[-300:]}")  # 디버그

            # Qwen2 응답 추출 (마커: <|im_start|>assistant ... <|im_end|>)
            if "<|im_start|>assistant" in response:
                # assistant 응답 부분만 추출
                answer = response.split("<|im_start|>assistant")[-1]
                # 끝 마커 제거
                if "<|im_end|>" in answer:
                    answer = answer.split("<|im_end|>")[0]
                answer = answer.strip()
            else:
                # 특수 토큰 없이 디코딩 후 프롬프트 제거
                response_clean = self.answer_tokenizer.decode(outputs[0], skip_special_tokens=True)
                # 프롬프트에서 user 메시지 부분 찾아서 그 이후만 추출
                if "핵심만 말해주세요." in response_clean:
                    answer = response_clean.split("핵심만 말해주세요.")[-1].strip()
                else:
                    answer = response_clean[-200:].strip()

            # 첫 문장만 (깔끔하게)
            answer = answer.split('\n')[0].strip()

            # 불필요한 마커 정리
            answer = answer.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()

            return answer

        except Exception as e:
            print(f"   ⚠️  자연어 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
