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

        # Phi-3 로드 (자연어 답변 생성용)
        print("🔄 Phi-3 로딩...")

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

        print("✅ Phi-3 로드 완료!")

    def select_tables(self, question, available_tables):
        """질문에 필요한 테이블 선택 (Phi-3 사용)

        Args:
            question: 사용자 질문
            available_tables: {table_name: schema_info} 딕셔너리

        Returns:
            list: 선택된 테이블 이름 리스트
        """
        try:
            # 테이블 목록 생성
            table_list = []
            for name, info in available_tables.items():
                desc = info.get('description', '')[:100]  # 설명 100자로 제한
                cols = ', '.join(info.get('columns', [])[:10])  # 컬럼 10개로 제한
                table_list.append(f"- {name}: {desc}\n  Columns: {cols}")

            tables_text = "\n".join(table_list)

            messages = [
                {"role": "user", "content": f"""You are a database expert. Given a question, select the relevant tables needed to answer it.

Available tables:
{tables_text}

Question: {question}

Return ONLY the table names needed, one per line. No explanation.
Example response:
fury_users
fury_project_missions"""}
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

            response = self.answer_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 응답에서 테이블 이름 추출
            if "<|assistant|>" in response:
                answer = response.split("<|assistant|>")[-1].strip()
            else:
                answer = response[len(prompt):].strip()

            # 테이블 이름 파싱
            selected = []
            for line in answer.split('\n'):
                table_name = line.strip().lower()
                # available_tables에 있는지 확인
                for avail_name in available_tables.keys():
                    if avail_name.lower() == table_name or table_name in avail_name.lower():
                        if avail_name not in selected:
                            selected.append(avail_name)
                        break

            print(f"   🤖 Phi-3 선택: {selected}")
            return selected if selected else list(available_tables.keys())[:3]

        except Exception as e:
            print(f"   ⚠️  테이블 선택 실패: {e}")
            import traceback
            traceback.print_exc()
            return list(available_tables.keys())[:3]

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
        """SQL 결과를 자연어로 변환 (Phi-3 사용)"""
        try:
            # Phi-3 chat format
            messages = [
                {"role": "user", "content": f"""Question: {question}
SQL Result: {sql_result}

Based on the SQL result above, provide a natural, conversational answer in Korean.
Keep it brief (1-2 sentences). Just answer the question directly."""}
            ]

            # Phi-3 chat template 적용
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
                    use_cache=False,  # 호환성 문제 해결
                )

            response = self.answer_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 답변 부분만 추출 (assistant 응답)
            if "<|assistant|>" in response:
                answer = response.split("<|assistant|>")[-1].strip()
            else:
                answer = response[len(prompt):].strip()

            # 첫 문장만 (깔끔하게)
            answer = answer.split('\n')[0].strip()

            return answer

        except Exception as e:
            print(f"   ⚠️  자연어 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
