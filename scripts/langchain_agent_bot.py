#!/usr/bin/env python3
import os
import torch
from typing import Any, List, Optional
from dotenv import load_dotenv

# 모델 로딩 및 파이프라인
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from langchain_huggingface import HuggingFacePipeline

# LangChain SQL 에이전트 관련
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent

load_dotenv()

# --- [1] 보안 클래스 복구 ---
class ReadOnlySQLDatabase(SQLDatabase):
    """실제 DB 수정을 방지하는 보안 계층"""
    WRITE_KEYWORDS = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE', 'REPLACE', 'MERGE']
    
    def run(self, command: str, fetch: str = "all", **kwargs):
        sql_upper = command.upper().strip()
        # 금지 키워드 체크
        for keyword in self.WRITE_KEYWORDS:
            if keyword in sql_upper:
                raise ValueError(f"🚫 {keyword} 명령어가 감지되었습니다! SELECT만 허용됩니다.")
        
        # 허용된 시작 단어 체크 (SELECT, SHOW, DESCRIBE만 허용)
        if not any(sql_upper.startswith(k) for k in ['SELECT', 'SHOW', 'DESCRIBE']):
            raise ValueError("🚫 SELECT, SHOW, DESCRIBE 쿼리만 실행 가능합니다.")
        
        print(f"✅ 안전한 쿼리 확인됨")
        return super().run(command, fetch=fetch, **kwargs)

class LangChainAgentBot:
    def __init__(self, model_path):
        print("🚀 보안 모드 및 8-bit 최적화로 시스템을 시작합니다...")
        
        base_model_id = "codellama/CodeLlama-7b-Instruct-hf"
        
        # 모델 로드 (RTX 4060 Ti 16GB 메모리 효율 최적화)
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True
        )
        
        # LoRA 어댑터 결합
        print(f"📦 어댑터 결합 중...")
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        
        # 추론 파이프라인 설정
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        # DB 연결 (보안 클래스 적용)
        uri = os.getenv("KNIGHTFURY_DB_URI", "").replace("mysql://", "mysql+pymysql://")
        self.db = ReadOnlySQLDatabase.from_uri(uri)
        print("✅ 시스템 준비 완료!")

    def ask(self, question):
        try:
            # 에이전트 생성 (파싱 에러 자동 핸들링 추가)
            agent = create_sql_agent(
                llm=self.llm,
                db=self.db,
                agent_type="zero-shot-react-description",
                verbose=True,
                # 파싱 에러 발생 시 출력 형식을 다시 강조하며 재시도 유도
                handle_parsing_errors="Check your format. If you found the answer, output 'Final Answer: [result]' only."
            )
            
            print(f"\n🔍 질문: {question}")
            
            # 모델이 형식을 엄격히 지키도록 유도하는 프롬프트
            prompt = (
                f"You are a SQL expert. Follow this format strictly:\n"
                f"Thought: I need to find the count of users.\n"
                f"Action: sql_db_query\n"
                f"Action Input: SELECT COUNT(*) FROM user\n"
                f"Observation: 978\n"
                f"Final Answer: 사용자 수는 총 978명입니다.\n\n"
                f"Question: {question}"
            )
            
            result = agent.invoke({"input": prompt})
            print(f"\n💡 결과: {result.get('output')}")
            
        except Exception as e:
            print(f"❌ 실행 에러 발생: {e}")

if __name__ == "__main__":
    # 실제 모델 경로 확인 필수
    MODEL_PATH = "/home/dongsucat1/ai/sql-generator/models/sql-generator-spider-plus-company"
    bot = LangChainAgentBot(MODEL_PATH)
    bot.ask("사용자 테이블에 등록된 전체 사용자 수는 몇 명인가요?")
