#!/usr/bin/env python3
import os
import torch
from typing import Any, List, Optional
from dotenv import load_dotenv
from pydantic import Field

# 모델 로딩 관련
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from langchain_huggingface import HuggingFacePipeline

# LangChain 관련
from langchain_core.language_models.llms import LLM
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent

load_dotenv()

# --- [중요] 보안 클래스 복구 ---
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
        
        # 1. 모델 로드
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True
        )
        
        # 2. 어댑터 결합
        print(f"📦 어댑터 결합 중...")
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        
        # 3. LangChain 파이프라인
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
        
        # 4. DB 연결 (보안 클래스인 ReadOnlySQLDatabase 사용)
        uri = os.getenv("KNIGHTFURY_DB_URI", "").replace("mysql://", "mysql+pymysql://")
        self.db = ReadOnlySQLDatabase.from_uri(uri)
        print("✅ 시스템 준비 완료!")

    def ask(self, question):
        try:
            # 에이전트 생성
            agent = create_sql_agent(
                llm=self.llm,
                db=self.db,
                agent_type="zero-shot-react-description",
                verbose=True,
                # 파싱 에러 발생 시 모델에게 다시 시도하도록 유도
                handle_parsing_errors="Check your output format. If you found the answer, use 'Final Answer:' only."
            )
            
            print(f"\n🔍 질문: {question}")
            
            # 파싱 에러를 줄이기 위해 형식을 아주 명확하게 지시하는 프롬프트
            prompt = (
                f"You are a SQL expert. Follow this format strictly:\n"
                f"Thought: I need to find the total number of users.\n"
                f"Action: sql_db_query\n"
                f"Action Input: SELECT COUNT(*) FROM user\n"
                f"Observation: (result from tool)\n"
                f"Final Answer: (The result in Korean)\n\n"
                f"Question: {question}"
            )
            
            result = agent.invoke({"input": prompt})
            print(f"\n💡 결과: {result.get('output')}")
            
        except Exception as e:
            print(f"❌ 에러 발생: {e}")

if __name__ == "__main__":
    MODEL_PATH = "/home/dongsucat1/ai/sql-generator/models/sql-generator-spider-plus-company"
    bot = LangChainAgentBot(MODEL_PATH)
    bot.ask("사용자 테이블에 등록된 전체 사용자 수는 몇 명인가요?")
