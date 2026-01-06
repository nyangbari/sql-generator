#!/usr/bin/env python3
import os
import torch
from typing import Any, List, Optional
from dotenv import load_dotenv

# 모듈 임포트 에러를 방지하는 최신 경로
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from langchain_huggingface import HuggingFacePipeline
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent

# AgentType 에러를 피하기 위해 문자열로 직접 지정하거나 아래 경로를 시도합니다.
try:
    from langchain.agents.agent_types import AgentType
except ImportError:
    try:
        from langchain.agents import AgentType
    except ImportError:
        # 두 곳 다 안될 경우 내부 문자열로 대체되도록 설정
        AgentType = None

load_dotenv()

# --- [보안 클래스] ---
class ReadOnlySQLDatabase(SQLDatabase):
    WRITE_KEYWORDS = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE', 'REPLACE', 'MERGE']
    def run(self, command: str, fetch: str = "all", **kwargs):
        sql_upper = command.upper().strip()
        for keyword in self.WRITE_KEYWORDS:
            if keyword in sql_upper: raise ValueError(f"🚫 {keyword} 차단!")
        if not any(sql_upper.startswith(k) for k in ['SELECT', 'SHOW', 'DESCRIBE']):
            raise ValueError("🚫 허용되지 않은 쿼리")
        print(f"✅ [실제 DB 조회]")
        return super().run(command, fetch=fetch, **kwargs)

class LangChainAgentBot:
    def __init__(self, model_path):
        print("🚀 모델 로딩 (8-bit 안정 모드)...")
        base_model_id = "codellama/CodeLlama-7b-Instruct-hf"
        
        # vLLM의 메모리 에러를 피하기 위해 transformers 8-bit 사용
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, torch_dtype=torch.float16, device_map="auto", load_in_8bit=True
        )
        
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer,
            max_new_tokens=256, temperature=0.1, top_p=0.9, return_full_text=False
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        # DB 설정
        self.databases = {}
        for proj in ["KNIGHTFURY", "FURYX"]:
            uri = os.getenv(f"{proj}_DB_URI")
            if uri: self.databases[proj.lower()] = uri.replace("mysql://", "mysql+pymysql://")
        
        self.agents = {}

    def get_agent(self, project):
        project = project.lower()
        if project not in self.agents:
            uri = self.databases.get(project)
            if not uri: raise ValueError(f"'{project}' DB 설정 없음")
            
            db = ReadOnlySQLDatabase.from_uri(uri)
            
            # AgentType 에러 방지를 위해 직접 문자열 "zero-shot-react-description" 사용
            self.agents[project] = create_sql_agent(
                llm=self.llm,
                db=db,
                agent_type="zero-shot-react-description", 
                verbose=True,
                handle_parsing_errors=True
            )
        return self.agents[project]

    def ask(self, project, question):
        print(f"\n📂 프로젝트: {project} | 질문: {question}")
        try:
            agent = self.get_agent(project)
            # 환각 방지: 테이블 목록을 먼저 보도록 강제
            prompt = f"1. sql_db_list_tables로 테이블 목록 확인\n2. 실제 있는 테이블만 쿼리\n질문: {question}"
            result = agent.invoke({"input": prompt})
            print(f"\n💡 결과: {result.get('output')}")
        except Exception as e:
            print(f"❌ 에러: {e}")

if __name__ == "__main__":
    MODEL_PATH = "/home/dongsucat1/ai/sql-generator/models/sql-generator-spider-plus-company"
    bot = LangChainAgentBot(MODEL_PATH)
    # 가짜 user 테이블 대신, 실제 테이블 목록을 불러오는지 테스트
    bot.ask("knightfury", "현재 DB에 어떤 테이블들이 있는지 이름만 다 알려줘.")
