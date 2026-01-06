#!/usr/bin/env python3
import os
import torch
import sys
from typing import Any, List, Optional
from dotenv import load_dotenv

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from langchain_huggingface import HuggingFacePipeline
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent

load_dotenv()

class ReadOnlySQLDatabase(SQLDatabase):
    WRITE_KEYWORDS = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE', 'REPLACE', 'MERGE']
    def run(self, command: str, fetch: str = "all", **kwargs):
        sql_upper = command.upper().strip()
        for keyword in self.WRITE_KEYWORDS:
            if keyword in sql_upper: raise ValueError(f"🚫 {keyword} 차단!")
        if not any(sql_upper.startswith(k) for k in ['SELECT', 'SHOW', 'DESCRIBE']):
            raise ValueError("🚫 SELECT/SHOW/DESCRIBE만 가능")
        print(f"✅ [실제 DB 조회]")
        return super().run(command, fetch=fetch, **kwargs)

class LangChainAgentBot:
    def __init__(self, model_path):
        print("🚀 모델 로딩 중...")
        base_model_id = "codellama/CodeLlama-7b-Instruct-hf"
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, torch_dtype=torch.float16, device_map="auto", load_in_8bit=True
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer,
            max_new_tokens=512, temperature=0.1, return_full_text=False
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        self.databases = {}
        for proj in ["KNIGHTFURY", "FURYX"]:
            uri = os.getenv(f"{proj}_DB_URI")
            if uri: self.databases[proj.lower()] = uri.replace("mysql://", "mysql+pymysql://")
        self.agents = {}

    def get_agent(self, project):
        project = project.lower()
        if project not in self.agents:
            db = ReadOnlySQLDatabase.from_uri(self.databases[project])
            self.agents[project] = create_sql_agent(
                llm=self.llm, db=db, agent_type="zero-shot-react-description",
                verbose=True, handle_parsing_errors=True
            )
        return self.agents[project]

    def ask(self, project, question):
        print(f"\n📂 프로젝트: {project} | 질문: {question}")
        try:
            agent = self.get_agent(project)
            # 중요: 추론 과정은 영어로 하되 최종 답변만 한국어로 하라고 명령
            prompt = (
                f"You are a SQL expert. Use the following steps:\n"
                f"1. Use `sql_db_list_tables` to see all tables.\n"
                f"2. Use `sql_db_schema` to check the table structure.\n"
                f"3. Write a SQL query and execute it.\n"
                f"4. Provide the final answer in Korean.\n\n"
                f"Question: {question}"
            )
            result = agent.invoke({"input": prompt})
            print(f"\n💡 최종 결과: {result.get('output')}")
        except Exception as e:
            print(f"❌ 에러: {e}")

if __name__ == "__main__":
    MODEL_PATH = "/home/dongsucat1/ai/sql-generator/models/sql-generator-spider-plus-company"
    bot = LangChainAgentBot(MODEL_PATH)
    
    if len(sys.argv) > 2:
        bot.ask(sys.argv[1], sys.argv[2])
    else:
        # 기본 테스트 질문
        bot.ask("knightfury", "fury_users 테이블의 전체 사용자 수는?")
