#!/usr/bin/env python3
# langchain_agent_bot.py
# Single Question Only - 강제 종료

import os
import sys
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from langchain_huggingface import HuggingFacePipeline
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent

load_dotenv()

class NoSampleSQLDatabase(SQLDatabase):
    """샘플 데이터 없는 Read-Only DB"""
    
    WRITE_KEYWORDS = [
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 
        'ALTER', 'TRUNCATE', 'REPLACE', 'MERGE'
    ]
    
    def __init__(self, *args, sample_rows_in_table_info=0, **kwargs):
        super().__init__(*args, sample_rows_in_table_info=0, **kwargs)
    
    def run(self, command: str, fetch: str = "all", **kwargs):
        sql_upper = command.upper().strip()
        
        for keyword in self.WRITE_KEYWORDS:
            if keyword in sql_upper:
                raise ValueError(f"🚫 {keyword} 차단!")
        
        if not any(sql_upper.startswith(k) for k in ['SELECT', 'SHOW', 'DESCRIBE']):
            raise ValueError("🚫 SELECT만 허용")
        
        print(f"\n🔍 SQL: {command}")
        
        result = super().run(command, fetch=fetch, **kwargs)
        
        print(f"📊 결과: {result}\n")
        
        return result

class LangChainAgentBot:
    def __init__(self, model_path):
        print("="*70)
        print("🤖 LangChain SQL Bot - Single Question Mode")
        print("="*70)
        
        print("\n🔄 모델 로딩...")
        
        base_model_id = "codellama/CodeLlama-7b-Instruct-hf"
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True
        )
        
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        
        print("✅ 모델 로드 완료!")
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,  # 512 → 256 줄임
            temperature=0.1,
            return_full_text=False
        )
        
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        self.databases = {}
        for proj in ["KNIGHTFURY", "FURYX"]:
            uri = os.getenv(f"{proj}_DB_URI")
            if uri:
                self.databases[proj.lower()] = uri.replace("mysql://", "mysql+pymysql://")
        
        print("\n📚 프로젝트:", ', '.join(self.databases.keys()))
        print("="*70)
        
        self.agents = {}
        self.db_connections = {}
    
    def get_db(self, project):
        project = project.lower()
        
        if project not in self.db_connections:
            uri = self.databases.get(project)
            if not uri:
                raise ValueError(f"프로젝트 '{project}' 없음")
            
            self.db_connections[project] = NoSampleSQLDatabase.from_uri(
                uri, sample_rows_in_table_info=0
            )
        
        return self.db_connections[project]
    
    def get_agent(self, project):
        project = project.lower()
        
        if project not in self.agents:
            db = self.get_db(project)
            
            self.agents[project] = create_sql_agent(
                llm=self.llm,
                db=db,
                agent_type="zero-shot-react-description",
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=3,  # 5 → 3으로 줄임
                max_execution_time=30,  # 60 → 30초
                early_stopping_method="generate"
            )
        
        return self.agents[project]
    
    def ask(self, project, question):
        """단일 질문 처리"""
        
        print("\n" + "="*70)
        print(f"📂 {project} | 💬 {question}")
        print("="*70)
        
        try:
            db = self.get_db(project)
            
            # 간단한 프롬프트
            prompt = f"""Answer this question ONLY. Do NOT ask follow-up questions.

Question: {question}

Instructions:
1. Check the schema if needed
2. Write ONE SQL query
3. Execute it
4. Return the answer
5. STOP (do not continue)

Answer:"""
            
            agent = self.get_agent(project)
            
            result = agent.invoke({"input": prompt})
            
            if isinstance(result, dict):
                answer = result.get('output', str(result))
            else:
                answer = str(result)
            
            print("\n" + "="*70)
            print(f"💡 {answer}")
            print("="*70)
            
            return answer
            
        except Exception as e:
            print(f"\n❌ {e}")
            return None

# 실행
if __name__ == "__main__":
    MODEL_PATH = "./models/sql-generator-spider-plus-company"
    
    bot = LangChainAgentBot(MODEL_PATH)
    
    if len(sys.argv) > 2:
        bot.ask(sys.argv[1], sys.argv[2])
    else:
        # 단일 테스트
        bot.ask("knightfury", "How many users are in fury_users?")
