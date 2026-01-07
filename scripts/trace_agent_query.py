#!/usr/bin/env python3
# trace_agent_query.py
# Agent의 SQL 실행 과정 완전 추적

import os
import sys
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from langchain_huggingface import HuggingFacePipeline
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from sqlalchemy import create_engine, event, text

load_dotenv()

# SQL 실행 추적을 위한 클래스
class TrackedSQLDatabase(SQLDatabase):
    """모든 SQL 실행을 추적"""
    
    def __init__(self, *args, **kwargs):
        kwargs['sample_rows_in_table_info'] = 0
        super().__init__(*args, **kwargs)
        
        # SQLAlchemy 이벤트 리스너로 모든 쿼리 추적
        @event.listens_for(self._engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, params, context, executemany):
            print("\n" + "🔥"*35)
            print("🔥 [SQLAlchemy] 실제 실행되는 SQL:")
            print("🔥"*35)
            print(statement)
            if params:
                print(f"Parameters: {params}")
            print("🔥"*35 + "\n")
        
        @event.listens_for(self._engine, "after_cursor_execute")
        def receive_after_cursor_execute(conn, cursor, statement, params, context, executemany):
            print("\n" + "✅"*35)
            print("✅ [SQLAlchemy] 실행 완료")
            print(f"✅ Rowcount: {cursor.rowcount}")
            print("✅"*35 + "\n")
    
    def run(self, command: str, fetch: str = "all", **kwargs):
        print("\n" + "🎯"*35)
        print("🎯 [LangChain] run() 메서드 호출됨")
        print("🎯"*35)
        print(f"Command: {command}")
        print(f"Fetch: {fetch}")
        print("🎯"*35 + "\n")
        
        # 실제 실행
        result = super().run(command, fetch=fetch, **kwargs)
        
        print("\n" + "📊"*35)
        print("📊 [LangChain] run() 반환 값:")
        print("📊"*35)
        print(f"Type: {type(result)}")
        print(f"Value: {result}")
        print("📊"*35 + "\n")
        
        return result

class TraceAgentBot:
    def __init__(self, model_path):
        print("="*70)
        print("🔍 Agent SQL 추적 모드")
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
        
        print("✅ 모델 로드!")
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
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
    
    def ask(self, project, question):
        print("\n" + "="*70)
        print(f"📂 {project} | 💬 {question}")
        print("="*70)
        
        uri = self.databases.get(project.lower())
        if not uri:
            print("❌ 프로젝트 없음")
            return None
        
        try:
            # Tracked DB
            db = TrackedSQLDatabase.from_uri(uri)
            
            print("\n🔗 Agent 생성 중...")
            
            agent = create_sql_agent(
                llm=self.llm,
                db=db,
                agent_type="zero-shot-react-description",
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=3,
                max_execution_time=30
            )
            
            print("\n🚀 Agent 실행 시작...\n")
            
            result = agent.invoke({"input": question})
            
            if isinstance(result, dict):
                answer = result.get('output', str(result))
            else:
                answer = str(result)
            
            print("\n" + "="*70)
            print(f"💡 최종 답변:")
            print(answer)
            print("="*70)
            
            return answer
            
        except Exception as e:
            print(f"\n❌ 오류: {e}")
            import traceback
            traceback.print_exc()
            return None

# 실행
if __name__ == "__main__":
    MODEL_PATH = "./models/sql-generator-spider-plus-company"
    
    bot = TraceAgentBot(MODEL_PATH)
    
    if len(sys.argv) > 2:
        bot.ask(sys.argv[1], sys.argv[2])
    else:
        bot.ask("knightfury", "얼마나 많은 사용자가 텔레그램 ID를 가지고 있어?")
