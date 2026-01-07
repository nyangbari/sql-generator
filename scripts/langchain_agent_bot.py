#!/usr/bin/env python3
# langchain_agent_bot.py
# 샘플 데이터 로딩 비활성화

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
    """샘플 데이터 없이 스키마만 제공하는 DB"""
    
    WRITE_KEYWORDS = [
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 
        'ALTER', 'TRUNCATE', 'REPLACE', 'MERGE'
    ]
    
    def __init__(self, *args, sample_rows_in_table_info=0, **kwargs):
        """샘플 데이터 0개로 초기화"""
        super().__init__(
            *args, 
            sample_rows_in_table_info=0,  # 샘플 데이터 비활성화!
            **kwargs
        )
    
    def run(self, command: str, fetch: str = "all", **kwargs):
        """SQL 실행"""
        
        sql_upper = command.upper().strip()
        
        # 보안 체크
        for keyword in self.WRITE_KEYWORDS:
            if keyword in sql_upper:
                raise ValueError(f"🚫 {keyword} 차단!")
        
        if not any(sql_upper.startswith(k) for k in ['SELECT', 'SHOW', 'DESCRIBE']):
            raise ValueError("🚫 SELECT/SHOW/DESCRIBE만 허용")
        
        # 실행 로그
        print("\n" + "🔍"*35)
        print("🔍 [실행 SQL]")
        print("🔍"*35)
        print(command)
        print("🔍"*35)
        
        # 실행
        result = super().run(command, fetch=fetch, **kwargs)
        
        # 결과 로그
        print("\n" + "📊"*35)
        print("📊 [DB 결과]")
        print("📊"*35)
        print(f"Type: {type(result)}")
        print(f"Result: {result}")
        print("📊"*35 + "\n")
        
        return result

class LangChainAgentBot:
    def __init__(self, model_path):
        """LangChain Agent Bot - No Sample Data"""
        
        print("="*70)
        print("🤖 LangChain Agent SQL Bot")
        print("   - 샘플 데이터 로딩 비활성화")
        print("   - 정확한 COUNT 보장")
        print("="*70)
        
        # 모델 로드
        print("\n🔄 모델 로딩 중...")
        
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
        
        print("✅ Spider + Company 학습 모델 로드 완료!")
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            return_full_text=False
        )
        
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        # DB 설정
        self.databases = {}
        for proj in ["KNIGHTFURY", "FURYX"]:
            uri = os.getenv(f"{proj}_DB_URI")
            if uri:
                self.databases[proj.lower()] = uri.replace("mysql://", "mysql+pymysql://")
        
        print("\n📚 프로젝트:")
        for project in self.databases.keys():
            print(f"  ✅ {project}")
        
        self.agents = {}
        self.db_connections = {}
        
        print("\n✅ 준비 완료!")
        print("="*70)
    
    def get_db(self, project):
        """프로젝트별 DB 연결 (샘플 데이터 없음)"""
        project = project.lower()
        
        if project not in self.db_connections:
            uri = self.databases.get(project)
            if not uri:
                raise ValueError(f"프로젝트 '{project}' 없음")
            
            # 샘플 데이터 0개로 DB 생성
            self.db_connections[project] = NoSampleSQLDatabase.from_uri(
                uri,
                sample_rows_in_table_info=0  # 중요!
            )
        
        return self.db_connections[project]
    
    def get_agent(self, project):
        """프로젝트별 Agent 생성"""
        
        project = project.lower()
        
        if project not in self.agents:
            db = self.get_db(project)
            
            print(f"\n🔗 [{project}] Agent 생성 중...")
            
            tables = db.get_usable_table_names()
            print(f"📊 테이블: {len(tables)}개")
            
            self.agents[project] = create_sql_agent(
                llm=self.llm,
                db=db,
                agent_type="zero-shot-react-description",
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5,
                max_execution_time=60
            )
            
            print("✅ Agent 생성 완료!")
        
        return self.agents[project]
    
    def ask(self, project, question):
        """질문 처리"""
        
        print("\n" + "="*70)
        print(f"📂 프로젝트: {project}")
        print(f"💬 질문: {question}")
        print("="*70)
        
        try:
            db = self.get_db(project)
            tables = db.get_usable_table_names()
            
            # fury_users 우선
            main_tables = ['fury_users'] if 'fury_users' in tables else tables[:3]
            schema_info = db.get_table_info(main_tables)
            
            print(f"\n📋 스키마 (샘플 데이터 없음):")
            print(schema_info[:500] + "..." if len(schema_info) > 500 else schema_info)
            
            # 명확한 프롬프트
            enhanced_prompt = f"""You are a SQL expert.

DATABASE SCHEMA (NO sample data included):
{schema_info}

RULES:
1. Use ONLY columns from schema
2. For COUNT: SELECT COUNT(*) FROM table (no LIMIT!)
3. Execute query and use ACTUAL result
4. Never use sample data for answers

Question: {question}

Answer with actual query result only.

Begin:"""
            
            agent = self.get_agent(project)
            
            print("\n🤔 Agent 실행 중...\n")
            
            result = agent.invoke({"input": enhanced_prompt})
            
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
            print("="*70)
            return None
    
    def list_tables(self, project):
        """테이블 목록"""
        try:
            db = self.get_db(project)
            tables = db.get_usable_table_names()
            
            print(f"\n📊 [{project}] 테이블 ({len(tables)}개):")
            for i, table in enumerate(tables, 1):
                print(f"  {i}. {table}")
        
        except Exception as e:
            print(f"❌ 오류: {e}")

# 실행
if __name__ == "__main__":
    MODEL_PATH = "./models/sql-generator-spider-plus-company"
    
    bot = LangChainAgentBot(MODEL_PATH)
    
    if len(sys.argv) > 2:
        bot.ask(sys.argv[1], sys.argv[2])
    else:
        bot.list_tables("knightfury")
        print("\n" + "="*70)
        print("🧪 테스트: fury_users 사용자 수")
        print("="*70)
        bot.ask("knightfury", "how many total users are in fury_users table?")
