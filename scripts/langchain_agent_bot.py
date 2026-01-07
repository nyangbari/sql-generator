#!/usr/bin/env python3
# langchain_agent_bot.py
# LangChain Agent with Correct Imports

import os
import sys
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from langchain_huggingface import HuggingFacePipeline
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent  # 올바른 import!
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import inspect

load_dotenv()

class FixedSQLDatabase(SQLDatabase):
    """Read-Only SQL Database with No Caching"""
    
    WRITE_KEYWORDS = [
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 
        'ALTER', 'TRUNCATE', 'REPLACE', 'MERGE'
    ]
    
    def __init__(self, *args, **kwargs):
        kwargs['sample_rows_in_table_info'] = 0
        super().__init__(*args, **kwargs)
        self._sample_rows_in_table_info = 0
    
    def get_table_info_no_throw(self, table_names=None):
        """캐시 없이 항상 새로 조회"""
        
        if table_names is None:
            table_names = self.get_usable_table_names()
        
        inspector = inspect(self._engine)
        
        all_info = []
        
        for table in table_names:
            try:
                columns = inspector.get_columns(table)
                pk = inspector.get_pk_constraint(table)
                pk_cols = pk.get('constrained_columns', [])
                
                create = f"CREATE TABLE {table} (\n"
                
                col_defs = []
                for col in columns:
                    col_def = f"  {col['name']} {col['type']}"
                    
                    if not col['nullable']:
                        col_def += " NOT NULL"
                    
                    if col['name'] in pk_cols:
                        col_def += " PRIMARY KEY"
                    
                    col_defs.append(col_def)
                
                create += ",\n".join(col_defs)
                create += "\n)"
                
                all_info.append(create)
                
            except Exception as e:
                print(f"⚠️  {table} 스키마 조회 실패: {e}")
        
        result = "\n\n".join(all_info)
        
        print(f"\n📋 실제 스키마:\n{result}\n")
        
        return result
    
    def run(self, command: str, fetch: str = "all", **kwargs):
        """SQL 실행"""
        
        sql_upper = command.upper().strip()
        
        for keyword in self.WRITE_KEYWORDS:
            if keyword in sql_upper:
                raise ValueError(f"🚫 {keyword} 차단!")
        
        if not any(sql_upper.startswith(k) for k in ['SELECT', 'SHOW', 'DESCRIBE', 'EXPLAIN']):
            raise ValueError("🚫 SELECT만 허용")
        
        print(f"\n🔍 [실행 SQL]\n{command}\n")
        
        result = super().run(command, fetch=fetch, **kwargs)
        
        print(f"📊 [결과]\n{result}\n")
        
        return result

class LangChainAgentBot:
    def __init__(self, model_path):
        print("="*70)
        print("🤖 LangChain Agent SQL Bot")
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
            max_new_tokens=256,
            temperature=0.1,
            return_full_text=False
        )
        
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        # DB
        self.databases = {}
        for proj in ["KNIGHTFURY", "FURYX"]:
            uri = os.getenv(f"{proj}_DB_URI")
            if uri:
                self.databases[proj.lower()] = uri.replace("mysql://", "mysql+pymysql://")
        
        print("\n📚 프로젝트:", ', '.join(self.databases.keys()))
        print("="*70)
    
    def create_fresh_agent(self, project):
        """새 Agent 생성"""
        
        uri = self.databases.get(project.lower())
        if not uri:
            raise ValueError(f"프로젝트 '{project}' 없음")
        
        # 새 DB
        db = FixedSQLDatabase.from_uri(uri, sample_rows_in_table_info=0)
        
        print(f"\n🔗 [{project}] Agent 생성 중...")
        
        tables = db.get_usable_table_names()
        print(f"📊 테이블: {len(tables)}개")
        
        # fury_users 스키마 확인
        if 'fury_users' in tables:
            db.get_table_info_no_throw(['fury_users'])
        
        # Agent 생성
        agent = create_sql_agent(
            llm=self.llm,
            db=db,
            agent_type="zero-shot-react-description",
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
            max_execution_time=30
        )
        
        print("✅ Agent 준비!")
        
        return agent, db
    
    def ask(self, project, question):
        """질문 처리"""
        
        print("\n" + "="*70)
        print(f"📂 {project} | 💬 {question}")
        print("="*70)
        
        try:
            # 새 Agent
            agent, db = self.create_fresh_agent(project)
            
            # 스키마 명시
            tables = db.get_usable_table_names()
            main_tables = ['fury_users'] if 'fury_users' in tables else tables[:2]
            schema = db.get_table_info_no_throw(main_tables)
            
            prompt = f"""Answer this ONE question only.

DATABASE SCHEMA:
{schema}

RULES:
1. Use ONLY the columns from schema above
2. For COUNT: SELECT COUNT(*) FROM table_name
3. Execute and report ACTUAL result
4. After answering, STOP

Question: {question}

Answer:"""
            
            print("\n🤔 Agent 실행 중...\n")
            
            result = agent.invoke({"input": prompt})
            
            if isinstance(result, dict):
                answer = result.get('output', str(result))
            else:
                answer = str(result)
            
            print("\n" + "="*70)
            print(f"💡 답변:")
            print(answer)
            print("="*70)
            
            return answer
            
        except Exception as e:
            print(f"\n❌ 오류: {e}")
            print("="*70)
            return None
    
    def verify(self, project, table='fury_users'):
        """직접 검증"""
        
        uri = self.databases.get(project.lower())
        db = FixedSQLDatabase.from_uri(uri, sample_rows_in_table_info=0)
        
        print(f"\n🔍 [{table}] 직접 COUNT:")
        result = db.run(f"SELECT COUNT(*) FROM {table}")
        print(f"✅ {result}")
        
        return result

# 실행
if __name__ == "__main__":
    MODEL_PATH = "./models/sql-generator-spider-plus-company"
    
    bot = LangChainAgentBot(MODEL_PATH)
    
    # 직접 검증
    print("\n" + "="*70)
    print("🧪 직접 검증")
    print("="*70)
    bot.verify("knightfury", "fury_users")
    
    # Agent
    if len(sys.argv) > 2:
        bot.ask(sys.argv[1], sys.argv[2])
    else:
        bot.ask("knightfury", "How many users are in fury_users?")
