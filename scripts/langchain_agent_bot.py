#!/usr/bin/env python3
# langchain_agent_bot.py
# 스키마 캐시 강제 새로고침

import os
import sys
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from langchain_huggingface import HuggingFacePipeline
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from sqlalchemy import create_engine, inspect, text

load_dotenv()

class FreshSchemaSQLDatabase(SQLDatabase):
    """항상 최신 스키마를 가져오는 DB"""
    
    WRITE_KEYWORDS = [
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 
        'ALTER', 'TRUNCATE', 'REPLACE', 'MERGE'
    ]
    
    def __init__(self, *args, **kwargs):
        # 샘플 데이터 0개
        kwargs['sample_rows_in_table_info'] = 0
        super().__init__(*args, **kwargs)
        
        # 캐시 무효화
        self._sample_rows_in_table_info = 0
        self._indexes_in_table_info = False
    
    def get_table_info(self, table_names=None):
        """실제 DB에서 직접 스키마 가져오기"""
        
        if table_names is None:
            table_names = self.get_usable_table_names()
        
        # Inspector로 실제 스키마 확인
        inspector = inspect(self._engine)
        
        all_table_info = []
        
        for table_name in table_names:
            # 실제 컬럼 정보
            columns = inspector.get_columns(table_name)
            pk_constraint = inspector.get_pk_constraint(table_name)
            pk_columns = pk_constraint.get('constrained_columns', [])
            
            # CREATE TABLE 문 생성
            create_table = f"\nCREATE TABLE {table_name} (\n"
            
            col_lines = []
            for col in columns:
                col_type = str(col['type'])
                nullable = "" if col['nullable'] else " NOT NULL"
                pk = " PRIMARY KEY" if col['name'] in pk_columns else ""
                
                col_lines.append(
                    f"    {col['name']} {col_type}{nullable}{pk}"
                )
            
            create_table += ",\n".join(col_lines)
            create_table += "\n)"
            
            all_table_info.append(create_table)
            
            print(f"\n📋 [{table_name}] 실제 스키마 확인:")
            print(create_table)
        
        return "\n\n".join(all_table_info)
    
    def run(self, command: str, fetch: str = "all", **kwargs):
        """SQL 실행"""
        
        sql_upper = command.upper().strip()
        
        # 보안 체크
        for keyword in self.WRITE_KEYWORDS:
            if keyword in sql_upper:
                raise ValueError(f"🚫 {keyword} 차단!")
        
        if not any(sql_upper.startswith(k) for k in ['SELECT', 'SHOW', 'DESCRIBE', 'EXPLAIN']):
            raise ValueError("🚫 SELECT만 허용")
        
        print(f"\n🔍 [실행 SQL]\n{command}\n")
        
        # 실행
        result = super().run(command, fetch=fetch, **kwargs)
        
        print(f"📊 [DB 결과]\n{result}\n")
        
        return result

class LangChainAgentBot:
    def __init__(self, model_path):
        print("="*70)
        print("🤖 LangChain SQL Bot - Fresh Schema")
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
        """프로젝트별 DB (캐시 안 함 - 항상 새로 생성)"""
        project = project.lower()
        
        uri = self.databases.get(project)
        if not uri:
            raise ValueError(f"프로젝트 '{project}' 없음")
        
        # 매번 새로 생성 (캐시 안 함!)
        return FreshSchemaSQLDatabase.from_uri(uri)
    
    def get_agent(self, project):
        """Agent 생성 (캐시 안 함)"""
        project = project.lower()
        
        # 매번 새로 생성
        db = self.get_db(project)
        
        return create_sql_agent(
            llm=self.llm,
            db=db,
            agent_type="zero-shot-react-description",
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
            max_execution_time=30
        )
    
    def ask(self, project, question):
        """질문 처리"""
        
        print("\n" + "="*70)
        print(f"📂 {project} | 💬 {question}")
        print("="*70)
        
        try:
            # 최신 스키마 확인
            db = self.get_db(project)
            tables = db.get_usable_table_names()
            print(f"\n📊 테이블: {len(tables)}개")
            
            # fury_users 스키마 강제 출력
            if 'fury_users' in tables:
                schema = db.get_table_info(['fury_users'])
                print(f"\n{schema}\n")
            
            # Agent 실행
            agent = self.get_agent(project)
            
            prompt = f"""Answer ONLY this question. Do NOT continue with other questions.

Question: {question}

Steps:
1. Check schema
2. Write SQL
3. Execute
4. Answer
5. STOP

Answer:"""
            
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
    
    def verify_count(self, project, table):
        """직접 COUNT 확인"""
        
        print(f"\n🔍 [{table}] 직접 COUNT 확인:")
        
        db = self.get_db(project)
        
        sql = f"SELECT COUNT(*) FROM {table}"
        result = db.run(sql)
        
        print(f"✅ 결과: {result}")
        
        return result

# 실행
if __name__ == "__main__":
    MODEL_PATH = "./models/sql-generator-spider-plus-company"
    
    bot = LangChainAgentBot(MODEL_PATH)
    
    # 직접 COUNT 먼저 확인
    print("\n" + "="*70)
    print("🧪 직접 COUNT 테스트")
    print("="*70)
    bot.verify_count("knightfury", "fury_users")
    
    # Agent로 질문
    if len(sys.argv) > 2:
        bot.ask(sys.argv[1], sys.argv[2])
    else:
        bot.ask("knightfury", "How many users are in fury_users table?")
