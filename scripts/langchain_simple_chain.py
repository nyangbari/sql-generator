#!/usr/bin/env python3
# langchain_simple_chain.py
# 현업 스타일: Agent 없이 Simple Chain

import os
import sys
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from langchain_huggingface import HuggingFacePipeline
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from sqlalchemy import inspect

load_dotenv()

class SimpleSQLChain:
    """현업 스타일: Chain 기반 SQL Bot"""
    
    def __init__(self, model_path):
        print("="*70)
        print("🤖 LangChain Simple Chain (현업 스타일)")
        print("   - Agent 없음")
        print("   - 예측 가능한 흐름")
        print("   - 멈춤 문제 없음")
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
            max_new_tokens=150,
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
        
        print("\n📚 프로젝트:", ', '.join(self.databases.keys()))
        print("="*70)
    
    def get_schema(self, db, table_names):
        """스키마 가져오기"""
        inspector = inspect(db._engine)
        
        schema = ""
        for table in table_names:
            columns = inspector.get_columns(table)
            pk = inspector.get_pk_constraint(table)
            pk_cols = pk.get('constrained_columns', [])
            
            schema += f"\nTable: {table}\nColumns:\n"
            for col in columns:
                pk_mark = " (PK)" if col['name'] in pk_cols else ""
                schema += f"  - {col['name']}: {col['type']}{pk_mark}\n"
        
        return schema
    
    def ask(self, project, question):
        """질문 → SQL → 실행 → 답변 (Chain)"""
        
        print("\n" + "="*70)
        print(f"📂 {project} | 💬 {question}")
        print("="*70)
        
        uri = self.databases.get(project.lower())
        if not uri:
            print("❌ 프로젝트 없음")
            return None
        
        try:
            # DB 연결
            db = SQLDatabase.from_uri(uri, sample_rows_in_table_info=0)
            
            # 스키마
            tables = db.get_usable_table_names()
            main_tables = ['fury_users'] if 'fury_users' in tables else tables[:1]
            schema = self.get_schema(db, main_tables)
            
            print(f"\n📋 스키마:\n{schema}")
            
            # Step 1: SQL 생성 Chain
            sql_prompt = PromptTemplate(
                input_variables=["schema", "question"],
                template="""Given this database schema:

{schema}

Generate a SQL query to answer: {question}

Return ONLY the SQL query, nothing else.

SQL:"""
            )
            
            sql_chain = LLMChain(llm=self.llm, prompt=sql_prompt)
            
            print("\n🔄 Step 1: SQL 생성 중...")
            
            sql = sql_chain.run(schema=schema, question=question)
            
            # SQL 정리
            sql = sql.strip()
            sql = sql.replace('```sql', '').replace('```', '').strip()
            sql = sql.split('\n')[0] if '\n\n' in sql else sql
            
            # 보안 체크
            sql_upper = sql.upper()
            if any(kw in sql_upper for kw in ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER']):
                print("🚫 위험한 SQL 차단!")
                return None
            
            print(f"\n💾 생성된 SQL:\n{sql}")
            
            # Step 2: SQL 실행
            print("\n🔄 Step 2: 실행 중...")
            result = db.run(sql)
            
            print(f"\n📊 DB 결과:\n{result}")
            
            # Step 3: 답변 생성 Chain
            answer_prompt = PromptTemplate(
                input_variables=["question", "sql", "result"],
                template="""Question: {question}
SQL: {sql}
Result: {result}

Provide a natural language answer in one sentence.

Answer:"""
            )
            
            answer_chain = LLMChain(llm=self.llm, prompt=answer_prompt)
            
            print("\n🔄 Step 3: 답변 생성 중...")
            
            answer = answer_chain.run(question=question, sql=sql, result=result)
            answer = answer.strip().split('\n')[0]
            
            print("\n" + "="*70)
            print(f"💡 최종 답변:")
            print(answer)
            print("="*70)
            
            return answer
            
        except Exception as e:
            print(f"\n❌ 오류: {e}")
            print("="*70)
            import traceback
            traceback.print_exc()
            return None

# 실행
if __name__ == "__main__":
    MODEL_PATH = "./models/sql-generator-spider-plus-company"
    
    bot = SimpleSQLChain(MODEL_PATH)
    
    if len(sys.argv) > 2:
        bot.ask(sys.argv[1], sys.argv[2])
    else:
        # 테스트
        bot.ask("knightfury", "How many users are in fury_users?")
        print("\n")
        bot.ask("knightfury", "What networks exist in fury_users?")
