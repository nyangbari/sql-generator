#!/usr/bin/env python3
# langchain_simple_chain_v2.py
# 모든 테이블 스키마 제공 버전

import os
import sys
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from langchain_huggingface import HuggingFacePipeline
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy import inspect

load_dotenv()

class SimpleSQLChain:
    def __init__(self, model_path):
        print("="*70)
        print("🤖 LangChain Simple Chain v2")
        print("   - 모든 테이블 스키마 제공")
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
            max_new_tokens=200,
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
    
    def get_schema(self, db, table_names):
        """스키마 (간단 버전)"""
        inspector = inspect(db._engine)
        
        schema = ""
        for table in table_names:
            columns = inspector.get_columns(table)
            
            schema += f"\nTable: {table}\n"
            schema += "Columns: "
            schema += ", ".join([col['name'] for col in columns])
            schema += "\n"
        
        return schema
    
    def ask(self, project, question):
        print("\n" + "="*70)
        print(f"📂 {project} | 💬 {question}")
        print("="*70)
        
        uri = self.databases.get(project.lower())
        if not uri:
            print("❌ 프로젝트 없음")
            return None
        
        try:
            db = SQLDatabase.from_uri(uri, sample_rows_in_table_info=0)
            
            # 모든 테이블 목록
            all_tables = db.get_usable_table_names()
            print(f"\n📊 전체 테이블: {len(all_tables)}개")
            
            # fury_ 로 시작하는 테이블만 (관련 테이블)
            relevant_tables = [t for t in all_tables if t.startswith('fury_')]
            
            if not relevant_tables:
                relevant_tables = all_tables[:10]  # 처음 10개
            
            print(f"📋 사용할 테이블: {len(relevant_tables)}개")
            print(f"   {', '.join(relevant_tables[:5])}...")
            
            # 스키마 (간단 버전 - 테이블명과 컬럼명만)
            schema = self.get_schema(db, relevant_tables)
            
            print(f"\n📄 스키마:\n{schema[:500]}...")
            
            # Step 1: SQL 생성
            print("\n🔄 Step 1: SQL 생성...")
            
            sql_prompt = PromptTemplate.from_template(
                """You have access to these tables:

{schema}

Generate a SQL query to answer: {question}

Choose the correct table based on the question.
Return ONLY the SQL query.

SQL:"""
            )
            
            sql_chain = sql_prompt | self.llm | StrOutputParser()
            
            sql = sql_chain.invoke({
                "schema": schema,
                "question": question
            })
            
            # SQL 정리
            sql = sql.strip()
            sql = sql.replace('```sql', '').replace('```', '').strip()
            sql = sql.split('\n')[0] if '\n\n' in sql else sql
            
            # 보안
            sql_upper = sql.upper()
            if any(kw in sql_upper for kw in ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER']):
                print("🚫 위험한 SQL!")
                return None
            
            print(f"\n💾 SQL:\n{sql}")
            
            # Step 2: 실행
            print("\n🔄 Step 2: 실행...")
            result = db.run(sql)
            
            print(f"\n📊 결과:\n{result}")
            
            # Step 3: 답변
            print("\n🔄 Step 3: 답변 생성...")
            
            answer_prompt = PromptTemplate.from_template(
                """Question: {question}
SQL: {sql}
Result: {result}

Provide a natural language answer in Korean.

Answer:"""
            )
            
            answer_chain = answer_prompt | self.llm | StrOutputParser()
            
            answer = answer_chain.invoke({
                "question": question,
                "sql": sql,
                "result": result
            })
            
            answer = answer.strip()
            
            print("\n" + "="*70)
            print(f"💡 답변:")
            print(answer)
            print("="*70)
            
            return answer
            
        except Exception as e:
            print(f"\n❌ {e}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == "__main__":
    MODEL_PATH = "./models/sql-generator-spider-plus-company"
    
    bot = SimpleSQLChain(MODEL_PATH)
    
    if len(sys.argv) > 2:
        bot.ask(sys.argv[1], sys.argv[2])
    else:
        # 테스트
        bot.ask("knightfury", "얼마나 많은 미션이 있어?")
