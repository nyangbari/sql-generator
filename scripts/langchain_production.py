#!/usr/bin/env python3
# langchain_production.py
# 실전용 LangChain SQL Bot (수정 버전)

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

class ProductionSQLBot:
    """실전용 SQL Bot"""
    
    def __init__(self, model_path):
        print("="*70)
        print("🤖 Production SQL Bot")
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
            max_new_tokens=200,
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
    
    def select_tables(self, question, all_tables):
        """관련 테이블 선택"""
        question_lower = question.lower()
        
        keywords = {
            'user': ['user'],
            'mission': ['mission', 'quest'],
            'project': ['project', 'airdrop'],
            'game': ['game', 'play'],
            'telegram': ['telegram'],
            'discord': ['discord'],
            'twitter': ['twitter'],
        }
        
        selected = set()
        
        for category, patterns in keywords.items():
            if any(p in question_lower for p in patterns):
                for table in all_tables:
                    if any(p in table.lower() for p in patterns):
                        selected.add(table)
        
        if not selected:
            selected = {'fury_users'}
        
        return list(selected)[:3]
    
    def get_schema(self, db, table_names):
        """CREATE TABLE 스타일 스키마"""
        inspector = inspect(db._engine)
        schema = ""
        
        for table in table_names:
            try:
                columns = inspector.get_columns(table)
                pk = inspector.get_pk_constraint(table)
                pk_cols = pk.get('constrained_columns', [])
                
                schema += f"CREATE TABLE {table} (\n"
                
                col_defs = []
                for col in columns:
                    col_type = str(col['type'])
                    nullable = "" if col['nullable'] else " NOT NULL"
                    is_pk = " PRIMARY KEY" if col['name'] in pk_cols else ""
                    col_defs.append(f"  {col['name']} {col_type}{nullable}{is_pk}")
                
                schema += ",\n".join(col_defs)
                schema += "\n)\n\n"
            except Exception as e:
                print(f"⚠️  {table}: {e}")
        
        return schema
    
    def ask(self, project, question):
        """질문 처리"""
        
        print("\n" + "="*70)
        print(f"📂 {project}")
        print(f"💬 {question}")
        print("="*70)
        
        uri = self.databases.get(project.lower())
        if not uri:
            print("❌ 프로젝트 없음")
            return None
        
        try:
            db = SQLDatabase.from_uri(uri, sample_rows_in_table_info=0)
            
            all_tables = db.get_usable_table_names()
            relevant_tables = self.select_tables(question, all_tables)
            
            print(f"\n🎯 선택: {relevant_tables}")
            
            schema = self.get_schema(db, relevant_tables)
            
            print(f"\n📋 스키마:")
            print(schema[:300] + "...\n" if len(schema) > 300 else schema)
            
            # SQL 생성 (완전한 SQL 생성)
            print("🔄 Step 1: SQL 생성...")
            
            sql_prompt = PromptTemplate.from_template(
                """{schema}
-- Question: {question}
-- Generate complete SQL query

"""
            )
            
            sql_chain = sql_prompt | self.llm | StrOutputParser()
            
            sql = sql_chain.invoke({
                "schema": schema,
                "question": question
            })
            
            # SQL 정리
            sql = sql.strip()
            
            # SQL: 로 시작하면 제거
            if sql.startswith("SQL:"):
                sql = sql[4:].strip()
            
            # 첫 줄만
            sql = sql.split('\n')[0].strip()
            
            # 백틱 제거
            sql = sql.replace('```sql', '').replace('```', '').strip()
            
            # 세미콜론 제거
            if ';' in sql:
                sql = sql.split(';')[0]
            
            print(f"\n💾 생성된 SQL:")
            print(sql)
            
            # SELECT 없으면 추가
            if not sql.upper().startswith('SELECT'):
                if 'count' in question.lower() or 'how many' in question.lower():
                    sql = f"SELECT COUNT(*) FROM {relevant_tables[0]}"
                else:
                    sql = f"SELECT * FROM {relevant_tables[0]} LIMIT 10"
                print(f"   → 수정: {sql}")
            
            # FROM 없으면 추가
            if 'FROM' not in sql.upper():
                # COUNT(*) 같은 경우
                if sql.upper().startswith('SELECT'):
                    sql = sql + f" FROM {relevant_tables[0]}"
                    print(f"   → FROM 추가: {sql}")
            
            # 보안 체크
            dangerous = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER']
            if any(kw in sql.upper() for kw in dangerous):
                print("🚫 위험한 SQL")
                return None
            
            # 실행
            print("\n🔄 Step 2: 실행...")
            
            result = db.run(sql)
            
            print(f"\n📊 결과:")
            print(result)
            
            # 답변
            print("\n🔄 Step 3: 답변...")
            
            if not result or result == "[]":
                answer = "결과 없음"
            else:
                answer_prompt = PromptTemplate.from_template(
                    """Question: {question}
Result: {result}

Answer in Korean (1 sentence):"""
                )
                
                answer_chain = answer_prompt | self.llm | StrOutputParser()
                
                answer = answer_chain.invoke({
                    "question": question,
                    "result": result
                })
                
                answer = answer.strip().split('\n')[0]
            
            print("\n" + "="*70)
            print(f"💡 답변:")
            print(answer)
            print("="*70)
            
            return {"sql": sql, "result": result, "answer": answer}
            
        except Exception as e:
            print(f"\n❌ 오류: {e}")
            print("="*70)
            return None

# 실행
if __name__ == "__main__":
    MODEL_PATH = "./models/sql-generator-spider-plus-company"
    
    bot = ProductionSQLBot(MODEL_PATH)
    
    if len(sys.argv) > 2:
        bot.ask(sys.argv[1], sys.argv[2])
    else:
        # Interactive
        projects = list(bot.databases.keys())
        print(f"\n프로젝트: {', '.join(projects)}")
        project = input("선택: ").strip().lower()
        
        if project not in projects:
            print(f"❌ '{project}' 없음")
            sys.exit(1)
        
        print(f"\n✅ '{project}' 선택")
        print("💬 질문 입력 (종료: exit)\n")
        
        while True:
            try:
                question = input(f"\n[{project}] ").strip()
                
                if question.lower() in ['exit', 'quit']:
                    break
                
                if question:
                    bot.ask(project, question)
                    
            except KeyboardInterrupt:
                print("\n\n종료")
                break
