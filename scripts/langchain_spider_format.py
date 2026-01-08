#!/usr/bin/env python3
# langchain_spider_format.py
# Spider 학습 형식 그대로 사용

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

class SpiderFormatBot:
    """Spider 형식 프롬프트 사용"""
    
    def __init__(self, model_path):
        print("="*70)
        print("🤖 Spider Format SQL Bot")
        print("   학습한 형식 그대로 사용!")
        print("="*70)
        
        print("\n🔄 로딩...")
        
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
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=150,
            temperature=0.1,
            return_full_text=False
        )
        
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        self.databases = {}
        for proj in ["KNIGHTFURY", "FURYX"]:
            uri = os.getenv(f"{proj}_DB_URI")
            if uri:
                self.databases[proj.lower()] = uri.replace("mysql://", "mysql+pymysql://")
        
        print("✅ 완료!")
        print(f"📚 프로젝트: {', '.join(self.databases.keys())}")
        print("="*70)
    
    def select_tables(self, question, all_tables):
        """관련 테이블 선택"""
        q = question.lower()
        
        keywords = {
            'user': ['user'],
            'mission': ['mission', 'quest'],
            'project': ['project', 'airdrop'],
            'game': ['game', 'play'],
            'config': ['config'],
        }
        
        selected = set()
        
        for category, patterns in keywords.items():
            if any(p in q for p in patterns):
                for table in all_tables:
                    if any(p in table.lower() for p in patterns):
                        selected.add(table)
        
        if not selected:
            selected = {'fury_users'}
        
        return list(selected)[:2]
    
    def get_spider_schema(self, db, tables):
        """Spider 형식 스키마 (학습한 형식!)"""
        
        inspector = inspect(db._engine)
        schema = ""
        
        for table in tables:
            try:
                columns = inspector.get_columns(table)
                pk = inspector.get_pk_constraint(table)
                pk_cols = pk.get('constrained_columns', [])
                
                # Spider 형식 그대로!
                schema += f"CREATE TABLE {table} (\n"
                
                col_defs = []
                for col in columns:
                    col_type = str(col['type'])
                    
                    # INT, VARCHAR만 (Spider 스타일)
                    if 'INT' in col_type.upper():
                        col_type = "INT"
                    elif 'VARCHAR' in col_type.upper() or 'CHAR' in col_type.upper():
                        col_type = "VARCHAR(100)"
                    elif 'TEXT' in col_type.upper():
                        col_type = "TEXT"
                    elif 'DATE' in col_type.upper():
                        col_type = "DATETIME"
                    
                    pk_marker = " PRIMARY KEY" if col['name'] in pk_cols else ""
                    
                    col_defs.append(f"    {col['name']} {col_type}{pk_marker}")
                
                schema += ",\n".join(col_defs)
                schema += "\n)\n\n"
                
            except Exception as e:
                print(f"⚠️  {table}: {e}")
        
        return schema
    
    def ask(self, project, question):
        
        print("\n" + "="*70)
        print(f"📂 {project}")
        print(f"💬 {question}")
        print("="*70)
        
        uri = self.databases.get(project.lower())
        if not uri:
            print("❌ 없음")
            return None
        
        try:
            db = SQLDatabase.from_uri(uri, sample_rows_in_table_info=0)
            
            all_tables = db.get_usable_table_names()
            tables = self.select_tables(question, all_tables)
            
            print(f"\n🎯 테이블: {tables}")
            
            # Spider 형식 스키마!
            schema = self.get_spider_schema(db, tables)
            
            print(f"\n📋 Spider 형식 스키마:")
            print(schema)
            
            # Spider 형식 프롬프트! (테스트에서 완벽하게 작동한 형식)
            print("🔄 SQL 생성 (Spider 형식)...")
            
            sql_prompt = PromptTemplate.from_template(
                """# Given the database schema:
{schema}

# Question: {question}

# SQL:
"""
            )
            
            sql_chain = sql_prompt | self.llm | StrOutputParser()
            
            sql = sql_chain.invoke({
                "schema": schema,
                "question": question
            })
            
            # 정리
            sql = sql.strip()
            
            # "# SQL:" 제거
            if "# SQL:" in sql:
                sql = sql.split("# SQL:")[-1].strip()
            
            # 첫 줄만
            sql = sql.split('\n')[0].strip()
            
            # 백틱 제거
            sql = sql.replace('```sql', '').replace('```', '').strip()
            
            # 세미콜론 제거
            if ';' in sql:
                sql = sql.split(';')[0].strip()
            
            print(f"\n💾 생성된 SQL:")
            print(sql)
            
            # 보안
            dangerous = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER']
            if any(kw in sql.upper() for kw in dangerous):
                print("🚫 차단")
                return None
            
            # 기본 검증만
            if not sql.upper().startswith('SELECT'):
                print("⚠️  SELECT로 시작하지 않음")
                sql = f"SELECT COUNT(*) FROM {tables[0]}"
                print(f"   → 기본 쿼리 사용: {sql}")
            
            # 실행
            print("\n🔄 실행...")
            
            result = db.run(sql)
            
            print(f"\n📊 결과:")
            print(result)
            
            # 간단한 답변
            if result and result != "[]":
                try:
                    if '[(' in str(result):
                        num = str(result).split('(')[1].split(',')[0]
                        answer = f"{num}개"
                    else:
                        answer = str(result)
                except:
                    answer = str(result)
            else:
                answer = "결과 없음"
            
            print("\n" + "="*70)
            print(f"💡 {answer}")
            print("="*70)
            
            return {"sql": sql, "result": result, "answer": answer}
            
        except Exception as e:
            print(f"\n❌ {e}")
            import traceback
            traceback.print_exc()
            return None

# 실행
if __name__ == "__main__":
    MODEL_PATH = "./models/sql-generator-spider-plus-company"
    
    bot = SpiderFormatBot(MODEL_PATH)
    
    if len(sys.argv) > 2:
        bot.ask(sys.argv[1], sys.argv[2])
    else:
        projects = list(bot.databases.keys())
        print(f"\n프로젝트: {', '.join(projects)}")
        project = input("선택: ").strip().lower()
        
        if project not in projects:
            sys.exit(1)
        
        print(f"✅ '{project}'\n")
        
        while True:
            try:
                q = input(f"[{project}] ").strip()
                
                if q.lower() in ['exit', 'quit', 'q']:
                    break
                
                if q:
                    bot.ask(project, q)
                    
            except KeyboardInterrupt:
                print("\n")
                break
