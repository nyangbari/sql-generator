#!/usr/bin/env python3
# langchain_spider_final.py
# 최종 완성 버전 - 스마트 테이블 선택

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

class FinalSQLBot:
    
    def __init__(self, model_path):
        print("="*70)
        print("🤖 Final SQL Bot - 완성!")
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
    
    def smart_table_selection(self, question, all_tables):
        """스마트 테이블 선택 (우선순위 기반)"""
        
        q = question.lower()
        
        # 명확한 매핑 (우선순위 순서!)
        table_map = {
            'mission': [
                'fury_mission_configs',  # 미션 정의 (1순위)
                'fury_project_missions',  # 프로젝트별 미션
                'fury_user_project_missions',  # 사용자-미션
		'fury_project_mission_quizzes',
		'fury_project_mission_quiz_choices'
            ],
            'project': [
                'fury_projects',  # 프로젝트
                'fury_project_teams'
            ],
            'user': [
                'fury_users'  # 사용자
            ],
            'game': [
                'fury_play_games',  # 게임
                'fury_play_users'  # 게임 사용자
            ],
            'config': [
                'fury_mission_configs',
                'fury_global_configs'
            ]
        }
        
        # 질문 키워드 확인
        for keyword, priority_tables in table_map.items():
            if keyword in q:
                # 우선순위 순서대로 존재하는 테이블 반환
                for table in priority_tables:
                    if table in all_tables:
                        print(f"   키워드 '{keyword}' → {table}")
                        return [table]
        
        # 기본값
        return ['fury_users']
    
    def get_spider_schema(self, db, tables):
        """Spider 형식 스키마"""
        
        inspector = inspect(db._engine)
        schema = ""
        
        for table in tables:
            try:
                columns = inspector.get_columns(table)
                pk = inspector.get_pk_constraint(table)
                pk_cols = pk.get('constrained_columns', [])
                
                schema += f"CREATE TABLE {table} (\n"
                
                col_defs = []
                for col in columns[:10]:  # 처음 10개만
                    col_type = str(col['type'])
                    
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
    
    def validate_sql(self, sql, question):
        """WHERE/조건 환각 체크"""
        
        sql_upper = sql.upper()
        
        if 'WHERE' in sql_upper:
            # 전체 COUNT 키워드
            total_keywords = ['total', 'all', 'how many', 'count all']
            is_total = any(k in question.lower() for k in total_keywords)
            
            # 조건 키워드
            condition_keywords = [
                'week', 'day', 'month', 'year',
                'id =', 'name =', 'address =',
                'where', 'which', 'specific',
                '0x', '=', '>', '<'
            ]
            has_condition = any(k in question.lower() for k in condition_keywords)
            
            # WHERE 있는데 조건 명시 안 됨
            if is_total and not has_condition:
                print("⚠️  WHERE 환각!")
                print(f"   원본: {sql}")
                sql = sql.split('WHERE')[0].strip()
                print(f"   수정: {sql}")
        
        return sql
    
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
            
            print(f"\n🎯 스마트 선택:")
            tables = self.smart_table_selection(question, all_tables)
            
            print(f"   선택: {tables}")
            
            schema = self.get_spider_schema(db, tables)
            
            print(f"\n📋 스키마:")
            print(schema[:250] + "...\n" if len(schema) > 250 else schema)
            
            # Spider 형식 프롬프트
            print("🔄 SQL 생성...")
            
            sql_prompt = PromptTemplate.from_template(
                """# Given the database schema:
{schema}

# Question: {question}

# Generate SQL query
# If the question asks for total count without specific conditions, do NOT add WHERE clause

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
            if "# SQL:" in sql:
                sql = sql.split("# SQL:")[-1].strip()
            
            sql = sql.split('\n')[0].strip()
            sql = sql.replace('```sql', '').replace('```', '').strip()
            
            if ';' in sql:
                sql = sql.split(';')[0].strip()
            
            print(f"\n💾 원본: {sql}")
            
            # 검증
            sql = self.validate_sql(sql, question)
            
            # 보안
            dangerous = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER']
            if any(kw in sql.upper() for kw in dangerous):
                print("🚫 차단")
                return None
            
            # SELECT 체크
            if not sql.upper().startswith('SELECT'):
                print("⚠️  SELECT 없음")
                sql = f"SELECT COUNT(*) FROM {tables[0]}"
                print(f"   → 기본: {sql}")
            
            print(f"\n✅ 최종: {sql}")
            
            # 실행
            print("\n🔄 실행...")
            
            result = db.run(sql)
            
            print(f"\n📊 결과: {result}")
            
            # 답변
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
                answer = "없음"
            
            print("\n" + "="*70)
            print(f"💡 {answer}")
            print("="*70)
            
            return {"sql": sql, "result": result, "answer": answer}
            
        except Exception as e:
            print(f"\n❌ {e}")
            return None

# 실행
if __name__ == "__main__":
    MODEL_PATH = "./models/sql-generator-spider-plus-company"
    
    bot = FinalSQLBot(MODEL_PATH)
    
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
