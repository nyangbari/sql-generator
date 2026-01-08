#!/usr/bin/env python3
# sqlcoder_rag_production.py
# SQLCoder-7B-2 + RAG + 실제 DB

import os
import sys
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sqlalchemy import inspect

load_dotenv()

class SQLCoderRAGBot:
    """SQLCoder + RAG 최종 버전"""
    
    def __init__(self):
        print("="*70)
        print("🚀 SQLCoder-7B-2 + RAG Production Bot")
        print("="*70)
        
        print("\n🔄 SQLCoder 로딩...")
        
        model_id = "defog/sqlcoder-7b-2"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True
        )
        
        print("✅ SQLCoder 로드!")
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            max_new_tokens=300,
            temperature=0.1,
            return_full_text=False
        )
        
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        self.databases = {}
        for proj in ["KNIGHTFURY", "FURYX"]:
            uri = os.getenv(f"{proj}_DB_URI")
            if uri:
                self.databases[proj.lower()] = uri.replace("mysql://", "mysql+pymysql://")
        
        print(f"\n📚 프로젝트: {', '.join(self.databases.keys())}")
        
        self.vector_stores = {}
        self.table_info_cache = {}
        
        print("\n🔄 RAG 인덱스 생성 중...")
        for proj, uri in self.databases.items():
            self._build_rag_index(proj, uri)
        
        print("\n✅ 완료!")
        print("="*70)
    
    def _build_rag_index(self, project, uri):
        """실제 DB 스키마로 RAG 인덱스 구축"""
        
        try:
            db = SQLDatabase.from_uri(uri, sample_rows_in_table_info=0)
            inspector = inspect(db._engine)
            
            all_tables = db.get_usable_table_names()
            documents = []
            table_info = {}
            
            print(f"   {project}: {len(all_tables)}개 테이블 인덱싱...")
            
            table_descriptions = {
                'fury_action_configs': 'KnightFury dashboard basic actions/quests/missions. Platform-level missions like connect Telegram, Discord, Twitter. System missions, not project quests.',
                'fury_mission_configs': 'Mission type definitions and templates. Defines KINDS of missions/quests: quiz, visit, NFT mint, swap. Mission categories and types.',
                'fury_projects': 'Projects registered with KnightFury. Companies wanting users to complete missions/quests. General project information.',
                'fury_airdrop_projects': 'Projects planning airdrops. Airdrop-specific projects only. Token airdrop projects.',
                'fury_project_missions': 'Quests that projects registered. Project-specific missions/quests. Which missions each project has. Project quests/missions.',
                'fury_user_project_missions': 'User mission completion tracking. Which users completed which missions/quests. User progress on quests/missions.',
                'fury_users': 'User accounts and profiles. User information: wallet, username, social connections.',
                'fury_play_games': 'Games available on platform. Game information and details.',
                'fury_spin_events': 'Spin wheel events. Lucky spin game events/missions.',
            }
            
            for table in all_tables:
                try:
                    columns = inspector.get_columns(table)
                    pk = inspector.get_pk_constraint(table)
                    pk_cols = pk.get('constrained_columns', [])
                    
                    col_names = [col['name'] for col in columns]
                    col_types = {col['name']: str(col['type']) for col in columns}
                    
                    create_stmt = f"CREATE TABLE {table} (\n"
                    col_defs = []
                    
                    for col in columns:
                        col_type = str(col['type'])
                        
                        if 'INT' in col_type.upper():
                            col_type = "INT"
                        elif 'VARCHAR' in col_type.upper() or 'CHAR' in col_type.upper():
                            col_type = "VARCHAR(100)"
                        elif 'TEXT' in col_type.upper():
                            col_type = "TEXT"
                        elif 'DATE' in col_type.upper() or 'TIME' in col_type.upper():
                            col_type = "DATETIME"
                        elif 'DECIMAL' in col_type.upper() or 'NUMERIC' in col_type.upper():
                            col_type = "DECIMAL"
                        
                        pk_marker = " PRIMARY KEY" if col['name'] in pk_cols else ""
                        col_defs.append(f"    {col['name']} {col_type}{pk_marker}")
                    
                    create_stmt += ",\n".join(col_defs) + "\n)"
                    
                    description = table_descriptions.get(
                        table,
                        f"Table containing {table.replace('fury_', '').replace('_', ' ')} related data"
                    )
                    
                    search_text = f"""Table: {table}
Purpose: {description}
Columns: {', '.join(col_names)}
Use for queries about: {description}
Schema:
{create_stmt}"""
                    
                    doc = Document(
                        page_content=search_text,
                        metadata={
                            "table": table,
                            "columns": col_names,
                            "types": col_types,
                            "create_statement": create_stmt,
                            "description": description
                        }
                    )
                    
                    documents.append(doc)
                    table_info[table] = {
                        "columns": col_names,
                        "types": col_types,
                        "create_statement": create_stmt,
                        "description": description
                    }
                    
                except Exception as e:
                    print(f"      ⚠️  {table}: {e}")
            
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            vector_store = FAISS.from_documents(documents, embeddings)
            
            self.vector_stores[project] = vector_store
            self.table_info_cache[project] = table_info
            
            print(f"      ✅ {len(documents)}개 테이블 인덱싱 완료!")
            
        except Exception as e:
            print(f"      ❌ {project} RAG 구축 실패: {e}")
    
    def retrieve_relevant_tables(self, project, question, k=3):
        """질문에 관련된 테이블 RAG 검색"""
        
        if project not in self.vector_stores:
            return []
        
        vector_store = self.vector_stores[project]
        docs = vector_store.similarity_search(question, k=k)
        
        tables = []
        for doc in docs:
            table_name = doc.metadata["table"]
            create_stmt = doc.metadata["create_statement"]
            tables.append({
                "name": table_name,
                "schema": create_stmt
            })
        
        return tables
    
    def generate_sql(self, question, tables):
        """SQLCoder로 SQL 생성"""
        
        schema = "\n\n".join([t["schema"] for t in tables])
        
        prompt = f"""### Task
Generate a SQL query to answer the following question: `{question}`

### Database Schema
{schema}

### Answer
Given the database schema, here is the SQL query that answers `{question}`:
```sql
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.pipeline.model.device)
        
        with torch.no_grad():
            outputs = self.llm.pipeline.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "```sql" in result:
            sql = result.split("```sql")[-1].split("```")[0].strip()
        else:
            after_answer = result.split("### Answer")[-1]
            lines = after_answer.strip().split('\n')
            sql_lines = []
            for line in lines:
                if line.strip().upper().startswith('SELECT') or sql_lines:
                    sql_lines.append(line)
                    if ';' in line:
                        break
            sql = '\n'.join(sql_lines).strip()
        
        sql = sql.replace('```sql', '').replace('```', '').strip()
        if ';' in sql:
            sql = sql.split(';')[0].strip()
        
        return sql
    
    def validate_sql(self, sql):
        """SQL 검증"""
        
        sql_upper = sql.upper()
        
        dangerous = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE']
        if any(kw in sql_upper for kw in dangerous):
            return None, "🚫 위험한 SQL"
        
        if not sql_upper.strip().startswith('SELECT'):
            return None, "⚠️  SELECT만 가능"
        
        return sql, None
    
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
            print("\n🔍 Step 1: RAG 검색...")
            
            relevant_tables = self.retrieve_relevant_tables(project, question, k=3)
            
            if not relevant_tables:
                print("❌ 관련 테이블 없음")
                return None
            
            print(f"   찾은 테이블: {[t['name'] for t in relevant_tables]}")
            
            print("\n🔄 Step 2: SQL 생성...")
            
            sql = self.generate_sql(question, relevant_tables)
            
            print(f"\n💾 생성된 SQL:")
            print(sql)
            
            sql, error = self.validate_sql(sql)
            
            if error:
                print(f"\n{error}")
                return None
            
            print("\n🔄 Step 3: 실행...")
            
            db = SQLDatabase.from_uri(uri, sample_rows_in_table_info=0)
            result = db.run(sql)
            
            print(f"\n📊 결과:")
            print(result)
            
            if result and result != "[]":
                try:
                    if '[(' in str(result):
                        num = str(result).split('(')[1].split(',')[0].strip()
                        
                        if result.count('(') > 1:
                            answer = f"결과:\n{result}"
                        else:
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
            
            return {
                "tables": [t['name'] for t in relevant_tables],
                "sql": sql,
                "result": result,
                "answer": answer
            }
            
        except Exception as e:
            print(f"\n❌ {e}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == "__main__":
    
    bot = SQLCoderRAGBot()
    
    if len(sys.argv) > 2:
        bot.ask(sys.argv[1], sys.argv[2])
    else:
        projects = list(bot.databases.keys())
        
        if not projects:
            print("\n❌ 프로젝트 없음")
            sys.exit(1)
        
        print(f"\n📚 프로젝트: {', '.join(projects)}")
        project = input("선택: ").strip().lower()
        
        if project not in projects:
            print(f"❌ '{project}' 없음")
            sys.exit(1)
        
        print(f"\n✅ '{project}' 선택")
        print("\n💬 질문 입력 (종료: exit)")
        print("")
        
        while True:
            try:
                question = input(f"\n[{project}] ").strip()
                
                if question.lower() in ['exit', 'quit', 'q']:
                    print("\n👋")
                    break
                
                if question:
                    bot.ask(project, question)
                    
            except KeyboardInterrupt:
                print("\n\n👋")
                break
