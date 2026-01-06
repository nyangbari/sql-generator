#!/usr/bin/env python3
# vllm_sql_bot.py
# vLLM 가속 + Read-Only 안전장치

import os
import re
from dotenv import load_dotenv
from vllm import LLM, SamplingParams
from langchain_community.utilities import SQLDatabase

load_dotenv()

class ReadOnlySQLDatabase(SQLDatabase):
    """Read-Only SQL Database Wrapper"""
    
    WRITE_KEYWORDS = [
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 
        'ALTER', 'TRUNCATE', 'REPLACE', 'MERGE'
    ]
    
    def run(self, command: str, fetch: str = "all", **kwargs):
        """SQL 실행 전 검증"""
        
        # SQL 정규화
        sql_upper = command.upper().strip()
        
        # Write 작업 차단
        for keyword in self.WRITE_KEYWORDS:
            if keyword in sql_upper:
                error_msg = f"🚫 보안 경고: {keyword} 작업은 허용되지 않습니다. SELECT만 가능합니다."
                print(f"\n❌ {error_msg}")
                raise ValueError(error_msg)
        
        # SELECT만 허용
        if not sql_upper.startswith('SELECT') and not sql_upper.startswith('SHOW') and not sql_upper.startswith('DESCRIBE'):
            error_msg = "🚫 보안 경고: SELECT, SHOW, DESCRIBE만 허용됩니다."
            print(f"\n❌ {error_msg}")
            raise ValueError(error_msg)
        
        # 안전하면 실행
        print(f"✅ 안전한 쿼리 확인됨")
        return super().run(command, fetch=fetch, **kwargs)

class VLLMSQLBot:
    def __init__(self, model_path):
        """vLLM 기반 SQL Bot with Read-Only"""
        
        print("="*70)
        print("🚀 vLLM SQL Bot (Read-Only)")
        print("="*70)
        
        # vLLM 모델 로드
        print("\n🔄 vLLM 모델 로딩...")
        print("   (최초 실행 시 1-2분 소요)")
        
        self.vllm_model = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            dtype="float16"
        )
        
        # Sampling 설정
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=250,
            stop=["<|end|>", "\n\n\n"]
        )
        
        print("✅ vLLM 로드 완료!")
        
        # DB 설정
        knightfury_uri = os.getenv("KNIGHTFURY_DB_URI")
        furyx_uri = os.getenv("FURYX_DB_URI")
        
        self.databases = {}
        if knightfury_uri:
            self.databases["knightfury"] = knightfury_uri.replace("mysql://", "mysql+pymysql://")
        if furyx_uri:
            self.databases["furyx"] = furyx_uri.replace("mysql://", "mysql+pymysql://")
        
        print("\n📚 프로젝트 설정:")
        for project in self.databases.keys():
            print(f"  ✅ {project}")
        
        print("\n✅ 준비 완료!")
        print("🔒 보안: SELECT/SHOW/DESCRIBE만 허용")
        print("="*70)
    
    def _validate_sql(self, sql: str) -> bool:
        """SQL 검증"""
        sql_upper = sql.upper().strip()
        
        write_keywords = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 
                         'ALTER', 'TRUNCATE', 'REPLACE', 'MERGE']
        
        for keyword in write_keywords:
            if keyword in sql_upper:
                return False
        
        return sql_upper.startswith('SELECT') or sql_upper.startswith('SHOW') or sql_upper.startswith('DESCRIBE')
    
    def generate_sql(self, project, question):
        """vLLM으로 SQL 생성"""
        
        print(f"\n📂 프로젝트: {project}")
        print(f"💬 질문: {question}")
        
        project = project.lower()
        uri = self.databases.get(project)
        
        if not uri:
            print("❌ 프로젝트를 찾을 수 없습니다")
            return None
        
        try:
            # Read-Only DB 연결
            db = ReadOnlySQLDatabase.from_uri(uri)
            
            # 스키마 (처음 10개 테이블)
            tables = db.get_usable_table_names()[:10]
            table_info = db.get_table_info(tables)
            
            # 프롬프트
            prompt = f"""You are a SQL expert. Generate a SELECT query to answer the question.

Database Schema:
{table_info}

Question: {question}

Important Rules:
- ONLY SELECT queries are allowed
- NO INSERT, UPDATE, DELETE, DROP, CREATE, ALTER
- Return ONLY the SQL query, nothing else

SQL Query:"""
            
            print("\n🤔 vLLM으로 SQL 생성 중...")
            
            # vLLM 생성
            outputs = self.vllm_model.generate([prompt], self.sampling_params)
            result = outputs[0].outputs[0].text
            
            # SQL 추출
            if "SQL Query:" in result:
                sql = result.split("SQL Query:")[-1].strip()
            else:
                sql = result.strip()
            
            sql = sql.replace('```sql', '').replace('```', '').strip()
            
            # 첫 문장만
            if '\n\n' in sql:
                sql = sql.split('\n\n')[0]
            
            # 검증
            if not self._validate_sql(sql):
                print("\n🚫 보안 경고: 안전하지 않은 쿼리가 생성되었습니다!")
                print(f"   생성된 쿼리: {sql}")
                return None
            
            print(f"\n💾 생성된 SQL:")
            print(sql)
            
            return sql
            
        except Exception as e:
            print(f"❌ 오류: {e}")
            return None
    
    def execute_sql(self, project, sql):
        """Read-Only SQL 실행"""
        
        project = project.lower()
        uri = self.databases.get(project)
        
        if not uri:
            print("❌ 프로젝트를 찾을 수 없습니다")
            return None
        
        try:
            # Read-Only DB
            db = ReadOnlySQLDatabase.from_uri(uri)
            
            print("\n🔄 SQL 실행 중...")
            result = db.run(sql)
            
            print(f"\n💡 결과:")
            print(result)
            
            return result
            
        except ValueError as e:
            # 보안 에러
            print(f"\n🚫 {e}")
            return None
        except Exception as e:
            print(f"❌ 실행 오류: {e}")
            return None
    
    def ask(self, project, question):
        """질문 → SQL 생성 → 실행"""
        
        print("\n" + "="*70)
        
        # SQL 생성
        sql = self.generate_sql(project, question)
        
        if sql:
            # SQL 실행
            result = self.execute_sql(project, sql)
            
            print("="*70)
            return result
        else:
            print("="*70)
            return None
    
    def list_tables(self, project):
        """테이블 목록"""
        
        project = project.lower()
        uri = self.databases.get(project)
        
        if not uri:
            print("❌ 프로젝트를 찾을 수 없습니다")
            return
        
        try:
            db = ReadOnlySQLDatabase.from_uri(uri)
            tables = db.get_usable_table_names()
            
            print(f"\n📊 {project} 테이블 ({len(tables)}개):")
            for i, table in enumerate(tables[:30], 1):
                print(f"  {i}. {table}")
            
            if len(tables) > 30:
                print(f"  ... 외 {len(tables)-30}개")
            
        except Exception as e:
            print(f"❌ 오류: {e}")
    
    def interactive(self, project):
        """대화형 모드"""
        
        print(f"\n🎯 대화형 모드 시작! (프로젝트: {project})")
        print("\n명령어:")
        print("  - 'tables': 테이블 목록")
        print("  - 'exit' or 'quit': 종료")
        print("="*70)
        
        while True:
            try:
                user_input = input(f"\n[{project}] 질문> ")
                
                if not user_input.strip():
                    continue
                
                cmd = user_input.lower().strip()
                
                if cmd in ['exit', 'quit', 'q']:
                    print("\n👋 종료합니다!")
                    break
                
                elif cmd == 'tables':
                    self.list_tables(project)
                    continue
                
                # 질문 처리
                self.ask(project, user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 종료합니다!")
                break
            except Exception as e:
                print(f"❌ 오류: {e}")

# 실행
if __name__ == "__main__":
    bot = VLLMSQLBot("./models/sql-generator-spider-plus-company")
    
    # 테이블 목록
    bot.list_tables("knightfury")
    
    # 테스트
    print("\n" + "="*70)
    print("🧪 테스트")
    print("="*70)
    
    # 안전한 쿼리
    bot.ask("knightfury", "총 사용자 수는?")
    
    # 대화형 모드 (주석 해제)
    # bot.interactive("knightfury")
