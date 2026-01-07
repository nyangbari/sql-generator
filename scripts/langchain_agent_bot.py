#!/usr/bin/env python3
# langchain_agent_bot.py
# LangChain Agent + 스키마 검증 + 환각 방지

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

class ReadOnlySQLDatabase(SQLDatabase):
    """Read-Only + 실행 로그 DB"""
    
    WRITE_KEYWORDS = [
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 
        'ALTER', 'TRUNCATE', 'REPLACE', 'MERGE'
    ]
    
    def run(self, command: str, fetch: str = "all", **kwargs):
        """SQL 실행 전후 검증 및 로깅"""
        
        # SQL 정규화
        sql_upper = command.upper().strip()
        
        # Write 작업 차단
        for keyword in self.WRITE_KEYWORDS:
            if keyword in sql_upper:
                raise ValueError(f"🚫 {keyword} 차단! SELECT만 허용")
        
        # SELECT 계열만 허용
        if not any(sql_upper.startswith(k) for k in ['SELECT', 'SHOW', 'DESCRIBE']):
            raise ValueError("🚫 SELECT/SHOW/DESCRIBE만 허용")
        
        # 실행 전 로그
        print(f"\n" + "="*70)
        print(f"🔍 [실제 DB에서 실행할 SQL]")
        print(command)
        print("="*70)
        
        # 실행
        result = super().run(command, fetch=fetch, **kwargs)
        
        # 실행 후 로그
        print(f"\n📊 [실제 DB 결과]")
        if result:
            print(result)
        else:
            print("(결과 없음 또는 NULL)")
        print("="*70 + "\n")
        
        return result

class LangChainAgentBot:
    def __init__(self, model_path):
        """LangChain Agent Bot with Enhanced Validation"""
        
        print("="*70)
        print("🤖 LangChain Agent SQL Bot")
        print("   - 스키마 검증 강화")
        print("   - 환각 방지")
        print("   - Read-Only 보안")
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
        
        # LoRA 어댑터 로드
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        
        print("✅ Spider + Company 학습 모델 로드 완료!")
        
        # Pipeline 생성
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
        """프로젝트별 DB 연결 (캐싱)"""
        project = project.lower()
        
        if project not in self.db_connections:
            uri = self.databases.get(project)
            if not uri:
                raise ValueError(f"프로젝트 '{project}' 없음")
            
            self.db_connections[project] = ReadOnlySQLDatabase.from_uri(uri)
        
        return self.db_connections[project]
    
    def verify_schema(self, project, tables):
        """실제 스키마 확인 및 출력"""
        
        db = self.get_db(project)
        
        print(f"\n📋 [{project}] 실제 테이블 스키마:")
        print("="*70)
        
        for table in tables:
            try:
                schema = db.get_table_info([table])
                print(schema)
            except Exception as e:
                print(f"⚠️ {table}: {e}")
        
        print("="*70)
    
    def get_agent(self, project):
        """프로젝트별 Agent 생성"""
        
        project = project.lower()
        
        if project not in self.agents:
            db = self.get_db(project)
            
            print(f"\n🔗 [{project}] Agent 생성 중...")
            
            # 테이블 목록
            tables = db.get_usable_table_names()
            print(f"📊 테이블 발견: {len(tables)}개")
            print(f"   {', '.join(tables[:10])}")
            
            # Agent 생성
            self.agents[project] = create_sql_agent(
                llm=self.llm,
                db=db,
                agent_type="zero-shot-react-description",
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=10,
                max_execution_time=120
            )
            
            print("✅ Agent 생성 완료!")
        
        return self.agents[project]
    
    def ask(self, project, question):
        """질문 처리 with 강화된 검증"""
        
        print("\n" + "="*70)
        print(f"📂 프로젝트: {project}")
        print(f"💬 질문: {question}")
        print("="*70)
        
        try:
            # 1. 스키마 먼저 확인 (주요 테이블)
            main_tables = ['fury_users', 'knightfury_users', 'users']  # 예상되는 주요 테이블
            db = self.get_db(project)
            all_tables = db.get_usable_table_names()
            
            # 실제 존재하는 테이블만 확인
            existing_tables = [t for t in main_tables if t in all_tables]
            if not existing_tables:
                existing_tables = all_tables[:3]  # 처음 3개
            
            self.verify_schema(project, existing_tables)
            
            # 2. 실제 스키마 정보 가져오기
            schema_info = db.get_table_info(existing_tables)
            
            # 3. Agent에게 명확한 지시
            enhanced_prompt = f"""You are a SQL expert. You MUST follow these rules:

DATABASE SCHEMA (ACTUAL):
{schema_info}

CRITICAL RULES:
1. Use ONLY the columns shown in the schema above
2. Do NOT assume columns like 'id', 'email', 'password' unless they exist in schema
3. If you execute a query and get NULL/empty result, report it honestly
4. NEVER make up data or names like "최준호", "홍길동", "test@example.com"
5. If data doesn't exist, say "No data found"

Question: {question}

Steps to follow:
1. Look at the ACTUAL schema above
2. Identify the correct table and columns
3. Write SQL query using ONLY existing columns
4. Execute the query
5. Report the ACTUAL result (if NULL, say "No data")

Begin:"""
            
            # 4. Agent 실행
            agent = self.get_agent(project)
            
            print("\n🤔 Agent 실행 중...\n")
            
            result = agent.invoke({"input": enhanced_prompt})
            
            # 5. 결과 추출
            if isinstance(result, dict):
                answer = result.get('output', str(result))
            else:
                answer = str(result)
            
            # 6. 환각 검증
            suspicious_patterns = [
                '최준호', '홍길동', '김철수', '이영희',
                'test@example.com', 'user@test.com',
                'password123', 'admin123'
            ]
            
            if any(pattern in answer for pattern in suspicious_patterns):
                print("\n⚠️  경고: 환각 가능성 있는 답변 감지!")
                print("   실제 DB 결과를 다시 확인하세요.")
            
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
        """테이블 목록 및 스키마 확인"""
        
        try:
            db = self.get_db(project)
            tables = db.get_usable_table_names()
            
            print(f"\n📊 [{project}] 테이블 목록 ({len(tables)}개):")
            for i, table in enumerate(tables, 1):
                print(f"  {i}. {table}")
            
            # 처음 3개 테이블 스키마 출력
            if tables:
                print(f"\n📋 상세 스키마 (처음 3개):")
                self.verify_schema(project, tables[:3])
        
        except Exception as e:
            print(f"❌ 오류: {e}")
    
    def interactive(self, project):
        """대화형 모드"""
        
        print(f"\n🎯 대화형 모드 시작! (프로젝트: {project})")
        print("\n명령어:")
        print("  - 'tables': 테이블 목록 및 스키마")
        print("  - 'schema <테이블명>': 특정 테이블 스키마")
        print("  - 'exit' or 'quit': 종료")
        print("="*70)
        
        while True:
            try:
                user_input = input(f"\n[{project}] 질문> ").strip()
                
                if not user_input:
                    continue
                
                cmd = user_input.lower()
                
                if cmd in ['exit', 'quit', 'q']:
                    print("\n👋 종료!")
                    break
                
                elif cmd == 'tables':
                    self.list_tables(project)
                    continue
                
                elif cmd.startswith('schema '):
                    table_name = cmd.split()[1]
                    self.verify_schema(project, [table_name])
                    continue
                
                # 질문 처리
                self.ask(project, user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 종료!")
                break
            except Exception as e:
                print(f"❌ 오류: {e}")

# 실행
if __name__ == "__main__":
    MODEL_PATH = "./models/sql-generator-spider-plus-company"
    
    bot = LangChainAgentBot(MODEL_PATH)
    
    if len(sys.argv) > 2:
        # 명령행 인자: python script.py <project> "<question>"
        bot.ask(sys.argv[1], sys.argv[2])
    
    elif len(sys.argv) > 1:
        # 대화형 모드: python script.py <project>
        bot.interactive(sys.argv[1])
    
    else:
        # 기본 테스트
        print("\n" + "="*70)
        print("🧪 테스트 모드")
        print("="*70)
        
        # 테이블 목록 먼저 확인
        bot.list_tables("knightfury")
        
        # 테스트 질문
        bot.ask("knightfury", "fury_users 테이블에는 총 몇 개의 레코드가 있어?")
