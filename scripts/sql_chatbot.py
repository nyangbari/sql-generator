#!/usr/bin/env python3
# sql_chatbot.py
# LangChain 최신 버전 (invoke 사용)

import os
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

from langchain_huggingface import HuggingFacePipeline
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

load_dotenv()

class MultiProjectSQLBot:
    def __init__(self, model_path):
        """LangChain 기반 멀티 프로젝트 SQL Bot"""
        
        print("="*70)
        print("🤖 SQL 챗봇 시작")
        print("="*70)
        
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"✅ Device: {self.device}")
        
        print("🔄 모델 로딩...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "codellama/CodeLlama-7b-Instruct-hf",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        model = PeftModel.from_pretrained(base_model, model_path)
        tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=250,
            temperature=0.1,
            do_sample=True
        )
        
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        # DB 설정 (pymysql 사용)
        knightfury_uri = os.getenv("KNIGHTFURY_DB_URI")
        furyx_uri = os.getenv("FURYX_DB_URI")
        
        self.databases = {}
        if knightfury_uri:
            self.databases["knightfury"] = knightfury_uri.replace("mysql://", "mysql+pymysql://")
        if furyx_uri:
            self.databases["furyx"] = furyx_uri.replace("mysql://", "mysql+pymysql://")
        
        print("\n📚 프로젝트 설정:")
        for project, uri in self.databases.items():
            if uri:
                safe_uri = uri.split('@')[-1]
                print(f"  ✅ {project}: mysql+pymysql://***@{safe_uri}")
        
        self.agents = {}
        
        print("\n✅ 준비 완료!")
        print("="*70)
    
    def get_agent(self, project):
        """프로젝트별 Agent"""
        
        project = project.lower()
        
        if project not in self.agents:
            uri = self.databases.get(project)
            
            if not uri:
                available = list(self.databases.keys())
                raise ValueError(f"프로젝트 '{project}' 없음. 사용 가능: {', '.join(available)}")
            
            print(f"\n🔗 {project} DB 연결 중...")
            
            try:
                db = SQLDatabase.from_uri(uri)
                
                tables = db.get_usable_table_names()
                print(f"📊 테이블: {len(tables)}개")
                print(f"   {', '.join(tables[:5])}{'...' if len(tables) > 5 else ''}")
                
                # Agent 생성 (더 관대한 설정)
                agent = create_sql_agent(
                    llm=self.llm,
                    db=db,
                    verbose=True,
                    handle_parsing_errors=True,  # 파싱 에러 자동 처리
                    max_iterations=3,  # 반복 제한
                    max_execution_time=30  # 시간 제한
                )
                
                self.agents[project] = agent
                print(f"✅ {project} 연결 완료!\n")
                
            except Exception as e:
                raise ConnectionError(f"{project} DB 연결 실패: {str(e)}")
        
        return self.agents[project]
    
    def ask(self, project, question):
        """질문하기"""
        
        print("\n" + "="*70)
        print(f"📂 프로젝트: {project}")
        print(f"💬 질문: {question}")
        print("="*70)
        
        try:
            agent = self.get_agent(project)
            
            print("🤔 SQL 생성 중...\n")
            
            # invoke 사용 (run은 deprecated)
            result = agent.invoke({"input": question})
            
            # 결과 추출
            if isinstance(result, dict):
                answer = result.get('output', str(result))
            else:
                answer = str(result)
            
            print("\n" + "="*70)
            print(f"💡 답변: {answer}")
            print("="*70)
            
            return answer
        
        except Exception as e:
            error_msg = f"오류: {str(e)}"
            print(f"\n❌ {error_msg}\n")
            return error_msg
    
    def ask_simple(self, project, question):
        """간단 SQL 생성 (Agent 없이)"""
        
        print("\n" + "="*70)
        print(f"📂 프로젝트: {project}")
        print(f"💬 질문: {question}")
        print("="*70)
        
        try:
            uri = self.databases.get(project.lower())
            if not uri:
                return "프로젝트를 찾을 수 없습니다"
            
            # DB 연결
            db = SQLDatabase.from_uri(uri)
            
            # 스키마 정보
            table_info = db.get_table_info()
            
            # 프롬프트 생성
            prompt = f"""Given the following database schema:

{table_info}

Question: {question}

Generate a SQL query to answer this question.

SQL Query:"""
            
            # LLM으로 SQL 생성
            result = self.llm.invoke(prompt)
            
            # SQL 추출
            if isinstance(result, str):
                sql = result
            else:
                sql = result.get('text', str(result))
            
            # SQL만 추출
            if "SQL Query:" in sql:
                sql = sql.split("SQL Query:")[-1].strip()
            
            sql = sql.replace('```sql', '').replace('```', '').strip()
            sql = sql.split('\n')[0] if '\n\n' in sql else sql
            
            print(f"\n💾 생성된 SQL:\n{sql}")
            
            # SQL 실행
            print("\n🔄 실행 중...")
            result = db.run(sql)
            
            print(f"\n💡 결과:\n{result}")
            print("="*70)
            
            return result
            
        except Exception as e:
            error_msg = f"오류: {str(e)}"
            print(f"\n❌ {error_msg}\n")
            return error_msg
    
    def list_projects(self):
        """프로젝트 목록"""
        configured = list(self.databases.keys())
        print(f"\n📚 사용 가능한 프로젝트: {', '.join(configured)}")
        return configured

# 메인 실행
if __name__ == "__main__":
    bot = MultiProjectSQLBot("./models/sql-generator-spider-plus-company")
    
    bot.list_projects()
    
    print("\n" + "="*70)
    print("🧪 테스트")
    print("="*70)
    
    # 간단 버전 사용 (더 안정적)
    bot.ask_simple("knightfury", "총 사용자 수는?")
