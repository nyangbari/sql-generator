#!/usr/bin/env python3
# sql_chatbot.py
# LangChain 최신 버전 + MySQLdb 문제 해결

import os
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# 최신 LangChain import
from langchain_huggingface import HuggingFacePipeline  # 업데이트!
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

load_dotenv()

class MultiProjectSQLBot:
    def __init__(self, model_path):
        """LangChain 기반 멀티 프로젝트 SQL Bot"""
        
        print("="*70)
        print("🤖 SQL 챗봇 시작")
        print("="*70)
        
        # Device 확인
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"✅ Device: {self.device}")
        
        # 모델 로드
        print("🔄 모델 로딩...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "codellama/CodeLlama-7b-Instruct-hf",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        model = PeftModel.from_pretrained(base_model, model_path)
        tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
        
        # LangChain 파이프라인
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=250,
            temperature=0.1,
            do_sample=True
        )
        
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        # DB 설정 (pymysql 사용!)
        knightfury_uri = os.getenv("KNIGHTFURY_DB_URI")
        furyx_uri = os.getenv("FURYX_DB_URI")
        
        # mysql:// → mysql+pymysql:// 변경
        self.databases = {}
        if knightfury_uri:
            self.databases["knightfury"] = knightfury_uri.replace("mysql://", "mysql+pymysql://")
        if furyx_uri:
            self.databases["furyx"] = furyx_uri.replace("mysql://", "mysql+pymysql://")
        
        # 설정 확인
        print("\n📚 프로젝트 설정:")
        for project, uri in self.databases.items():
            if uri:
                # 비밀번호 숨기기
                safe_uri = uri.split('@')[-1]
                print(f"  ✅ {project}: mysql+pymysql://***@{safe_uri}")
            else:
                print(f"  ⚠️  {project}: 설정 안 됨")
        
        self.agents = {}
        
        print("\n✅ 준비 완료!")
        print("="*70)
    
    def get_agent(self, project):
        """프로젝트별 Agent 가져오기 (캐싱)"""
        
        project = project.lower()
        
        if project not in self.agents:
            uri = self.databases.get(project)
            
            if not uri:
                available = list(self.databases.keys())
                raise ValueError(
                    f"❌ 프로젝트 '{project}'를 찾을 수 없습니다.\n"
                    f"사용 가능한 프로젝트: {', '.join(available)}"
                )
            
            print(f"\n🔗 {project} DB 연결 중...")
            
            try:
                # MySQL DB 연결 (pymysql 사용)
                db = SQLDatabase.from_uri(uri)
                
                # 테이블 목록 출력
                tables = db.get_usable_table_names()
                print(f"📊 테이블 발견: {len(tables)}개")
                print(f"   {', '.join(tables[:5])}{'...' if len(tables) > 5 else ''}")
                
                # SQL Agent 생성
                agent = create_sql_agent(
                    llm=self.llm,
                    db=db,
                    verbose=True,
                    handle_parsing_errors=True
                )
                
                self.agents[project] = agent
                print(f"✅ {project} 연결 완료!\n")
                
            except Exception as e:
                raise ConnectionError(f"❌ {project} DB 연결 실패: {str(e)}")
        
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
            
            result = agent.run(question)
            
            print("\n" + "="*70)
            print(f"💡 답변: {result}")
            print("="*70)
            
            return result
        
        except Exception as e:
            error_msg = f"오류: {str(e)}"
            print(f"\n❌ {error_msg}\n")
            return error_msg
    
    def list_projects(self):
        """설정된 프로젝트 목록"""
        configured = list(self.databases.keys())
        print(f"\n📚 사용 가능한 프로젝트: {', '.join(configured)}")
        return configured
    
    def interactive(self):
        """대화형 모드"""
        print("\n🎯 대화형 모드 시작!")
        print("\n명령어:")
        print("  - 'list': 프로젝트 목록 보기")
        print("  - 'switch <프로젝트명>': 프로젝트 변경")
        print("  - 'exit' or 'quit': 종료")
        print("="*70)
        
        current_project = None
        
        while True:
            try:
                if current_project:
                    user_input = input(f"\n[{current_project}] 질문> ")
                else:
                    user_input = input(f"\n질문> ")
                
                if not user_input.strip():
                    continue
                
                # 명령어 처리
                cmd = user_input.lower().strip()
                
                if cmd in ['exit', 'quit', 'q']:
                    print("\n👋 종료합니다!")
                    break
                
                elif cmd == 'list':
                    self.list_projects()
                    continue
                
                elif cmd.startswith('switch '):
                    project = cmd.split()[1]
                    if project in self.databases:
                        current_project = project
                        print(f"✅ {current_project} 프로젝트로 전환되었습니다")
                    else:
                        print(f"❌ '{project}' 프로젝트를 찾을 수 없습니다")
                        self.list_projects()
                    continue
                
                # 질문 처리
                if not current_project:
                    print("❌ 먼저 프로젝트를 선택하세요")
                    print("   예: switch knightfury")
                    self.list_projects()
                    continue
                
                self.ask(current_project, user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 종료합니다!")
                break
            except Exception as e:
                print(f"❌ 오류: {e}")

# 메인 실행
if __name__ == "__main__":
    bot = MultiProjectSQLBot("./models/sql-generator-spider-plus-company")
    
    # 프로젝트 목록 표시
    bot.list_projects()
    
    # 간단한 테스트
    print("\n" + "="*70)
    print("🧪 기본 테스트")
    print("="*70)
    
    try:
        # KnightFury 테스트
        bot.ask("knightfury", "테이블 목록을 보여줘")
        
    except Exception as e:
        print(f"테스트 중 오류: {e}")
    
    # 대화형 모드 (주석 해제하면 활성화)
    # bot.interactive()
