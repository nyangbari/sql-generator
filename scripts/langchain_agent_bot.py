#!/usr/bin/env python3
import os
import torch
import sys
from typing import Any, List, Optional
from dotenv import load_dotenv

# 라이브러리 로드
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from langchain_huggingface import HuggingFacePipeline
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent

load_dotenv()

# --- [1] 보안 및 읽기 전용 DB 클래스 ---
class ReadOnlySQLDatabase(SQLDatabase):
    WRITE_KEYWORDS = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE', 'REPLACE', 'MERGE']
    
    def run(self, command: str, fetch: str = "all", **kwargs):
        sql_upper = command.upper().strip()
        # 쓰기 명령어 차단
        for keyword in self.WRITE_KEYWORDS:
            if keyword in sql_upper:
                raise ValueError(f"🚫 보안 경고: {keyword} 명령어는 실행할 수 없습니다. SELECT만 가능합니다.")
        
        # 허용된 조회 명령어 체크
        if not any(sql_upper.startswith(k) for k in ['SELECT', 'SHOW', 'DESCRIBE']):
            raise ValueError("🚫 허용되지 않은 쿼리입니다. SELECT, SHOW, DESCRIBE만 사용하세요.")
        
        print(f"✅ [DB 조회 실행]")
        return super().run(command, fetch=fetch, **kwargs)

# --- [2] 메인 에이전트 봇 클래스 ---
class LangChainAgentBot:
    def __init__(self, model_path):
        print("🚀 모델 로딩 중 (RTX 4060 Ti 8-bit 최적화 모드)...")
        base_model_id = "codellama/CodeLlama-7b-Instruct-hf"
        
        # 16GB VRAM 안성맞춤 세팅
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True
        )
        
        # 어댑터 결합
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        # 환경변수에서 DB URI 로드
        self.databases = {}
        for proj in ["KNIGHTFURY", "FURYX"]:
            uri = os.getenv(f"{proj}_DB_URI")
            if uri:
                self.databases[proj.lower()] = uri.replace("mysql://", "mysql+pymysql://")
        
        self.agents = {}
        print("✅ 시스템 준비 완료!")

    def get_agent(self, project):
        project = project.lower()
        if project not in self.agents:
            uri = self.databases.get(project)
            if not uri:
                raise ValueError(f"❌ '{project}' DB 설정이 .env에 없습니다.")
            
            db = ReadOnlySQLDatabase.from_uri(uri)
            
            # 에이전트 생성 (문자열 타입 지정으로 임포트 에러 원천 차단)
            self.agents[project] = create_sql_agent(
                llm=self.llm,
                db=db,
                agent_type="zero-shot-react-description",
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=10 # 루프 방지용 최대 실행 횟수
            )
        return self.agents[project]

    def ask(self, project, question):
        print(f"\n📂 프로젝트: {project} | 💬 질문: {question}")
        try:
            agent = self.get_agent(project)
            # 환각 방지 및 답변 형식 강제 프롬프트
            prompt = (
                f"당신은 SQL 전문가입니다. 반드시 아래 순서대로 작업하세요:\n"
                f"1. sql_db_list_tables로 어떤 테이블이 있는지 먼저 확인한다.\n"
                f"2. 실제 확인된 테이블에 대해서만 SQL을 생성한다.\n"
                f"3. 결과는 한국어로 친절하게 답변한다.\n"
                f"질문: {question}"
            )
            result = agent.invoke({"input": prompt})
            
            # 결과 추출
            answer = result.get('output') if isinstance(result, dict) else str(result)
            print("-" * 50)
            print(f"💡 최종 답변: {answer}")
            print("-" * 50)
            return answer
        except Exception as e:
            print(f"❌ 에러 발생: {e}")
            return None

# --- [3] 실행부 (명령행 인자 지원) ---
if __name__ == "__main__":
    # 1. 모델 경로 설정
    MODEL_PATH = "/home/dongsucat1/ai/sql-generator/models/sql-generator-spider-plus-company"
    bot = LangChainAgentBot(MODEL_PATH)

    # 2. 터미널 명령행 인자가 있을 경우 (예: python scripts/bot.py knightfury "질문")
    if len(sys.argv) > 2:
        proj_arg = sys.argv[1]
        quest_arg = sys.argv[2]
        bot.ask(proj_arg, quest_arg)
    else:
        # 인자가 없을 경우 기본 테스트 질문 실행
        bot.ask("knightfury", "현재 DB에 어떤 테이블들이 있는지 이름만 다 알려줘.")
