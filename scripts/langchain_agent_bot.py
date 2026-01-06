#!/usr/bin/env python3
import os
from typing import Any, List, Optional, Mapping
from dotenv import load_dotenv
from pydantic import Field

# vLLM 및 LangChain 핵심 모듈
from vllm import LLM as VLLM_Model, SamplingParams
from langchain.llms.base import LLM
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_types import AgentType

load_dotenv()

class ReadOnlySQLDatabase(SQLDatabase):
    """실제 DB 수정을 방지하고 SELECT만 허용하는 보안 계층"""
    WRITE_KEYWORDS = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 
                      'ALTER', 'TRUNCATE', 'REPLACE', 'MERGE']

    def run(self, command: str, fetch: str = "all", **kwargs):
        sql_upper = command.upper().strip()
        for keyword in self.WRITE_KEYWORDS:
            if keyword in sql_upper:
                raise ValueError(f"🚫 {keyword} 차단됨! SELECT만 허용됩니다.")
        
        if not any(sql_upper.startswith(k) for k in ['SELECT', 'SHOW', 'DESCRIBE']):
            raise ValueError("🚫 SELECT/SHOW/DESCRIBE 쿼리만 실행 가능합니다.")
        
        print(f"✅ 안전한 쿼리 실행 중...")
        return super().run(command, fetch=fetch, **kwargs)

class VLLMWrapper(LLM):
    """vLLM 엔진을 LangChain LLM 인터페이스로 래핑"""
    vllm_model: Any = Field(default=None, exclude=True)
    sampling_params: Any = Field(default=None, exclude=True)

    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        print("🔄 vLLM 엔진 로딩 중 (RTX 4060 Ti 최적화 모드)...")
        
        # OOM 방지를 위한 핵심 설정
        self.vllm_model = VLLM_Model(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8, # 16GB VRAM 중 80% 점유
            max_model_len=1024,         # 컨텍스트 길이를 줄여 메모리 확보
            enforce_eager=True,         # CUDA Graph 비활성화로 안정성 확보
            dtype="float16"
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=250,
            stop=["\n\n", "Observation:", "Thought:"]
        )
        print("✅ vLLM 로드 완료!")

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        outputs = self.vllm_model.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text

    @property
    def _llm_type(self) -> str:
        return "vllm"

class LangChainAgentBot:
    def __init__(self, model_path):
        print("="*70)
        print("🤖 LangChain Agent SQL Bot (vLLM + 보안 모드)")
        print("="*70)

        # vLLM 초기화
        self.llm = VLLMWrapper(model_path=model_path)

        # DB URI 설정
        k_uri = os.getenv("KNIGHTFURY_DB_URI")
        f_uri = os.getenv("FURYX_DB_URI")

        self.databases = {}
        if k_uri: self.databases["knightfury"] = k_uri.replace("mysql://", "mysql+pymysql://")
        if f_uri: self.databases["furyx"] = f_uri.replace("mysql://", "mysql+pymysql://")

        self.agents = {}
        print(f"\n📚 연결된 프로젝트: {', '.join(self.databases.keys())}")
        print("✅ 시스템 준비 완료!")
        print("="*70)

    def get_agent(self, project):
        project = project.lower()
        if project not in self.agents:
            uri = self.databases.get(project)
            if not uri: raise ValueError(f"❌ 프로젝트 '{project}' 설정이 없습니다.")
            
            # 보안 DB 적용
            db = ReadOnlySQLDatabase.from_uri(uri)
            
            # 에이전트 생성 (Parsing Error 방지를 위한 설정 추가)
            self.agents[project] = create_sql_agent(
                llm=self.llm,
                db=db,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5
            )
            print(f"✅ {project} 에이전트 생성 완료!")
        return self.agents[project]

    def ask(self, project, question):
        print(f"\n📂 [프로젝트: {project}] 질문: {question}")
        try:
            agent = self.get_agent(project)
            # 모델이 형식을 지키도록 프롬프트 보완
            formatted_question = (
                f"Answer the following question in Korean by querying the database.\n"
                f"Question: {question}"
            )
            result = agent.invoke({"input": formatted_question})
            
            answer = result.get('output', str(result)) if isinstance(result, dict) else str(result)
            print(f"\n💡 답변: {answer}")
            return answer
        except Exception as e:
            print(f"\n❌ 에러: {e}")
            return None

    def list_tables(self, project):
        try:
            uri = self.databases.get(project.lower())
            db = ReadOnlySQLDatabase.from_uri(uri)
            tables = db.get_usable_table_names()
            print(f"\n📊 {project} 테이블 목록: {', '.join(tables)}")
        except Exception as e:
            print(f"❌ 테이블 목록 로드 실패: {e}")

if __name__ == "__main__":
    # 모델 경로는 본인의 환경에 맞게 절대 경로로 지정하세요.
    MODEL_PATH = "/home/dongsucat1/ai/sql-generator/models/sql-generator-spider-plus-company"
    
    bot = LangChainAgentBot(MODEL_PATH)
    
    # 1. 테이블 확인
    bot.list_tables("knightfury")
    
    # 2. 질문 테스트
    bot.ask("knightfury", "사용자 테이블의 전체 사용자 수는?")
