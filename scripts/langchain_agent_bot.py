#!/usr/bin/env python3
import os
from typing import Any, List, Optional, Mapping
from dotenv import load_dotenv
from pydantic import Field

# vLLM 및 최신 LangChain 모듈 경로 수정
from vllm import LLM as VLLM_Model, SamplingParams
from langchain_core.language_models.llms import LLM  # 경로 수정됨
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType

load_dotenv()

class ReadOnlySQLDatabase(SQLDatabase):
    """보안이 강화된 Read-Only SQL Database"""
    WRITE_KEYWORDS = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE', 'REPLACE', 'MERGE']

    def run(self, command: str, fetch: str = "all", **kwargs):
        sql_upper = command.upper().strip()
        for keyword in self.WRITE_KEYWORDS:
            if keyword in sql_upper:
                raise ValueError(f"🚫 {keyword} 차단됨! SELECT만 가능합니다.")
        
        if not any(sql_upper.startswith(k) for k in ['SELECT', 'SHOW', 'DESCRIBE']):
            raise ValueError("🚫 허용되지 않은 쿼리 타입입니다.")
        
        print(f"✅ [실제 DB 쿼리 실행]")
        return super().run(command, fetch=fetch, **kwargs)

class VLLMWrapper(LLM):
    vllm_model: Any = Field(default=None, exclude=True)
    sampling_params: Any = Field(default=None, exclude=True)

    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        print("🔄 vLLM 로딩 (RTX 4060 Ti 최적화 모드)...")
        self.vllm_model = VLLM_Model(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            max_model_len=1024, # 메모리 부족 방지를 위해 길이 제한
            enforce_eager=True,
            dtype="float16"
        )
        self.sampling_params = SamplingParams(
            temperature=0.0, # 정확한 SQL 생성을 위해 0으로 설정
            max_tokens=256,
            stop=["Observation:"]
        )

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        outputs = self.vllm_model.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text

    @property
    def _llm_type(self) -> str: return "vllm"

class LangChainAgentBot:
    def __init__(self, model_path):
        self.llm = VLLMWrapper(model_path=model_path)
        self.databases = {}
        # .env 파일에서 URI 로드
        for proj in ["KNIGHTFURY", "FURYX"]:
            uri = os.getenv(f"{proj}_DB_URI")
            if uri: self.databases[proj.lower()] = uri.replace("mysql://", "mysql+pymysql://")
        
        self.agents = {}

    def get_agent(self, project):
        project = project.lower()
        if project not in self.agents:
            uri = self.databases.get(project)
            if not uri: raise ValueError(f"'{project}' URI 없음")
            
            # 실제 DB 연결 및 스키마 정보 강제 로드
            db = ReadOnlySQLDatabase.from_uri(uri)
            
            self.agents[project] = create_sql_agent(
                llm=self.llm,
                db=db,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True
            )
        return self.agents[project]

    def ask(self, project, question):
        print(f"\n📂 프로젝트: {project} | 질문: {question}")
        try:
            agent = self.get_agent(project)
            # 모델이 실제 테이블 목록을 먼저 확인하도록 지시하는 프롬프트
            prompt = (
                f"당신은 SQL 전문가입니다. 반드시 다음 순서를 지키세요:\n"
                f"1. `sql_db_list_tables`로 존재하는 테이블을 확인한다.\n"
                f"2. 질문과 관련된 테이블이 없으면 '정보 없음'이라고 답한다.\n"
                f"3. 테이블이 있으면 `sql_db_schema`를 확인 후 쿼리한다.\n"
                f"질문: {question}"
            )
            result = agent.invoke({"input": prompt})
            print(f"💡 답변: {result.get('output')}")
        except Exception as e:
            print(f"❌ 에러: {e}")

if __name__ == "__main__":
    MODEL_PATH = "/home/dongsucat1/ai/sql-generator/models/sql-generator-spider-plus-company"
    bot = LangChainAgentBot(MODEL_PATH)
    bot.ask("knightfury", "현재 데이터베이스에 존재하는 모든 테이블의 목록을 알려줘.")
