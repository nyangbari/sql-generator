#!/usr/bin/env python3
import os
from typing import Any, List, Optional, Mapping
from dotenv import load_dotenv
from pydantic import Field

# vLLM 및 LoRA 관련
from vllm import LLM as VLLM_Model, SamplingParams
from vllm.lora.request import LoRARequest

# LangChain 최신 표준 경로
from langchain_core.language_models.llms import LLM
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent

load_dotenv()

class ReadOnlySQLDatabase(SQLDatabase):
    """안전한 조회를 위한 Read-Only SQL Database 래퍼"""
    WRITE_KEYWORDS = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 
                      'ALTER', 'TRUNCATE', 'REPLACE', 'MERGE']
    
    def run(self, command: str, fetch: str = "all", **kwargs):
        sql_upper = command.upper().strip()
        for keyword in self.WRITE_KEYWORDS:
            if keyword in sql_upper:
                raise ValueError(f"🚫 {keyword} 명령어가 감지되었습니다! SELECT만 허용됩니다.")
        
        if not any(sql_upper.startswith(k) for k in ['SELECT', 'SHOW', 'DESCRIBE']):
            raise ValueError("🚫 SELECT/SHOW/DESCRIBE 쿼리만 실행 가능합니다.")
        
        print(f"✅ 안전한 쿼리 확인됨")
        return super().run(command, fetch=fetch, **kwargs)

class VLLMWrapper(LLM):
    """vLLM (Base + LoRA)을 LangChain LLM으로 래핑"""
    vllm_model: Any = Field(default=None, exclude=True)
    sampling_params: Any = Field(default=None, exclude=True)
    lora_request: Any = Field(default=None, exclude=True)

    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        print("🔄 vLLM 엔진 및 LoRA 어댑터 로딩 중...")
        
        # 1. 베이스 모델 지정 (학습 시 사용한 모델명)
        # 만약 Qwen2가 아니라면 실제 사용하신 베이스 모델로 수정하세요.
        base_model = "Qwen/Qwen2-7B-Instruct" 

        self.vllm_model = VLLM_Model(
            model=base_model,
            enable_lora=True,          # LoRA 기능 활성화
            max_lora_rank=64,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            max_model_len=4096,
            dtype="float16"
        )
        
        # 2. 어댑터 설정
        self.lora_request = LoRARequest("sql_adapter", 1, model_path)
        
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=300,
            stop=["\n\n\n", "Observation:", "Thought:"]
        )
        print("✅ vLLM 로드 완료!")

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        # 실행 시 lora_request를 포함하여 어댑터 적용
        outputs = self.vllm_model.generate(
            [prompt], 
            self.sampling_params, 
            lora_request=self.lora_request
        )
        return outputs[0].outputs[0].text

    @property
    def _llm_type(self) -> str:
        return "vllm_lora"

class LangChainAgentBot:
    def __init__(self, model_path):
        print("="*70)
        print("🤖 SQL Generator Bot (vLLM + LangChain)")
        print("="*70)
        
        self.llm = VLLMWrapper(model_path=model_path)
        
        # DB URI 처리
        k_uri = os.getenv("KNIGHTFURY_DB_URI", "").replace("mysql://", "mysql+pymysql://")
        f_uri = os.getenv("FURYX_DB_URI", "").replace("mysql://", "mysql+pymysql://")
        
        self.databases = {}
        if k_uri: self.databases["knightfury"] = k_uri
        if f_uri: self.databases["furyx"] = f_uri
        
        self.agents = {}
        print(f"📚 등록된 프로젝트: {list(self.databases.keys())}")

    def get_agent(self, project):
        project = project.lower()
        if project not in self.agents:
            uri = self.databases.get(project)
            if not uri: raise ValueError(f"'{project}' DB 정보를 .env에서 찾을 수 없습니다.")
            
            db = ReadOnlySQLDatabase.from_uri(uri)
            
            # Agent 생성 시 agent_type을 문자열로 직접 지정하여 호환성 확보
            self.agents[project] = create_sql_agent(
                llm=self.llm,
                db=db,
                agent_type="zero-shot-react-description",
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5
            )
        return self.agents[project]

    def ask(self, project, question):
        try:
            agent = self.get_agent(project)
            print(f"\n🤔 '{project}'에 질문하는 중: {question}")
            result = agent.invoke({"input": question})
            answer = result.get('output', str(result)) if isinstance(result, dict) else str(result)
            print(f"\n💡 답변: {answer}")
        except Exception as e:
            print(f"❌ 에러 발생: {e}")

# --- 실행부 ---
if __name__ == "__main__":
    # 윈도우 환경의 실제 절대 경로 사용
    MODEL_PATH = "/home/dongsucat1/ai/sql-generator/models/sql-generator-spider-plus-company"
    
    bot = LangChainAgentBot(MODEL_PATH)
    
    # 1. Knightfury 테스트
    bot.ask("knightfury", "총 사용자 수는 몇 명인가요?")
