#!/usr/bin/env python3
import os
from typing import Any, List, Optional
from dotenv import load_dotenv
from pydantic import Field

from vllm import LLM as VLLM_Model, SamplingParams
from vllm.lora.request import LoRARequest

from langchain_core.language_models.llms import LLM
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent

load_dotenv()

class ReadOnlySQLDatabase(SQLDatabase):
    WRITE_KEYWORDS = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE', 'REPLACE', 'MERGE']
    def run(self, command: str, fetch: str = "all", **kwargs):
        sql_upper = command.upper().strip()
        for keyword in self.WRITE_KEYWORDS:
            if keyword in sql_upper: raise ValueError(f"🚫 {keyword} 권한 없음")
        return super().run(command, fetch=fetch, **kwargs)

class VLLMWrapper(LLM):
    vllm_model: Any = Field(default=None, exclude=True)
    sampling_params: Any = Field(default=None, exclude=True)
    lora_request: Any = Field(default=None, exclude=True)

    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        print("🔄 vLLM 로딩 (RTX 4060 Ti 16GB 전용 최적화)...")
        
        # 실제 학습 시 사용한 베이스 모델
        base_model = "codellama/CodeLlama-7b-Instruct-hf"

        self.vllm_model = VLLM_Model(
            model=base_model,
            enable_lora=True,
            max_lora_rank=64,
            tensor_parallel_size=1,
            # --- OOM 해결을 위한 최종 설정 ---
            gpu_memory_utilization=0.85, # 16GB 중 약 13.6GB 사용 예약
            max_model_len=512,           # 컨텍스트 길이를 512로 제한 (캐시 메모리 최소화)
            enforce_eager=True,          # CUDA Graph 생성 방지 (WSL 메모리 스파이크 방지)
            disable_custom_all_reduce=True,
            # -------------------------------
            dtype="float16"
        )
        
        self.lora_request = LoRARequest("sql_adapter", 1, model_path)
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=200,
            stop=["\n\n", "Observation:", "Thought:"]
        )
        print("✅ vLLM 로드 완료!")

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        outputs = self.vllm_model.generate([prompt], self.sampling_params, lora_request=self.lora_request)
        return outputs[0].outputs[0].text

    @property
    def _llm_type(self) -> str: return "vllm_lora_optimized"

class LangChainAgentBot:
    def __init__(self, model_path):
        self.llm = VLLMWrapper(model_path=model_path)
        db_uri = os.getenv("KNIGHTFURY_DB_URI", "").replace("mysql://", "mysql+pymysql://")
        self.db = ReadOnlySQLDatabase.from_uri(db_uri)

    def ask(self, question):
        try:
            agent = create_sql_agent(
                llm=self.llm, db=self.db, agent_type="zero-shot-react-description",
                verbose=True, handle_parsing_errors=True
            )
            print(f"\n🤔 질문: {question}")
            result = agent.invoke({"input": question})
            print(f"\n💡 답변: {result.get('output')}")
        except Exception as e: print(f"❌ 에러: {e}")

if __name__ == "__main__":
    MODEL_PATH = "/home/dongsucat1/ai/sql-generator/models/sql-generator-spider-plus-company"
    bot = LangChainAgentBot(MODEL_PATH)
    bot.ask("전체 사용자 수는 몇 명인가요?")
