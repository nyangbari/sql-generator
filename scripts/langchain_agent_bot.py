#!/usr/bin/env python3
import os
from typing import Any, List, Optional, Mapping
from dotenv import load_dotenv
from pydantic import Field

# vLLM 및 LoRA 관련 필수 라이브러리
from vllm import LLM as VLLM_Model, SamplingParams
from vllm.lora.request import LoRARequest

# LangChain 최신 표준 경로 (ImportError 해결)
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
        
        # 기본 조회 쿼리만 허용
        if not any(sql_upper.startswith(k) for k in ['SELECT', 'SHOW', 'DESCRIBE']):
            raise ValueError("🚫 SELECT/SHOW/DESCRIBE 쿼리만 실행 가능합니다.")
        
        print(f"✅ 안전한 쿼리 확인됨: {command[:50]}...")
        return super().run(command, fetch=fetch, **kwargs)

class VLLMWrapper(LLM):
    """vLLM (CodeLlama Base + LoRA 어댑터)을 LangChain LLM으로 래핑"""
    vllm_model: Any = Field(default=None, exclude=True)
    sampling_params: Any = Field(default=None, exclude=True)
    lora_request: Any = Field(default=None, exclude=True)

    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        print("🔄 vLLM 엔진 로딩 (CodeLlama-7b-Instruct-hf + LoRA 어댑터)...")
        
        # 학습 시 사용한 베이스 모델
        base_model = "codellama/CodeLlama-7b-Instruct-hf" 

        self.vllm_model = VLLM_Model(
            model=base_model,
            enable_lora=True,          # LoRA 기능 활성화
            max_lora_rank=64,          # 어댑터 랭크 설정
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            max_model_len=4096,
            dtype="float16"
        )
        
        # 현재 지정된 절대 경로의 LoRA 어댑터 설정
        self.lora_request = LoRARequest("sql_adapter", 1, model_path)
        
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=300,
            stop=["\n\n\n", "Observation:", "Thought:"]
        )
        print("✅ vLLM (CodeLlama) 로드 완료!")

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        # 실행 시 lora_request를 포함하여 학습된 어댑터 적용
        outputs = self.vllm_model.generate(
            [prompt], 
            self.sampling_params, 
            lora_request=self.lora_request
        )
        return outputs[0].outputs[0].text

    @property
    def _llm_type(self) -> str:
        return "vllm_lora_codellama"

class LangChainAgentBot:
    def __init__(self, model_path):
        print("="*70)
        print("🤖 SQL Generator Bot (CodeLlama-LoRA + LangChain)")
        print("="*70)
        
        # LLM 초기화 (여기서 vLLM 로딩 시작)
        self.llm = VLLMWrapper(model_path=model_path)
        
        # .env 파일에서 DB URI 가져오기
        k_uri = os.getenv("KNIGHTFURY_DB_URI", "").replace("mysql://", "mysql+pymysql://")
        f_uri = os.getenv("FURYX_DB_URI", "").replace("mysql://", "mysql+pymysql://")
        
        self.databases = {}
        if k_uri: self.databases["knightfury"] = k_uri
        if f_uri: self.databases["furyx"] = f_uri
        
        self.agents = {}
        print(f"📚 연결 가능한 프로젝트: {list(self.databases.keys())}")

    def get_agent(self, project):
        project = project.lower()
        if project not in self.agents:
            uri = self.databases.get(project)
            if not uri: 
                raise ValueError(f"'{project}' 프로젝트를 찾을 수 없습니다. .env 파일을 확인하세요.")
            
            # Read-Only DB 객체 생성
            db = ReadOnlySQLDatabase.from_uri(uri)
            
            # Agent 생성 (AgentType 열거형 대신 문자열 사용으로 버전 충돌 방지)
            self.agents[project] = create_sql_agent(
                llm=self.llm,
                db=db,
                agent_type="zero-shot-react-description",
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5
            )
            print(f"✅ {project} 에이전트 생성 완료")
        return self.agents[project]

    def ask(self, project, question):
        try:
            agent = self.get_agent(project)
            print(f"\n🤔 질문 실행 중: {question}")
            result = agent.invoke({"input": question})
            
            answer = result.get('output', str(result)) if isinstance(result, dict) else str(result)
            print(f"\n💡 답변: {answer}")
            return answer
        except Exception as e:
            print(f"❌ 에러 발생: {e}")
            return None

# --- 실행부 ---
if __name__ == "__main__":
    # 윈도우 WSL 절대 경로 (이전 pwd 확인 결과 적용)
    MODEL_PATH = "/home/dongsucat1/ai/sql-generator/models/sql-generator-spider-plus-company"
    
    bot = LangChainAgentBot(MODEL_PATH)
    
    # Knightfury 프로젝트 테스트
    print("\n" + "="*70)
    print("🧪 시스템 테스트 시작")
    print("="*70)
    
    bot.ask("knightfury", "총 사용자 수는 몇 명인가요?")
