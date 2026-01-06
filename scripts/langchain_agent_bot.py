#!/usr/bin/env python3
# langchain_agent_bot.py
# LangChain Agent + vLLM + Read-Only

import os
from dotenv import load_dotenv
from vllm import LLM as VLLM_Model, SamplingParams
from langchain.llms.base import LLM
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_types import AgentType
from typing import Any, List, Optional, Mapping
from pydantic import Field

load_dotenv()

class ReadOnlySQLDatabase(SQLDatabase):
    """Read-Only SQL Database"""
    
    WRITE_KEYWORDS = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 
                     'ALTER', 'TRUNCATE', 'REPLACE', 'MERGE']
    
    def run(self, command: str, fetch: str = "all", **kwargs):
        sql_upper = command.upper().strip()
        
        for keyword in self.WRITE_KEYWORDS:
            if keyword in sql_upper:
                raise ValueError(f"🚫 {keyword} 차단됨! SELECT만 허용")
        
        if not (sql_upper.startswith('SELECT') or sql_upper.startswith('SHOW') or sql_upper.startswith('DESCRIBE')):
            raise ValueError("🚫 SELECT/SHOW/DESCRIBE만 허용")
        
        print(f"✅ 안전한 쿼리")
        return super().run(command, fetch=fetch, **kwargs)

class VLLMWrapper(LLM):
    """vLLM을 LangChain LLM으로 래핑"""
    
    vllm_model: Any = Field(default=None, exclude=True)
    sampling_params: Any = Field(default=None, exclude=True)
    
    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        
        print("🔄 vLLM 로딩...")
        
        self.vllm_model = VLLM_Model(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            dtype="float16"
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=250,
            stop=["\n\n\n", "Observation:"]
        )
        
        print("✅ vLLM 로드 완료!")
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """LangChain이 호출하는 메서드"""
        
        outputs = self.vllm_model.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text
    
    @property
    def _llm_type(self) -> str:
        return "vllm"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": "vllm-sql-generator"}

class LangChainAgentBot:
    def __init__(self, model_path):
        """LangChain Agent Bot"""
        
        print("="*70)
        print("🤖 LangChain Agent SQL Bot (vLLM + Read-Only)")
        print("="*70)
        
        # vLLM을 LangChain LLM으로 래핑
        self.llm = VLLMWrapper(model_path=model_path)
        
        # DB 설정
        knightfury_uri = os.getenv("KNIGHTFURY_DB_URI")
        furyx_uri = os.getenv("FURYX_DB_URI")
        
        self.databases = {}
        if knightfury_uri:
            self.databases["knightfury"] = knightfury_uri.replace("mysql://", "mysql+pymysql://")
        if furyx_uri:
            self.databases["furyx"] = furyx_uri.replace("mysql://", "mysql+pymysql://")
        
        print("\n📚 프로젝트:")
        for project in self.databases.keys():
            print(f"  ✅ {project}")
        
        self.agents = {}
        
        print("\n✅ 준비 완료!")
        print("🔒 보안: SELECT/SHOW/DESCRIBE만 허용")
        print("="*70)
    
    def get_agent(self, project):
        """프로젝트별 LangChain Agent 생성"""
        
        project = project.lower()
        
        if project not in self.agents:
            uri = self.databases.get(project)
            
            if not uri:
                raise ValueError(f"프로젝트 '{project}' 없음")
            
            print(f"\n🔗 {project} DB 연결 중...")
            
            # Read-Only DB
            db = ReadOnlySQLDatabase.from_uri(uri)
            
            # 테이블 확인
            tables = db.get_usable_table_names()
            print(f"📊 테이블: {len(tables)}개")
            
            # LangChain Agent 생성
            agent = create_sql_agent(
                llm=self.llm,
                db=db,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5,
                max_execution_time=60
            )
            
            self.agents[project] = agent
            print(f"✅ Agent 생성 완료!")
        
        return self.agents[project]
    
    def ask(self, project, question):
        """LangChain Agent로 질문"""
        
        print("\n" + "="*70)
        print(f"📂 프로젝트: {project}")
        print(f"💬 질문: {question}")
        print("="*70)
        
        try:
            agent = self.get_agent(project)
            
            print("\n🤔 Agent 실행 중...\n")
            
            # Agent 실행
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
            print(f"\n❌ 오류: {e}")
            print("="*70)
            return None
    
    def list_tables(self, project):
        """테이블 목록"""
        
        project = project.lower()
        uri = self.databases.get(project)
        
        if not uri:
            print("❌ 프로젝트 없음")
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
        
        print(f"\n🎯 대화형 모드 (프로젝트: {project})")
        print("\n명령어: 'tables', 'exit'")
        print("="*70)
        
        while True:
            try:
                user_input = input(f"\n[{project}] 질문> ")
                
                if not user_input.strip():
                    continue
                
                cmd = user_input.lower().strip()
                
                if cmd in ['exit', 'quit', 'q']:
                    print("\n👋 종료!")
                    break
                
                elif cmd == 'tables':
                    self.list_tables(project)
                    continue
                
                self.ask(project, user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 종료!")
                break
            except Exception as e:
                print(f"❌ 오류: {e}")

# 실행
if __name__ == "__main__":
    bot = LangChainAgentBot("./models/sql-generator-spider-plus-company")
    
    # 테이블 목록
    bot.list_tables("knightfury")
    
    # 테스트
    print("\n" + "="*70)
    print("🧪 LangChain Agent 테스트")
    print("="*70)
    
    bot.ask("knightfury", "총 사용자 수는?")
    
    # 대화형 모드
    # bot.interactive("knightfury")
