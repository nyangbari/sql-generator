#!/usr/bin/env python3
import os
import torch
from typing import Any, List, Optional
from dotenv import load_dotenv
from pydantic import Field

# HuggingFace & PEFT (vLLM 대신 사용)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from langchain_huggingface import HuggingFacePipeline

# LangChain 관련
from langchain_core.language_models.llms import LLM
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent

load_dotenv()

class LangChainAgentBot:
    def __init__(self, model_path):
        print("🚀 모델 로딩 시작 (RTX 4060 Ti 최적화 모드)...")
        
        base_model_id = "codellama/CodeLlama-7b-Instruct-hf"
        
        # 1. 토크나이저 및 베이스 모델 로드 (8-bit 양자화로 메모리 확보)
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True  # 16GB VRAM에서 가장 안정적인 모드
        )
        
        # 2. LoRA 어댑터 결합
        print(f"📦 어댑터 결합 중: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload() # 속도 향상을 위한 병합
        
        # 3. LangChain용 파이프라인 생성
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.15,
            return_full_text=False
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        # 4. DB 연결
        uri = os.getenv("KNIGHTFURY_DB_URI", "").replace("mysql://", "mysql+pymysql://")
        self.db = SQLDatabase.from_uri(uri)
        print("✅ 시스템 준비 완료!")

    def ask(self, question):
        try:
            # 에이전트 생성
            agent = create_sql_agent(
                llm=self.llm,
                db=self.db,
                agent_type="zero-shot-react-description",
                verbose=True,
                handle_parsing_errors=True
            )
            print(f"\n🔍 질문: {question}")
            # 한글 답변 유도
            prompt = f"질문에 대해 SQL을 생성하고 실행하여 한글로 답변하세요: {question}"
            result = agent.invoke({"input": prompt})
            print(f"\n💡 결과: {result.get('output')}")
        except Exception as e:
            print(f"❌ 에러 발생: {e}")

if __name__ == "__main__":
    # 절대 경로
    MODEL_PATH = "/home/dongsucat1/ai/sql-generator/models/sql-generator-spider-plus-company"
    
    bot = LangChainAgentBot(MODEL_PATH)
    bot.ask("사용자 테이블에 등록된 전체 사용자 수는 몇 명인가요?")
