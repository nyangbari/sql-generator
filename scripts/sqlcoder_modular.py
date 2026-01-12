#!/usr/bin/env python3
# sqlcoder_modular.py
# Modular SQLCoder Bot

import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.utilities.sql_database import SQLDatabase
from config import DATABASE_CONFIG, TABLE_DESCRIPTIONS
from services import RAGService, SQLService, ValidationService

class ModularSQLBot:
    """모듈형 SQL 봇"""
    
    def __init__(self):
        print("="*70)
        print("🚀 Modular SQLCoder Bot")
        print("="*70)
        
        # 서비스 초기화
        self.rag = RAGService()
        self.sql = SQLService()
        self.validator = ValidationService()
        
        # DB 설정
        self.databases = {
            name: config['uri'] 
            for name, config in DATABASE_CONFIG.items() 
            if config['uri']
        }
        
        print(f"\n📚 프로젝트: {', '.join(self.databases.keys())}")
        
        # RAG 인덱스 구축
        print("\n🔄 RAG 인덱스 생성 중...")
        for name, uri in self.databases.items():
            self.rag.build_index(name, uri)
        
        print("\n✅ 완료!")
        print("="*70)
    
    def ask(self, project, question):
        """질문 처리"""
        print("\n" + "="*70)
        print(f"📂 {project}")
        print(f"💬 {question}")
        print("="*70)
        
        uri = self.databases.get(project)
        if not uri:
            print("❌ 프로젝트 없음")
            return None
        
        try:
            # Step 1: RAG 검색
            print("\n🔍 Step 1: RAG 검색...")
            tables = self.rag.search(project, question)
            
            if not tables:
                print("❌ 관련 테이블 없음")
                return None
            
            print(f"   찾은 테이블: {[t['name'] for t in tables]}")
            
            # Step 2: SQL 생성
            print("\n🔄 Step 2: SQL 생성...")
            sql = self.sql.generate(question, tables)
            
            print(f"\n💾 생성된 SQL:")
            print(sql)
            
            # Step 3: 검증
            valid, error = self.validator.validate(sql)
            
            if not valid:
                print(f"\n{error}")
                return None
            
            # Step 4: 실행
            print("\n🔄 Step 3: 실행...")
            db = SQLDatabase.from_uri(uri, sample_rows_in_table_info=0)
            result = db.run(sql)
            
            print(f"\n📊 결과:")
            print(result)
            
            # Step 5: 답변 포맷팅
            answer = self._format_answer(result)
            
            print("\n" + "="*70)
            print(f"💡 {answer}")
            print("="*70)
            
            return {
                "tables": [t['name'] for t in tables],
                "sql": sql,
                "result": result,
                "answer": answer
            }
            
        except Exception as e:
            print(f"\n❌ {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _format_answer(self, result):
        """결과 포맷팅"""
        if not result or result == "[]":
            return "결과 없음"
        
        try:
            if '[(' in str(result):
                num = str(result).split('(')[1].split(',')[0].strip()
                if result.count('(') > 1:
                    return f"결과:\n{result}"
                else:
                    return f"{num}개"
            return str(result)
        except:
            return str(result)

if __name__ == "__main__":
    bot = ModularSQLBot()
    
    if len(sys.argv) > 2:
        bot.ask(sys.argv[1], sys.argv[2])
    else:
        projects = list(bot.databases.keys())
        
        if not projects:
            print("\n❌ 프로젝트 없음")
            sys.exit(1)
        
        print(f"\n📚 프로젝트: {', '.join(projects)}")
        project = input("선택: ").strip().lower()
        
        if project not in projects:
            print(f"❌ '{project}' 없음")
            sys.exit(1)
        
        print(f"\n✅ '{project}' 선택")
        print("\n💬 질문 입력 (종료: exit)")
        print("")
        
        while True:
            try:
                question = input(f"\n[{project}] ").strip()
                
                if question.lower() in ['exit', 'quit', 'q']:
                    print("\n👋")
                    break
                
                if question:
                    bot.ask(project, question)
                    
            except KeyboardInterrupt:
                print("\n\n👋")
                break
