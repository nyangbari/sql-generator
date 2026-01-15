#!/usr/bin/env python3
"""Modular SQLCoder Bot - DB-level queries"""

import sys
import os
import ast
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.utilities.sql_database import SQLDatabase
from config import DATABASE_CONFIG
from services import RAGService, SQLService, ValidationService, QueryPreprocessor

class ModularSQLBot:
    """모듈형 SQL 봇 - DB 레벨 쿼리"""
    
    def __init__(self):
        print("="*70)
        print("🚀 Modular SQLCoder Bot")
        print("="*70)
        
        self.rag = RAGService()
        self.sql = SQLService()
        self.validator = ValidationService()
        self.preprocessor = QueryPreprocessor()
        
        self.databases = {}
        self.db_types = {}
        
        for name, config in DATABASE_CONFIG.items():
            if config['uri']:
                self.databases[name] = config['uri']
                self.db_types[name] = self._detect_db_type(config['uri'])
        
        print(f"\n📚 데이터베이스:")
        for name in self.databases.keys():
            print(f"   - {name} ({self.db_types[name]})")
        
        print("\n🔄 초기화 중...")
        for name, uri in self.databases.items():
            print(f"\n   [{name}]")
            # DB 한 번만 연결해서 공유
            db = SQLDatabase.from_uri(uri, sample_rows_in_table_info=0)
            self.rag.build_index(name, db=db)
            self.preprocessor.build_entity_cache(name, db=db)
        
        print("\n✅ 완료!")
        print("="*70)
    
    def _detect_db_type(self, uri):
        """Detect database type from URI"""
        uri_lower = uri.lower()
        
        if 'mysql' in uri_lower or 'pymysql' in uri_lower:
            return "MySQL"
        elif 'postgres' in uri_lower or 'psycopg' in uri_lower:
            return "PostgreSQL"
        elif 'sqlite' in uri_lower:
            return "SQLite"
        elif 'mssql' in uri_lower or 'sqlserver' in uri_lower:
            return "SQL Server"
        else:
            return "MySQL"
    
    def ask(self, db_name, question):
        """질문 처리"""
        print("\n" + "="*70)
        print(f"📂 {db_name} ({self.db_types.get(db_name, 'Unknown')})")
        print(f"💬 {question}")
        print("="*70)
        
        uri = self.databases.get(db_name)
        db_type = self.db_types.get(db_name, "MySQL")
        
        if not uri:
            print("❌ DB 없음")
            return None
        
        try:
            # Step 0: Query preprocessing (optional project detection)
            print("\n🔍 Step 0: 질문 분석...")
            preprocessed = self.preprocessor.preprocess(db_name, question)
            
            if preprocessed['entities']:
                print(f"   발견된 엔티티:")
                for key, value in preprocessed['entities'].items():
                    if key == 'project':
                        print(f"      project: {value['projectId']} ({value.get('displayTeamName', 'N/A')})")
            
            if preprocessed['hints']:
                print(f"   SQL 힌트:")
                for hint in preprocessed['hints']:
                    print(f"      - {hint}")
            else:
                print(f"   전체 DB 조회 (프로젝트 필터 없음)")
            
            # Step 1: 테이블 선택 (Phi-3)
            print("\n🔍 Step 1: 테이블 선택 (Phi-3)...")
            available_tables = self.rag.table_cache.get(db_name, {})

            if not available_tables:
                print("❌ 테이블 정보 없음")
                return None

            selected_table_names = self.sql.select_tables(question, available_tables)

            if not selected_table_names:
                print("❌ 관련 테이블 없음")
                return None

            # 선택된 테이블의 스키마 가져오기
            tables = []
            for name in selected_table_names:
                if name in available_tables:
                    tables.append({
                        "name": name,
                        "schema": available_tables[name]["create_statement"]
                    })

            print(f"   선택된 테이블: {[t['name'] for t in tables]}")
            
            # Step 2: SQL generation
            print(f"\n🔄 Step 2: SQL 생성 ({db_type})...")
            sql = self.sql.generate(
                question, 
                tables, 
                hints=preprocessed.get('hints'),
                db_type=db_type
            )
            
            print(f"\n💾 생성된 SQL:")
            print(sql)

            # Step 3: Validation (보안 검증만 - 테이블 검증은 경고만)
            valid, error = self.validator.validate(sql)  # 보안 검증

            if not valid:
                print(f"\n{error}")
                return None

            # 테이블 검증 (경고만, 차단하지 않음)
            table_names = [t['name'] for t in tables]
            table_valid, table_error = self.validator.validate_tables(sql, table_names)

            if not table_valid:
                print(f"\n   ⚠️  {table_error}")
                print(f"   📋 Available tables: {table_names}")
                # Fallback for user count
                if 'fury_users' in table_names and len(table_names) == 1:
                    sql = "SELECT COUNT(*) FROM fury_users"
                    print(f"   🔧 Using fallback: {sql}")
            
            # Step 4: Execution
            print("\n🔄 Step 3: 실행...")
            db = SQLDatabase.from_uri(uri, sample_rows_in_table_info=0)
            result = db.run(sql)
            
            print(f"\n📊 결과:")

            formatted = self._format_result(result, sql, preprocessed.get('entities'))
            print(formatted)

            # Step 5: 자연어 답변 생성 (SQLCoder 재사용)
            print("\n🔄 Step 4: 자연어 답변 생성...")
            natural_answer = self.sql.generate_answer(question, result)

            print("\n" + "="*70)
            if natural_answer:
                print(f"💬 {natural_answer}")
            else:
                print(f"💡 {self._format_answer(result, sql, preprocessed.get('entities'))}")
            print("="*70)
            
            return {
                "tables": [t['name'] for t in tables],
                "sql": sql,
                "result": result,
                "formatted": formatted
            }
            
        except Exception as e:
            print(f"\n❌ {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _parse_result(self, result, sql):
        """결과 파싱 - 공통 로직

        Returns:
            dict: {
                'type': 'group_count' | 'count' | 'list' | 'raw',
                'data': parsed data,
                'count': single count value (for 'count' type),
                'total': total count (for 'group_count' type)
            }
        """
        if not result or result == "[]":
            return {'type': 'empty', 'data': None}

        sql_upper = sql.upper()

        # GROUP BY + COUNT
        if 'GROUP BY' in sql_upper and 'COUNT' in sql_upper:
            try:
                data = ast.literal_eval(result)
                total = sum(row[-1] if isinstance(row, tuple) else row for row in data)
                return {'type': 'group_count', 'data': data, 'total': total}
            except:
                pass

        # Simple COUNT
        if 'COUNT' in sql_upper and 'GROUP BY' not in sql_upper:
            matches = re.findall(r'\[\((\d+)[,\)]', str(result))
            if matches:
                return {'type': 'count', 'data': None, 'count': int(matches[0])}

        # List result
        if result.startswith('['):
            try:
                data = ast.literal_eval(result)
                return {'type': 'list', 'data': data}
            except:
                pass

        return {'type': 'raw', 'data': result}

    def _format_result(self, result, sql, entities=None):
        """결과 포맷팅 (상세)"""
        parsed = self._parse_result(result, sql)

        if parsed['type'] == 'empty':
            return "결과 없음"

        if parsed['type'] == 'group_count':
            data = parsed['data']
            if len(data) == 0:
                return "결과 없음"

            lines = [f"\n총 {len(data)}개 카테고리, {parsed['total']}개 미션:"]
            lines.append("-" * 60)

            for i, row in enumerate(data, 1):
                if isinstance(row, tuple) and len(row) >= 3:
                    lines.append(f"{i}. {row[0]} {row[1]}: {row[2]}개")
                else:
                    lines.append(f"{i}. {row}")

            return "\n".join(lines)

        if parsed['type'] == 'count':
            entity_name = ""
            if entities and 'project' in entities:
                entity_name = f" ({entities['project'].get('displayTeamName', '')})"
            return f"총 {parsed['count']}개{entity_name}"

        if parsed['type'] == 'list':
            data = parsed['data']
            if len(data) == 0:
                return "결과 없음"

            display_count = min(5, len(data))
            lines = [f"\n총 {len(data)}개 (처음 {display_count}개 표시):"]
            lines.append("-" * 60)

            for i, row in enumerate(data[:display_count], 1):
                lines.append(f"{i}. {row}")

            if len(data) > 5:
                lines.append(f"... (나머지 {len(data)-5}개)")

            return "\n".join(lines)

        return str(result)

    def _format_answer(self, result, sql, entities=None):
        """간단한 답변 (요약)"""
        parsed = self._parse_result(result, sql)

        prefix = ""
        if entities and 'project' in entities:
            name = entities['project'].get('displayTeamName') or entities['project'].get('projectName')
            prefix = f"'{name}': "

        if parsed['type'] == 'empty':
            return "결과 없음"

        if parsed['type'] == 'group_count':
            data = parsed['data']
            parts = []
            for row in data:
                if isinstance(row, tuple) and len(row) >= 3:
                    parts.append(f"{row[-1]}개 {row[0]} {row[1]}")

            return f"{prefix}{parsed['total']}개 미션 ({', '.join(parts)})"

        if parsed['type'] == 'count':
            return f"{prefix}{parsed['count']}개"

        if parsed['type'] == 'list':
            return f"{prefix}{len(parsed['data'])}개의 결과"

        return str(result)

if __name__ == "__main__":
    bot = ModularSQLBot()
    
    if len(sys.argv) > 2:
        bot.ask(sys.argv[1], sys.argv[2])
    else:
        dbs = list(bot.databases.keys())
        
        if not dbs:
            print("\n❌ DB 없음")
            sys.exit(1)
        
        print(f"\n📚 데이터베이스: {', '.join(dbs)}")
        db_name = input("선택: ").strip().lower()
        
        if db_name not in dbs:
            print(f"❌ '{db_name}' 없음")
            sys.exit(1)
        
        print(f"\n✅ '{db_name}' 선택")
        print(f"💡 Tip: 특정 프로젝트를 질문에 포함하면 해당 프로젝트만 조회됩니다")
        print(f"   예: 'SuperWalk 사용자 몇 명?' vs '전체 사용자 몇 명?'")
        print("\n💬 질문 입력 (종료: exit)")
        print("")
        
        while True:
            try:
                question = input(f"\n[{db_name}] ").strip()
                
                if question.lower() in ['exit', 'quit', 'q']:
                    print("\n👋")
                    break
                
                if question:
                    bot.ask(db_name, question)
                    
            except KeyboardInterrupt:
                print("\n\n👋")
                break
