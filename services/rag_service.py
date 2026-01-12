"""RAG Service - Enhanced with better pattern matching"""
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sqlalchemy import inspect
from langchain_community.utilities.sql_database import SQLDatabase
from config.tables import TABLE_DESCRIPTIONS, TABLE_PRIORITY
from config.settings import RAG_CONFIG
from typing import List
import re

class DirectEmbeddings(Embeddings):
    """sentence-transformers 직접 사용"""
    
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text], convert_to_tensor=False)
        return embedding[0].tolist()

class RAGService:
    """RAG 검색 서비스 - 강화된 패턴 매칭"""
    
    def __init__(self):
        self.embeddings = DirectEmbeddings(RAG_CONFIG['embedding_model'])
        self.vector_stores = {}
        self.table_cache = {}
    
    def build_index(self, project_name, db_uri):
        """DB 스키마로 RAG 인덱스 구축"""
        try:
            db = SQLDatabase.from_uri(db_uri, sample_rows_in_table_info=0)
            inspector = inspect(db._engine)
            all_tables = db.get_usable_table_names()
            
            documents = []
            table_info = {}
            
            for table in all_tables:
                try:
                    columns = inspector.get_columns(table)
                    pk = inspector.get_pk_constraint(table)
                    pk_cols = pk.get('constrained_columns', [])
                    
                    col_names = [col['name'] for col in columns]
                    create_stmt = self._build_create_statement(table, columns, pk_cols)
                    description = self._get_table_description(table)
                    
                    search_text = f"""Table: {table}
Purpose: {description}
Columns: {', '.join(col_names)}
Use for queries about: {description}
Schema:
{create_stmt}"""
                    
                    doc = Document(
                        page_content=search_text,
                        metadata={
                            "table": table,
                            "columns": col_names,
                            "create_statement": create_stmt,
                            "description": description
                        }
                    )
                    
                    documents.append(doc)
                    table_info[table] = {
                        "columns": col_names,
                        "create_statement": create_stmt,
                        "description": description
                    }
                    
                except Exception as e:
                    print(f"      ⚠️  {table}: {e}")
            
            if documents:
                vector_store = FAISS.from_documents(documents, self.embeddings)
                self.vector_stores[project_name] = vector_store
                self.table_cache[project_name] = table_info
                print(f"      ✅ {len(documents)}개 테이블 인덱싱 완료!")
            else:
                print(f"      ⚠️  인덱싱할 테이블 없음")
            
        except Exception as e:
            print(f"      ❌ {project_name} RAG 구축 실패: {e}")
    
    def search(self, project_name, question, k=None):
        """질문에 관련된 테이블 검색 - 강화된 패턴 분석"""
        if k is None:
            k = RAG_CONFIG['k_results']
        
        if project_name not in self.vector_stores:
            return []
        
        # 1. 질문 정규화 및 분석
        analyzed = self._analyze_question(question)
        
        # 2. 우선순위 테이블 체크 (분석 결과 기반)
        priority_tables = self._check_priority_tables(question, analyzed)
        
        # 3. RAG 검색
        try:
            vector_store = self.vector_stores[project_name]
            docs = vector_store.similarity_search(question, k=k+2)
        except Exception as e:
            print(f"⚠️  RAG 검색 실패: {e}")
            docs = []
        
        tables = []
        
        # 4. 우선순위 테이블 먼저 추가
        for table_name in priority_tables:
            if table_name in self.table_cache[project_name]:
                info = self.table_cache[project_name][table_name]
                tables.append({
                    "name": table_name,
                    "schema": info["create_statement"]
                })
        
        # 5. RAG 결과 추가 (중복 제거)
        for doc in docs:
            table_name = doc.metadata["table"]
            if table_name not in [t['name'] for t in tables]:
                tables.append({
                    "name": table_name,
                    "schema": doc.metadata["create_statement"]
                })
            
            if len(tables) >= k:
                break
        
        return tables[:k]
    
    def _analyze_question(self, question):
        """질문 분석 및 패턴 추출"""
        question_lower = question.lower()
        
        analysis = {
            'has_project_name': bool(re.search(r'\b(2pic|project\s+\w+)\b', question_lower)),
            'asking_about_missions': any(w in question_lower for w in [
                'mission', 'quest', '미션', '퀘스트'
            ]),
            'asking_about_types': any(w in question_lower for w in [
                'what kind', 'what type', 'which type', 'type of', 'kind of',
                '어떤', '무슨', '종류'
            ]),
            'asking_for_list': any(w in question_lower for w in [
                'what', 'which', 'show', 'list', 'does', 'have',
                '뭐', '무엇', '어떤', '보여', '있어'
            ]),
            'asking_for_count': any(w in question_lower for w in [
                'how many', 'count', 'number of',
                '몇', '개수', '얼마나'
            ]),
            'mentions_specific': any(w in question_lower for w in [
                'specific', 'actual', 'real', 'concrete',
                '구체적', '실제', '진짜'
            ]),
        }
        
        return analysis
    
    def _check_priority_tables(self, question, analyzed=None):
        """질문 패턴에 따른 우선순위 테이블 반환 - 강화 버전
        
        🎯 핵심 원칙:
        1. 가장 구체적인 패턴부터 체크
        2. 질문 의도 파악 (리스트 vs 타입)
        3. 프로젝트명 포함 여부 확인
        """
        question_lower = question.lower()
        
        if analyzed is None:
            analyzed = self._analyze_question(question)
        
        # ============================================
        # 1단계: 가장 구체적 - 프로젝트의 미션 리스트
        # ============================================
        
        # 패턴 1: "프로젝트명 + 미션" + (what/which/show/list)
        if analyzed['has_project_name'] and analyzed['asking_about_missions']:
            # "what missions does 2pic have?"
            # "2pic 프로젝트는 어떤 미션을 해?"
            # "show missions for project X"
            if analyzed['asking_for_list'] or 'does' in question_lower or 'have' in question_lower:
                return TABLE_PRIORITY.get('project_missions', [])
        
        # 패턴 2: "어떤 미션" 키워드 (매우 명확한 신호)
        if any(kw in question_lower for kw in [
            'what missions does', 'which missions', 'missions for',
            'what kind of missions does', 'what missions', 'missions does',
            '어떤 미션', '무슨 미션', '미션 목록'
        ]):
            return TABLE_PRIORITY.get('project_missions', [])
        
        # ============================================
        # 2단계: 미션 타입/종류 (카테고리 질문)
        # ============================================
        
        if analyzed['asking_about_missions']:
            # "what types of missions exist?"
            # "what kind of missions are there?"
            if analyzed['asking_about_types'] and not analyzed['has_project_name']:
                return TABLE_PRIORITY.get('mission_types', [])
            
            # "platform missions", "dashboard missions"
            if 'platform' in question_lower or 'dashboard' in question_lower:
                return TABLE_PRIORITY.get('platform_missions', [])
            
            # "mission" + "project" (일반적)
            if 'project' in question_lower:
                return TABLE_PRIORITY.get('project_quests', [])
        
        # ============================================
        # 3단계: 프로젝트 관련 (일반적)
        # ============================================
        
        if 'project' in question_lower:
            if 'airdrop' in question_lower:
                return TABLE_PRIORITY.get('airdrop_count', [])
            return TABLE_PRIORITY.get('project_count', [])
        
        # ============================================
        # 4단계: 유저 관련
        # ============================================
        
        if 'user' in question_lower:
            return TABLE_PRIORITY.get('user_count', [])
        
        return []
    
    def _build_create_statement(self, table, columns, pk_cols):
        """CREATE TABLE 문 생성"""
        create_stmt = f"CREATE TABLE {table} (\n"
        col_defs = []
        
        for col in columns:
            col_type = str(col['type'])
            
            if 'INT' in col_type.upper():
                col_type = "INT"
            elif 'VARCHAR' in col_type.upper() or 'CHAR' in col_type.upper():
                col_type = "VARCHAR(100)"
            elif 'TEXT' in col_type.upper():
                col_type = "TEXT"
            elif 'DATE' in col_type.upper() or 'TIME' in col_type.upper():
                col_type = "DATETIME"
            elif 'DECIMAL' in col_type.upper() or 'NUMERIC' in col_type.upper():
                col_type = "DECIMAL"
            
            pk_marker = " PRIMARY KEY" if col['name'] in pk_cols else ""
            col_defs.append(f"    {col['name']} {col_type}{pk_marker}")
        
        create_stmt += ",\n".join(col_defs) + "\n)"
        return create_stmt
    
    def _get_table_description(self, table):
        """테이블 설명 가져오기"""
        if table in TABLE_DESCRIPTIONS:
            return TABLE_DESCRIPTIONS[table]['description'].strip()
        return f"Table containing {table.replace('fury_', '').replace('_', ' ')} related data"
