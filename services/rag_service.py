"""RAG (Retrieval-Augmented Generation) Service"""
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sqlalchemy import inspect
from langchain_community.utilities.sql_database import SQLDatabase
from config.tables import TABLE_DESCRIPTIONS, TABLE_PRIORITY
from config.settings import RAG_CONFIG
from typing import List

class DirectEmbeddings(Embeddings):
    """sentence-transformers 직접 사용 (최신 버전 호환)"""
    
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """여러 문서 임베딩"""
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """단일 쿼리 임베딩"""
        embedding = self.model.encode([text], convert_to_tensor=False)
        return embedding[0].tolist()

class RAGService:
    """RAG 검색 서비스"""
    
    def __init__(self):
        # 직접 sentence-transformers 사용 (최신 버전 호환!)
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
                    
                    # CREATE TABLE 문
                    create_stmt = self._build_create_statement(table, columns, pk_cols)
                    
                    # 설명 가져오기
                    description = self._get_table_description(table)
                    
                    # 검색용 텍스트
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
            
            # 벡터 스토어 생성
            if documents:
                vector_store = FAISS.from_documents(documents, self.embeddings)
                
                self.vector_stores[project_name] = vector_store
                self.table_cache[project_name] = table_info
                
                print(f"      ✅ {len(documents)}개 테이블 인덱싱 완료!")
            else:
                print(f"      ⚠️  인덱싱할 테이블 없음")
            
        except Exception as e:
            print(f"      ❌ {project_name} RAG 구축 실패: {e}")
            import traceback
            traceback.print_exc()
    
    def search(self, project_name, question, k=None):
        """질문에 관련된 테이블 검색"""
        if k is None:
            k = RAG_CONFIG['k_results']
        
        if project_name not in self.vector_stores:
            return []
        
        # 우선순위 테이블 체크 (가장 구체적 → 일반적 순서!)
        priority_tables = self._check_priority_tables(question)
        
        # RAG 검색
        try:
            vector_store = self.vector_stores[project_name]
            docs = vector_store.similarity_search(question, k=k+2)
        except Exception as e:
            print(f"⚠️  RAG 검색 실패: {e}")
            docs = []
        
        tables = []
        
        # 우선순위 테이블 먼저 추가
        for table_name in priority_tables:
            if table_name in self.table_cache[project_name]:
                info = self.table_cache[project_name][table_name]
                tables.append({
                    "name": table_name,
                    "schema": info["create_statement"]
                })
        
        # RAG 결과 추가 (중복 제거)
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
    
    def _check_priority_tables(self, question):
        """질문 패턴에 따른 우선순위 테이블 반환
        
        🎯 중요: 구체적인 패턴부터 체크!
        """
        question_lower = question.lower()
        
        # 1. 가장 구체적: 특정 프로젝트의 미션
        if any(kw in question_lower for kw in ['어떤 미션', 'what mission', 'which quest', 'missions for', 'quests for']):
            return TABLE_PRIORITY.get('project_missions', [])
        
        # 2. 미션 관련
        if 'mission' in question_lower:
            if 'type' in question_lower or 'kind' in question_lower or 'category' in question_lower:
                return TABLE_PRIORITY.get('mission_types', [])
            if 'dashboard' in question_lower or 'platform' in question_lower:
                return TABLE_PRIORITY.get('platform_missions', [])
            if 'project' in question_lower or 'quest' in question_lower:
                return TABLE_PRIORITY.get('project_quests', [])
        
        # 3. 프로젝트 관련
        if 'project' in question_lower:
            if 'airdrop' in question_lower:
                return TABLE_PRIORITY.get('airdrop_count', [])
            return TABLE_PRIORITY.get('project_count', [])
        
        # 4. 유저 관련
        if 'user' in question_lower:
            return TABLE_PRIORITY.get('user_count', [])
        
        return []
