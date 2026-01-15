"""RAG Service - Simplified user count logic"""
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
    """Direct sentence-transformers wrapper for compatibility"""
    
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed single query"""
        embedding = self.model.encode([text], convert_to_tensor=False)
        return embedding[0].tolist()

class RAGService:
    """RAG search service"""
    
    def __init__(self):
        self.embeddings = DirectEmbeddings(RAG_CONFIG['embedding_model'])
        self.vector_stores = {}
        self.table_cache = {}
    
    def build_index(self, project_name, db=None, db_uri=None):
        """Build RAG index from DB schema

        Args:
            project_name: 프로젝트 이름
            db: SQLDatabase 객체 (우선 사용)
            db_uri: DB URI (db가 없으면 이걸로 연결)
        """
        try:
            if db is None:
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

                    # Get table comment
                    table_comment = ""
                    try:
                        comment_info = inspector.get_table_comment(table)
                        table_comment = comment_info.get('text', '') or ''
                    except:
                        pass

                    col_names = [col['name'] for col in columns]
                    create_stmt = self._build_create_statement(table, columns, pk_cols, table_comment)
                    description = self._get_table_description(table, table_comment)
                    
                    search_text = f"""Table: {table}
Purpose: {description}
Columns: {', '.join(col_names)}
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
            import traceback
            traceback.print_exc()
    
    def search(self, project_name, question, k=None):
        """Search for relevant tables (legacy - with priority)"""
        if k is None:
            k = RAG_CONFIG['k_results']

        if project_name not in self.vector_stores:
            return []

        # Check priority tables first
        priority_tables = self._check_priority_tables(question)

        # If priority tables found, use ONLY those
        if priority_tables:
            tables = []
            for table_name in priority_tables:
                if table_name in self.table_cache[project_name]:
                    info = self.table_cache[project_name][table_name]
                    tables.append({
                        "name": table_name,
                        "schema": info["create_statement"]
                    })

            if tables:
                return tables[:k]

        # Fallback: RAG search
        return self.get_candidates(project_name, question, k)

    def get_candidates(self, project_name, question, k=5):
        """유사도 + 컬럼명 매칭 기반 후보 테이블 검색

        Args:
            project_name: DB 이름
            question: 질문
            k: 반환할 후보 수 (기본 5개)

        Returns:
            list: 후보 테이블 정보 리스트
        """
        if project_name not in self.vector_stores:
            return []

        try:
            # 1. 임베딩 유사도 검색
            vector_store = self.vector_stores[project_name]
            docs = vector_store.similarity_search(question, k=k)

            candidates = []
            seen_tables = set()

            for doc in docs:
                table_name = doc.metadata["table"]
                seen_tables.add(table_name)
                candidates.append({
                    "name": table_name,
                    "schema": doc.metadata["create_statement"],
                    "description": doc.metadata.get("description", ""),
                    "columns": doc.metadata.get("columns", [])
                })

            # 2. 컬럼명 직접 매칭 (동적 검색)
            column_matched = self._find_tables_by_column(project_name, question)

            for table_name in column_matched:
                if table_name not in seen_tables:
                    info = self.table_cache[project_name][table_name]
                    candidates.append({
                        "name": table_name,
                        "schema": info["create_statement"],
                        "description": info.get("description", ""),
                        "columns": info.get("columns", []),
                        "column_matched": True  # 컬럼 매칭 플래그
                    })
                    seen_tables.add(table_name)
                    print(f"   🔗 컬럼 매칭으로 추가: {table_name}")
                else:
                    # 이미 있는 테이블도 플래그 추가
                    for c in candidates:
                        if c["name"] == table_name:
                            c["column_matched"] = True
                            print(f"   🔗 컬럼 매칭 확인: {table_name}")
                            break

            return candidates

        except Exception as e:
            print(f"   ⚠️  RAG 검색 실패: {e}")
            return []

    def _find_tables_by_column(self, project_name, question):
        """질문의 키워드로 컬럼명 동적 검색"""
        if project_name not in self.table_cache:
            return []

        # 불용어 제거
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'show', 'me', 'get', 'find', 'list', 'give', 'what', 'which',
                     'how', 'many', 'much', 'with', 'their', 'from', 'for', 'and',
                     'or', 'in', 'on', 'at', 'to', 'of', 'by', 'top', 'all'}

        # 질문에서 키워드 추출
        words = []
        for word in question.lower().replace("'", " ").replace('"', ' ').split():
            # 숫자 제거, 2글자 이상만
            clean = ''.join(c for c in word if c.isalpha())
            if len(clean) >= 3 and clean not in stopwords:
                words.append(clean)

        matched_tables = []

        for table, info in self.table_cache[project_name].items():
            for column in info['columns']:
                col_lower = column.lower()
                for word in words:
                    # 키워드가 컬럼명에 포함되어 있으면 매칭
                    if word in col_lower:
                        if table not in matched_tables:
                            matched_tables.append(table)
                        break

        return matched_tables
    
    def _build_create_statement(self, table, columns, pk_cols, table_comment=""):
        """Build CREATE TABLE statement with comments"""
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
            elif 'DECIMAL' in col_type.upper():
                col_type = "DECIMAL"

            pk_marker = " PRIMARY KEY" if col['name'] in pk_cols else ""

            # Add column comment if exists
            col_comment = col.get('comment', '') or ''
            comment_marker = f" -- {col_comment}" if col_comment else ""

            col_defs.append(f"    {col['name']} {col_type}{pk_marker}{comment_marker}")

        create_stmt += ",\n".join(col_defs) + "\n)"

        # Add table comment if exists
        if table_comment:
            create_stmt += f"\n-- Table comment: {table_comment}"

        return create_stmt
    
    def _get_table_description(self, table, table_comment=""):
        """Get table description (DB comment + tables.py fallback)"""
        parts = []

        # 1. DB table comment (priority)
        if table_comment:
            parts.append(table_comment)

        # 2. tables.py description (fallback/supplement)
        if table in TABLE_DESCRIPTIONS:
            parts.append(TABLE_DESCRIPTIONS[table]['description'].strip())

        if parts:
            return " | ".join(parts)

        return f"Table for {table.replace('fury_', '').replace('_', ' ')}"
    
    def _check_priority_tables(self, question):
        """Check priority tables - RETURNS EXACTLY WHAT'S NEEDED"""
        question_lower = question.lower()
        
        # Total user count - ONLY fury_users
        if any(kw in question_lower for kw in ['사용자', 'user']) and \
           any(kw in question_lower for kw in ['총', 'total', '몇 명', 'how many', '전체', 'all']):
            # Check if it's asking for project-specific
            if any(kw in question_lower for kw in ['project', '프로젝트', 'in ', '에서']) and \
               not any(kw in question_lower for kw in ['all project', '모든 프로젝트', 'total']):
                # Project-specific user count
                return ['fury_user_project_missions']
            else:
                # Total platform users - ONLY this table!
                return ['fury_users']
        
        # Date/campaign questions
        if any(kw in question_lower for kw in ['when', 'end', 'start', '언제', '종료', '시작', 'campaign', 'week']):
            return TABLE_PRIORITY.get('campaign_dates', [])
        
        # Mission type questions
        if ('mission' in question_lower or '미션' in question_lower) and \
           ('type' in question_lower or 'kind' in question_lower or '종류' in question_lower):
            return TABLE_PRIORITY.get('mission_types', [])
        
        # Specific mission questions
        if any(kw in question_lower for kw in ['어떤 미션', 'what mission', 'which quest', 'missions for', 'quests for', '퀘스트']):
            return TABLE_PRIORITY.get('project_missions', [])
        
        # Platform missions
        if ('mission' in question_lower or '미션' in question_lower) and \
           ('dashboard' in question_lower or 'platform' in question_lower):
            return TABLE_PRIORITY.get('platform_missions', [])
        
        # Project missions
        if ('mission' in question_lower or '미션' in question_lower) and \
           ('project' in question_lower or '프로젝트' in question_lower):
            return TABLE_PRIORITY.get('project_quests', [])
        
        # Project count
        if 'project' in question_lower or '프로젝트' in question_lower:
            if 'airdrop' in question_lower:
                return TABLE_PRIORITY.get('airdrop_count', [])
            return TABLE_PRIORITY.get('project_count', [])
        
        return []
