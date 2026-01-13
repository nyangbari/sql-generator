"""RAG Service - With DirectEmbeddings for latest sentence-transformers"""
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
        # Use DirectEmbeddings for compatibility!
        self.embeddings = DirectEmbeddings(RAG_CONFIG['embedding_model'])
        self.vector_stores = {}
        self.table_cache = {}
    
    def build_index(self, project_name, db_uri):
        """Build RAG index from DB schema"""
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
        """Search for relevant tables"""
        if k is None:
            k = RAG_CONFIG['k_results']
        
        if project_name not in self.vector_stores:
            return []
        
        # Check priority tables first
        priority_tables = self._check_priority_tables(question)
        
        # RAG search
        try:
            vector_store = self.vector_stores[project_name]
            docs = vector_store.similarity_search(question, k=k+2)
        except Exception as e:
            print(f"   ⚠️  RAG 검색 실패: {e}")
            docs = []
        
        tables = []
        
        # Add priority tables first
        for table_name in priority_tables:
            if table_name in self.table_cache[project_name]:
                info = self.table_cache[project_name][table_name]
                tables.append({
                    "name": table_name,
                    "schema": info["create_statement"]
                })
        
        # Add RAG results
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
        """Build CREATE TABLE statement"""
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
            col_defs.append(f"    {col['name']} {col_type}{pk_marker}")
        
        create_stmt += ",\n".join(col_defs) + "\n)"
        return create_stmt
    
    def _get_table_description(self, table):
        """Get table description"""
        if table in TABLE_DESCRIPTIONS:
            return TABLE_DESCRIPTIONS[table]['description'].strip()
        return f"Table for {table.replace('fury_', '').replace('_', ' ')}"
    
    def _check_priority_tables(self, question):
        """Check priority tables based on question pattern"""
        question_lower = question.lower()
        
        # Most specific: project missions
        if any(kw in question_lower for kw in ['어떤 미션', 'what mission', 'which quest', 'missions for', 'quests for', '퀘스트']):
            return TABLE_PRIORITY.get('project_missions', [])
        
        # Mission related
        if 'mission' in question_lower or '미션' in question_lower:
            if 'type' in question_lower or 'kind' in question_lower:
                return TABLE_PRIORITY.get('mission_types', [])
            if 'dashboard' in question_lower or 'platform' in question_lower:
                return TABLE_PRIORITY.get('platform_missions', [])
            if 'project' in question_lower:
                return TABLE_PRIORITY.get('project_quests', [])
        
        # Project related
        if 'project' in question_lower or '프로젝트' in question_lower:
            if 'airdrop' in question_lower:
                return TABLE_PRIORITY.get('airdrop_count', [])
            return TABLE_PRIORITY.get('project_count', [])
        
        # User related
        if 'user' in question_lower:
            return TABLE_PRIORITY.get('user_count', [])
        
        return []
