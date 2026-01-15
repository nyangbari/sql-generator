"""Database and system settings"""
import os
from dotenv import load_dotenv

load_dotenv()

# Database configurations
DATABASE_CONFIG = {
    'knightfury': {
        'uri': os.getenv('KNIGHTFURY_DB_URI', '').replace('mysql://', 'mysql+pymysql://'),
        'name': 'KnightFury',
        'description': 'KnightFury platform database'
    },
    'furyx': {
        'uri': os.getenv('FURYX_DB_URI', '').replace('mysql://', 'mysql+pymysql://'),
        'name': 'FuryX',
        'description': 'FuryX platform database'
    }
}

# Model settings - SQL 생성용
MODEL_CONFIG = {
    'model_id': 'defog/sqlcoder-7b-2',
    'device_map': 'auto',
    'load_in_8bit': True,
    'max_new_tokens': 500,
    'min_new_tokens': 50,
    'temperature': 0.5,
    'top_p': 0.95,
}

# 자연어 생성용 모델 (한국어 지원)
ANSWER_MODEL_CONFIG = {
    'model_id': 'Qwen/Qwen2-1.5B-Instruct',
    'device_map': 'auto',
    'load_in_8bit': False,  # 1.5B는 작아서 8bit 불필요
    'max_new_tokens': 150,
    'temperature': 0.7,
}

# RAG settings
RAG_CONFIG = {
    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
    'k_results': 3,
    'min_similarity': 0.5
}

# Security settings
SECURITY_CONFIG = {
    'allowed_operations': ['SELECT'],
    'forbidden_keywords': ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE'],
    'max_result_rows': 1000
}
