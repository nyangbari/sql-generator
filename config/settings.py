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

# Model settings
MODEL_CONFIG = {
    'model_id': 'defog/sqlcoder-7b-2',
    'device_map': 'auto',
    'load_in_8bit': True,
    'max_new_tokens': 300,
    'temperature': 0.1,
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
