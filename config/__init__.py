"""Configuration package"""
from .settings import DATABASE_CONFIG
from .tables import TABLE_DESCRIPTIONS, TABLE_PRIORITY
from .prompts import SQL_GENERATION_PROMPT, ANSWER_PROMPT

__all__ = [
    'DATABASE_CONFIG',
    'TABLE_DESCRIPTIONS',
    'TABLE_PRIORITY',
    'SQL_GENERATION_PROMPT',
    'ANSWER_PROMPT'
]
