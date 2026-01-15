"""SQL Validation Service - Word boundary check + Table validation"""
from config.settings import SECURITY_CONFIG
import re

class ValidationService:
    """SQL 검증 서비스"""

    def __init__(self):
        self.allowed_ops = SECURITY_CONFIG['allowed_operations']
        self.forbidden = SECURITY_CONFIG['forbidden_keywords']

    def validate(self, sql, allowed_tables=None):
        """SQL 검증 - 보안 + 테이블 검증

        Args:
            sql: 검증할 SQL 쿼리
            allowed_tables: 허용된 테이블 목록 (None이면 테이블 검증 스킵)

        Returns:
            (bool, str): (유효 여부, 에러 메시지)
        """
        sql_upper = sql.upper()

        # 1. Check allowed operations
        if not any(op in sql_upper for op in self.allowed_ops):
            return False, f"❌ Only {', '.join(self.allowed_ops)} allowed"

        # 2. Check forbidden keywords with WORD BOUNDARIES
        for keyword in self.forbidden:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, sql_upper):
                return False, f"❌ '{keyword}' operation not allowed"

        # 3. Check table usage (if allowed_tables provided)
        if allowed_tables:
            is_valid, error = self._validate_tables(sql, allowed_tables)
            if not is_valid:
                return False, error

        return True, None

    def _validate_tables(self, sql, allowed_tables):
        """SQL이 허용된 테이블만 사용하는지 검증"""
        # Extract table names from SQL (after FROM and JOIN)
        pattern = r'(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        used_tables = re.findall(pattern, sql, re.IGNORECASE)

        allowed_upper = [t.upper() for t in allowed_tables]

        for table in used_tables:
            if table.upper() not in allowed_upper:
                return False, f"❌ Table '{table}' not in schema"

        return True, None
