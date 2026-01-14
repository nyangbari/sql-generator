"""SQL Validation Service - Word boundary check"""
from config.settings import SECURITY_CONFIG
import re

class ValidationService:
    """SQL 검증 서비스"""
    
    def __init__(self):
        self.allowed_ops = SECURITY_CONFIG['allowed_operations']
        self.forbidden = SECURITY_CONFIG['forbidden_keywords']
    
    def validate(self, sql):
        """SQL 검증 - with word boundaries"""
        sql_upper = sql.upper()
        
        # Check allowed operations
        if not any(op in sql_upper for op in self.allowed_ops):
            return False, f"❌ Only {', '.join(self.allowed_ops)} allowed"
        
        # Check forbidden keywords with WORD BOUNDARIES
        # This prevents "updatedAt" from matching "UPDATE"
        for keyword in self.forbidden:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, sql_upper):
                return False, f"❌ '{keyword}' operation not allowed"
        
        return True, None
