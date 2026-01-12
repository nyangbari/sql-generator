"""SQL Validation Service"""
from config.settings import SECURITY_CONFIG

class ValidationService:
    """SQL 검증 서비스"""
    
    def validate(self, sql):
        """SQL 보안 검증"""
        sql_upper = sql.upper()
        
        # 1. 허용된 작업인지
        if not any(op in sql_upper for op in SECURITY_CONFIG['allowed_operations']):
            return False, "⚠️  허용되지 않은 SQL 작업"
        
        # 2. 금지된 키워드
        for keyword in SECURITY_CONFIG['forbidden_keywords']:
            if keyword in sql_upper:
                return False, f"🚫 위험한 SQL: {keyword} 작업 차단"
        
        # 3. SELECT로 시작하는지
        if not sql_upper.strip().startswith('SELECT'):
            return False, "⚠️  SELECT로 시작해야 함"
        
        return True, None
