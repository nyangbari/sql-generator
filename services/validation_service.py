"""SQL Validation Service - Enhanced"""
from config.settings import SECURITY_CONFIG

class ValidationService:
    """SQL 검증 서비스"""
    
    def __init__(self):
        self.allowed_ops = SECURITY_CONFIG['allowed_operations']
        self.forbidden = SECURITY_CONFIG['forbidden_keywords']
        
        # PostgreSQL-specific syntax not allowed in MySQL
        self.mysql_forbidden = [
            'NULLS FIRST',
            'NULLS LAST',
            'OFFSET',  # Use LIMIT x, y instead
            'RETURNING',
            '::',  # Type casting
            'ILIKE',
        ]
    
    def validate(self, sql):
        """SQL 검증"""
        sql_upper = sql.upper()
        
        # Check allowed operations
        if not any(op in sql_upper for op in self.allowed_ops):
            return False, f"❌ Only {', '.join(self.allowed_ops)} allowed"
        
        # Check forbidden keywords
        for keyword in self.forbidden:
            if keyword in sql_upper:
                return False, f"❌ '{keyword}' not allowed"
        
        # Check MySQL compatibility
        for syntax in self.mysql_forbidden:
            if syntax in sql_upper:
                return False, f"❌ '{syntax}' is PostgreSQL syntax, not MySQL. Remove it!"
        
        return True, None
