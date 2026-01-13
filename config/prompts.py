"""Prompt templates"""

# SQL 생성 프롬프트
SQL_GENERATION_PROMPT = """### Task
Generate a SQL query to answer: {question}

### Database Schema
{schema}

### Instructions
- Use COUNT(DISTINCT column) for unique counts
- IMPORTANT: Use projectId column for filtering (NOT projectName)
- Prefer main tables over subset tables
- Add LIMIT 100 for SELECT * queries
- Use proper table aliases

### Answer
"""

# 답변 생성 프롬프트
ANSWER_PROMPT = """Question: {question}
SQL Result: {result}

Generate a natural language answer in Korean (1-2 sentences):
"""

# 프롬프트 버전
PROMPT_VERSION = "2.5"
