"""Prompt templates"""

# SQL 생성 프롬프트
SQL_GENERATION_PROMPT = """### Task
Generate a SQL query to answer the following question: `{question}`

### Database Schema
{schema}

### Instructions
- Use COUNT(DISTINCT column) for unique counts
- Use projectId column (NOT projectName) for filtering projects
- Prefer main tables (fury_projects) over subset tables (fury_airdrop_projects)
- Add LIMIT 100 for SELECT * queries
- Use proper table aliases (short names like p, u, m)

### Answer
Given the database schema, here is the SQL query that answers `{question}`:
SELECT"""

# 답변 생성 프롬프트
ANSWER_PROMPT = """Question: {question}
SQL Result: {result}

Generate a natural language answer in Korean (1-2 sentences):
"""

# 프롬프트 버전
PROMPT_VERSION = "2.4"
