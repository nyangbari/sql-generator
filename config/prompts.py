"""Prompt templates"""

# SQL 생성 프롬프트
SQL_GENERATION_PROMPT = """### Task
Generate a SQL query to answer: {question}

### Database Schema
{schema}

### Critical Instructions
- Use COUNT(DISTINCT) for unique counts
- ALWAYS filter by projectId when a project is specified
- Use projectId column for filtering (NOT projectName or teamId)
- Add LIMIT 100 for SELECT * queries
- Use proper JOIN conditions with aliases

### Answer
SQL query:"""

ANSWER_PROMPT = """Question: {question}
SQL Result: {result}

Generate a natural language answer in Korean (1-2 sentences):
"""

PROMPT_VERSION = "2.6"
