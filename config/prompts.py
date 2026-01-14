"""Prompt templates"""

# SQL 생성 프롬프트
SQL_GENERATION_PROMPT = """### Task
Generate a SQL query to answer: {question}

### Database Schema
{schema}

### Critical Instructions
- For "what kind" or "what type" questions: Use SELECT DISTINCT to LIST the types, NOT COUNT(DISTINCT)
- For "how many" questions: Use COUNT(DISTINCT) or COUNT(*)
- ALWAYS filter by projectId when a project is specified
- Use projectId column for filtering (NOT projectName)
- Add LIMIT 100 for SELECT * queries
- Use proper JOIN conditions with aliases

### Examples
- "what missions does X have?" → COUNT(*) WHERE projectId = 'X'
- "what kind of missions?" → SELECT DISTINCT missionCategory1, missionCategory2 WHERE projectId = 'X'
- "what types?" → SELECT DISTINCT [category_column]
- "how many?" → COUNT(*)

### Answer
SQL query:"""

ANSWER_PROMPT = """Question: {question}
SQL Result: {result}

Generate a natural language answer in Korean (1-2 sentences):
"""

PROMPT_VERSION = "2.7"
