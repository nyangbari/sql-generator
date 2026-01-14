"""Prompt templates"""

# SQL 생성 프롬프트
SQL_GENERATION_PROMPT = """### Task
Generate a MySQL-compatible SQL query to answer: {question}

### Database Schema
{schema}

### Critical Instructions - MySQL Syntax Only!
- Use MySQL syntax (NOT PostgreSQL!)
- Do NOT use: NULLS FIRST, NULLS LAST, LIMIT x OFFSET y
- Use MySQL LIMIT syntax: LIMIT offset, count
- For "what kind/type of missions": SELECT category1, category2, COUNT(*) as count ... GROUP BY category1, category2
- For "how many missions": SELECT COUNT(*) FROM table WHERE ...
- For "list missions": SELECT * FROM table WHERE ...
- ALWAYS filter by projectId when a project is specified
- Use projectId column for filtering (NOT projectName)
- missionSeq is part of primary key - each row = unique mission
- Add LIMIT 100 for SELECT *

### Examples
- "how many missions?" → SELECT COUNT(*) FROM fury_project_missions WHERE projectId = 'X'
- "what kind of missions?" → SELECT missionCategory1, missionCategory2, COUNT(*) as count FROM fury_project_missions WHERE projectId = 'X' GROUP BY missionCategory1, missionCategory2
- "list missions" → SELECT * FROM fury_project_missions WHERE projectId = 'X' LIMIT 100

### Answer
SQL query:"""

ANSWER_PROMPT = """Question: {question}
SQL Result: {result}

Generate a natural language answer in Korean (1-2 sentences):
"""

PROMPT_VERSION = "2.9"
