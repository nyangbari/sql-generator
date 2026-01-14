"""Prompt templates"""

# SQL 생성 프롬프트
SQL_GENERATION_PROMPT = """### Task
Generate a SQL query to answer: {question}

### Database Schema
{schema}

### Critical Instructions
- For "what kind/type of missions" questions: Use GROUP BY with COUNT to show types AND their counts
- For "how many missions" questions: Use COUNT(*) to count total missions
- For "list missions" questions: Use SELECT * to show all mission details
- ALWAYS filter by projectId when a project is specified
- Use projectId column for filtering (NOT projectName)
- The missionSeq column is part of the primary key - each row is a unique mission
- Add LIMIT 100 for SELECT * queries

### Examples
- "how many missions does X have?" → SELECT COUNT(*) FROM fury_project_missions WHERE projectId = 'X'
- "what kind of missions does X have?" → SELECT missionCategory1, missionCategory2, COUNT(*) as count FROM fury_project_missions WHERE projectId = 'X' GROUP BY missionCategory1, missionCategory2
- "list missions for X" → SELECT * FROM fury_project_missions WHERE projectId = 'X'

### Answer
SQL query:"""

ANSWER_PROMPT = """Question: {question}
SQL Result: {result}

Generate a natural language answer in Korean (1-2 sentences):
"""

PROMPT_VERSION = "2.8"
