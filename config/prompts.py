"""Prompt templates"""

# SQL 생성 프롬프트 템플릿
SQL_GENERATION_PROMPT_TEMPLATE = """### Task
Generate a {db_type} compatible SQL query to answer: {question}

### Database Type: {db_type}

### Database Schema
{schema}

### Query Instructions
- For "what kind/type": SELECT category1, category2, COUNT(*) ... GROUP BY category1, category2
- For "how many": SELECT COUNT(*) ...
- For "when does X end/start": SELECT endDate/startDate FROM table WHERE projectId = 'X' ORDER BY endDate DESC LIMIT 1
- For "list": SELECT * ... LIMIT 100
- ALWAYS filter by projectId when project specified
- Use projectId (NOT projectName) for filtering
- Look for date columns: createdAt, updatedAt, startDate, endDate, startedAt, endedAt
- missionSeq is part of PK - each row = unique mission

### Examples
- "how many missions?" → SELECT COUNT(*) FROM fury_project_missions WHERE projectId = 'X'
- "what kind of missions?" → SELECT missionCategory1, missionCategory2, COUNT(*) as count FROM fury_project_missions WHERE projectId = 'X' GROUP BY missionCategory1, missionCategory2 ORDER BY count DESC
- "when does campaign end?" → SELECT endDate FROM fury_project_weeks WHERE projectId = 'X' ORDER BY endDate DESC LIMIT 1
- "when did project start?" → SELECT startDate FROM fury_project_weeks WHERE projectId = 'X' ORDER BY startDate ASC LIMIT 1
- "list missions" → SELECT * FROM fury_project_missions WHERE projectId = 'X' LIMIT 100

### Answer
{db_type} query:"""

ANSWER_PROMPT = """Question: {question}
SQL Result: {result}

Generate a natural language answer in Korean (1-2 sentences):
"""

PROMPT_VERSION = "3.2"
