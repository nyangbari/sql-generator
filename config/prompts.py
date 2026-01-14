"""Prompt templates"""

SQL_GENERATION_PROMPT_TEMPLATE = """### Task
Generate a {db_type} compatible SQL query to answer: {question}

### Database Type: {db_type}

### Database Schema
{schema}

### CRITICAL: Column Name Rules
- ONLY use column names that appear in the schema above
- Do NOT invent or guess column names
- If a column doesn't exist, use what's available or skip it
- Common mistakes to avoid:
  * "p.name" → Use "p.projectName" or "p.projectId"
  * "m.category1" → Use "m.missionCategory1"
  * Always check the CREATE TABLE statement for exact column names

### Query Instructions
- For "what kind/type": SELECT category1, category2, COUNT(*) ... GROUP BY
- For "how many": SELECT COUNT(*) ...
- For "when": SELECT startDate/endDate ... ORDER BY ... LIMIT 1
- For "list": SELECT * ... LIMIT 100
- ALWAYS filter by projectId when project specified
- Use projectId (NOT projectName) for filtering
- fury_users has NO projectId - use fury_user_project_missions for per-project users

### Examples
- "how many missions?" → SELECT COUNT(*) FROM fury_project_missions WHERE projectId = 'X'
- "what kind?" → SELECT missionCategory1, missionCategory2, COUNT(*) as count FROM fury_project_missions WHERE projectId = 'X' GROUP BY missionCategory1, missionCategory2
- "how many users in X?" → SELECT COUNT(DISTINCT userId) FROM fury_user_project_missions WHERE projectId = 'X'
- "how many total users?" → SELECT COUNT(*) FROM fury_users
- "when does X end?" → SELECT endDate FROM fury_project_weeks WHERE projectId = 'X' ORDER BY endDate DESC LIMIT 1

### Answer
{db_type} query:"""

ANSWER_PROMPT = """Question: {question}
SQL Result: {result}

Generate a natural language answer in Korean (1-2 sentences):
"""

PROMPT_VERSION = "3.4"
