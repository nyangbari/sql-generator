"""Prompt templates - Extremely simplified"""

SQL_GENERATION_PROMPT_TEMPLATE = """### Task
Generate a {db_type} SQL query to answer: {question}

### Database Schema
{schema}

### RULES
1. ONLY use the tables shown above in "Database Schema"
2. NEVER join tables that are not both present in the schema
3. NEVER invent column names - only use columns from CREATE TABLE statements

### How to count users
- If ONLY fury_users table is available:
  Query: SELECT COUNT(*) FROM fury_users
  
- If fury_user_project_missions table is available with projectId 'X':
  Query: SELECT COUNT(DISTINCT address) FROM fury_user_project_missions WHERE projectId = 'X'

### Other common queries
- Count missions: SELECT COUNT(*) FROM fury_project_missions WHERE projectId = 'X'
- Mission types: SELECT missionCategory1, missionCategory2, COUNT(*) FROM fury_project_missions WHERE projectId = 'X' GROUP BY missionCategory1, missionCategory2
- Campaign end date: SELECT endDate FROM fury_project_weeks WHERE projectId = 'X' ORDER BY endDate DESC LIMIT 1

### Your {db_type} query:"""

ANSWER_PROMPT = """Question: {question}
SQL Result: {result}

Generate a natural language answer in Korean (1-2 sentences):
"""

PROMPT_VERSION = "4.0"
