"""Prompt templates - Extremely simplified"""

SQL_GENERATION_PROMPT_TEMPLATE = """### Task
Generate a {db_type} SQL query to answer: {question}

### Database Schema
{schema}

### RULES
1. ONLY use the tables shown above in "Database Schema"
2. NEVER join tables that are not both present in the schema
3. NEVER invent column names - only use columns from CREATE TABLE statements
4. If NO specific project is mentioned → query ALL data (no WHERE projectId clause)
5. ONLY use WHERE projectId = '...' if a specific project name is in the question

### Examples

When project is specified (e.g. "SuperWalk missions"):
  SELECT COUNT(*) FROM fury_project_missions WHERE projectId = 'superwalk'

When NO project is specified (e.g. "Which user did the most missions?"):
  SELECT address, COUNT(*) AS total FROM fury_user_project_missions GROUP BY address ORDER BY total DESC LIMIT 1

When counting all users:
  SELECT COUNT(*) FROM fury_users

### Your {db_type} query:"""

ANSWER_PROMPT = """Question: {question}
SQL Result: {result}

Generate a natural language answer in Korean (1-2 sentences):
"""

PROMPT_VERSION = "4.0"
