"""Prompt templates"""

SQL_GENERATION_PROMPT_TEMPLATE = """### Task
Generate a {db_type} compatible SQL query to answer: {question}

### Database Type: {db_type}

### Database Schema (THESE ARE THE ONLY TABLES AVAILABLE!)
{schema}

### ⚠️ CRITICAL RULES - READ CAREFULLY ⚠️
1. You can ONLY use tables that appear in the "Database Schema" section above
2. If you see ONLY "CREATE TABLE fury_users" above:
   - You CANNOT use fury_user_project_missions
   - You CANNOT use fury_projects  
   - You CANNOT JOIN any other tables
   - ONLY write: SELECT COUNT(*) FROM fury_users
3. Do NOT invent tables, columns, or joins that don't exist in the schema
4. If a table is not in the schema above, it DOES NOT EXIST

### Column Name Rules
- ONLY use column names from the CREATE TABLE statements above
- Common mistakes to avoid:
  * "p.name" does not exist → Use "p.projectName"
  * "p.userId" does not exist → Check the actual column names

### Query Instructions Based on Available Tables

IF ONLY fury_users is in schema:
→ "how many users?" → SELECT COUNT(*) FROM fury_users (NO JOIN!)
→ "total users?" → SELECT COUNT(*) FROM fury_users (NO JOIN!)

IF fury_user_project_missions is in schema:
→ "users in project X?" → SELECT COUNT(DISTINCT address) FROM fury_user_project_missions WHERE projectId = 'X'

IF fury_project_missions is in schema:
→ "how many missions in X?" → SELECT COUNT(*) FROM fury_project_missions WHERE projectId = 'X'
→ "what kind of missions?" → SELECT missionCategory1, missionCategory2, COUNT(*) FROM fury_project_missions WHERE projectId = 'X' GROUP BY missionCategory1, missionCategory2

IF fury_project_weeks is in schema:
→ "when does X end?" → SELECT endDate FROM fury_project_weeks WHERE projectId = 'X' ORDER BY endDate DESC LIMIT 1

### Answer
Write ONLY a {db_type} query using the tables provided in the schema section above.
Do NOT use any tables not listed in the schema.

SQL query:"""

ANSWER_PROMPT = """Question: {question}
SQL Result: {result}

Generate a natural language answer in Korean (1-2 sentences):
"""

PROMPT_VERSION = "3.6"
