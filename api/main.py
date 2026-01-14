"""FastAPI SQL Chatbot"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.sqlcoder_modular import ModularSQLBot

app = FastAPI(title="SQL Chatbot API", version="1.0")

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize bot (loads once at startup)
bot = None

@app.on_event("startup")
async def startup_event():
    global bot
    print("🚀 Initializing SQL Chatbot...")
    bot = ModularSQLBot()
    print("✅ Ready!")

class QueryRequest(BaseModel):
    project: str
    question: str

class QueryResponse(BaseModel):
    success: bool
    project: str
    question: str
    sql: Optional[str] = None
    result: Optional[str] = None
    formatted: Optional[str] = None
    error: Optional[str] = None

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "ok",
        "message": "SQL Chatbot API",
        "projects": list(bot.databases.keys()) if bot else []
    }

@app.get("/projects")
async def get_projects():
    """Get available projects"""
    if not bot:
        raise HTTPException(status_code=503, detail="Bot not initialized")
    
    return {
        "projects": [
            {"name": name, "db_type": bot.db_types[name]}
            for name in bot.databases.keys()
        ]
    }

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Ask a question"""
    if not bot:
        raise HTTPException(status_code=503, detail="Bot not initialized")
    
    if req.project not in bot.databases:
        raise HTTPException(
            status_code=404, 
            detail=f"Project '{req.project}' not found. Available: {list(bot.databases.keys())}"
        )
    
    try:
        result = bot.ask(req.project, req.question)
        
        if not result:
            return QueryResponse(
                success=False,
                project=req.project,
                question=req.question,
                error="Query failed - no results or error occurred"
            )
        
        return QueryResponse(
            success=True,
            project=req.project,
            question=req.question,
            sql=result.get("sql"),
            result=str(result.get("result")),
            formatted=result.get("formatted")
        )
        
    except Exception as e:
        return QueryResponse(
            success=False,
            project=req.project,
            question=req.question,
            error=str(e)
        )

@app.get("/health")
async def health():
    """Health check with details"""
    return {
        "status": "healthy",
        "bot_initialized": bot is not None,
        "projects_loaded": len(bot.databases) if bot else 0
    }
