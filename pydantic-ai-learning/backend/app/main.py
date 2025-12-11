from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from temporalio.client import Client
from pydantic_ai.durable_exec.temporal import PydanticAIPlugin

from .lessons.lesson1_temporal_basics import start_lesson1_workflow
from .lessons.lesson2_deep_research import start_lesson2_workflow
from .lessons.lesson3_human_loop import start_lesson3_workflow
from .settings import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connect to Temporal Client on startup
    try:
        app.state.temporal_client = await Client.connect(
            settings.temporal_host,
            plugins=[PydanticAIPlugin()]
        )
        print("Connected to Temporal.")
    except Exception as e:
        print(f"Failed to connect to Temporal: {e}")
        # In production, you might want to fail hard here
        app.state.temporal_client = None
    
    yield
    
    # Cleanup
    # Client doesn't strictly need close(), but good practice if wrapper exists

app = FastAPI(title="PydanticAI Learning Hub", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

def get_client(app):
    if not app.state.temporal_client:
        raise HTTPException(status_code=503, detail="Temporal service unavailable")
    return app.state.temporal_client

@app.post("/api/lesson1")
async def lesson1_endpoint(request: QueryRequest):
    try:
        response = await start_lesson1_workflow(get_client(app), request.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/lesson2")
async def lesson2_endpoint(request: QueryRequest):
    try:
        response = await start_lesson2_workflow(get_client(app), request.query)
        return {"response": str(response)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/lesson3")
async def lesson3_endpoint(request: QueryRequest):
    try:
        response = await start_lesson3_workflow(get_client(app), request.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount frontend wrapper
# Must be after API routes
from fastapi.staticfiles import StaticFiles
import os

# Check if we are running from root or backend
if os.path.exists("frontend"):
    static_dir = "frontend"
elif os.path.exists("../frontend"):
    static_dir = "../frontend"
else:
    # Fallback/Error case, though we expect to run from root
    static_dir = "frontend"

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
