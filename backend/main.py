import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from backend.agent import get_agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialise once at startup
    app.state.agent = get_agent()
    yield


app = FastAPI(title="UAE Expat Navigator", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list
    tool_calls: dict


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    try:
        # Run the synchronous agent in a thread pool so we don't block the event loop
        result = await asyncio.to_thread(app.state.agent, req.question)
        return {
            "answer": result["answer"],
            "sources": result["source_documents"],
            "tool_calls": result.get("tool_calls", {}),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
