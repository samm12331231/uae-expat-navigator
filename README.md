# UAE Expat Navigator

UAE Expat Navigator is a FastAPI + LangChain assistant that answers UAE government service questions using official documents (PDFs) and optional helper tools for eligibility and fee calculations. It provides a simple web UI and a JSON API.

## Features
- Retrieval‑augmented answers grounded in official UAE documents
- Eligibility checks for driving‑license exchange by country
- Fee calculation from document fee tables
- FastAPI JSON API with a lightweight HTML frontend

## Project Structure
```
backend/         FastAPI app, agent, tools, ingestion
frontend/        Static HTML UI
data/pdfs/       Source official PDFs
data/chroma_db/  Vector store (created after ingestion)
requirements.txt Python dependencies
```

## Prerequisites
- Python 3.10+
- API keys for:
  - OpenAI (embeddings)
  - Groq (LLM)

## Setup
1. Create and activate a virtual environment (recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the repo root with the required keys:
   ```bash
   OPENAI_API_KEY=...
   GROQ_API_KEY=...
   ```
4. Run the commands below from the repository root so `backend.*` modules resolve correctly.

Optional environment variables:
```
CHROMA_PATH=./data/chroma_db
SOURCE_DIR=./data/pdfs
PORT=8000
```

## Ingest Documents
Place official PDFs in `data/pdfs/`, then run ingestion to build the vector store:
```bash
python -m backend.ignest
```
This clears any existing Chroma DB at `CHROMA_PATH` and rebuilds it.

## Run the API
```bash
python -m backend.main
```
Or with auto-reload:
```bash
uvicorn backend.main:app --reload
```
The backend serves the frontend at `http://localhost:8000/` by default.
API endpoints:
- `POST /ask` — body: `{ "question": "..." }`
- `GET /health`

Example:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"How do I convert my UK driving license?"}'
```

## Run the Frontend (Optional)
If you prefer hosting the static UI separately, serve it and open it in your browser:
```bash
cd frontend
python -m http.server 3000
```
Then visit: `http://localhost:3000`

## Testing
The repo includes a simple retrieval test script:
```bash
python -m backend.test_retrieval
```
This requires valid API keys and an ingested vector store.

## Notes
- Answers are restricted to information found in the official documents.
- Fee and eligibility tools are used only when relevant to the question.
