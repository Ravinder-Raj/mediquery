# MediQuery 🏥
### Conversational Medical Document Assistant

A production-grade RAG (Retrieval-Augmented Generation) API that lets users upload medical PDFs and have context-aware conversations about them. Built with LangChain LCEL, NVIDIA NIM, ChromaDB, and FastAPI.

---

## What it does

- Upload any medical PDF (lab report, discharge summary, clinical guideline)
- Ask questions about the document in natural language
- Ask follow-up questions — the system **remembers conversation history**
- Each user session has isolated memory — no context leakage between users

---

## Tech Stack

| Technology | Purpose |
|---|---|
| FastAPI | REST API framework |
| LangChain LCEL | Chain composition (`prompt | llm | parser`) |
| NVIDIA NIM APIs | LLM (Llama 3.1) + Embeddings |
| ChromaDB | Vector database for document storage |
| Pydantic | Request/response validation |
| Uvicorn | ASGI server |

---

## Project Structure

```
mediquery/
├── app/
│   ├── main.py                  # FastAPI routes
│   ├── config.py                # Settings from .env
│   ├── chains/
│   │   ├── rag_chain.py         # Basic LCEL chain
│   │   └── memory_chain.py      # Memory-aware RAG chain
│   ├── ingestion/
│   │   └── pdf_ingestor.py      # PDF → chunks → ChromaDB
│   ├── retriever/
│   │   └── chroma_retriever.py  # ChromaDB retriever
│   ├── memory/
│   │   └── session_store.py     # Per-session memory store
│   └── schemas/
│       └── models.py            # Pydantic request/response models
├── data/uploads/                # Uploaded PDFs (gitignored)
├── vectorstore/                 # ChromaDB files (gitignored)
├── requirements.txt
├── .env.example                 # Environment variable template
└── .env                         # API keys (gitignored — never commit)
```

---

## Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/mediquery.git
cd mediquery
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
# Open .env and add your NVIDIA API key
```

### 5. Run the server
```bash
uvicorn app.main:app --reload
```

### 6. Open API docs
```
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### GET /
Health check — verify the server is running.

```bash
curl http://127.0.0.1:8000/
```

Response:
```json
{
  "message": "MediQuery API is running"
}
```

---

### POST /upload
Upload a medical PDF for ingestion into ChromaDB.

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@report.pdf"
```

Response:
```json
{
  "message": "PDF ingested successfully",
  "chunks": 83
}
```

---

### POST /chat
Send a message and get a context-aware answer.

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user_123", "message": "What is the platelet count?"}'
```

Response:
```json
{
  "session_id": "user_123",
  "reply": "The platelet count is 150,000 - 410,000 /cmm."
}
```

---

### Follow-up question (memory in action)
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user_123", "message": "Is that normal?"}'
```

Response:
```json
{
  "session_id": "user_123",
  "reply": "The platelet count is within normal limits."
}
```

---

## Key Concepts Demonstrated

- **LCEL chains** — `prompt | llm | StrOutputParser()` pipe composition
- **RAG pipeline** — PDF ingestion → vector search → LLM answer
- **Conversation memory** — session-based `ChatMessageHistory`
- **Smart retrieval** — combines chat history with current query for better context
- **Production structure** — modular, config-driven, zero hardcoded values
- **Pydantic validation** — automatic request/response validation via schemas
- **FastAPI auto-docs** — Swagger UI at `/docs` out of the box

---

## Environment Variables

Create a `.env` file in the project root with the following:

```
NVIDIA_API_KEY=your_key_here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
LLM_MODEL=meta/llama-3.1-8b-instruct
EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5
COLLECTION_NAME=mediquery
CHROMA_PATH=./vectorstore/chroma_db
```

---

## How it works — Architecture

```
User uploads PDF
      ↓
pdf_ingestor.py → splits into chunks → embeds with NVIDIA → saves to ChromaDB
      
User sends message
      ↓
session_store.py → loads chat history for session_id
      ↓
chroma_retriever.py → searches ChromaDB with (history + question)
      ↓
memory_chain.py → LCEL chain: prompt | llm | StrOutputParser()
      ↓
memory updated → answer returned via FastAPI
```

---

## Author

Built as part of a 30-day AI Engineering learning journey.  
Demonstrates Day 12 (LangChain LCEL) and Day 13-14 (RAG + Memory + FastAPI).