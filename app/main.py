# app/main.py

# ---------- IMPORTS ----------
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas.models import ChatRequest, ChatResponse, UploadResponse
from app.chains.memory_chain import run_memory_chain
from app.ingestion.pdf_ingestor import ingest_pdf
import shutil
import os

# ---------- APP INSTANCE ----------
app = FastAPI(
    title="MediQuery API",
    description="Conversational Medical Document Assistant",
    version="1.0.0"
)

#------------ MIDDLEWARE ------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # React/Vite frontend
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- ROUTES ----------

@app.get("/")
async def root():
    return {"message": "MediQuery API is running"}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):

    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    # Save uploaded file temporarily
    file_path = f"data/uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Ingest into ChromaDB
    result = ingest_pdf(file_path)

    # Extract chunk count from result string
    chunks = int(result.split()[2])

    return UploadResponse(
        message="PDF ingested successfully",
        chunks=chunks
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    answer = run_memory_chain(
        question=request.message,
        session_id=request.session_id
    )

    return ChatResponse(
        session_id=request.session_id,
        reply=answer
    )