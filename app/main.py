# app/main.py

# ---------- IMPORTS ----------
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas.models import ChatRequest, ChatResponse, UploadResponse
from app.chains.memory_chain import run_memory_chain
from app.ingestion.pdf_ingestor import ingest_pdf
from dotenv import load_dotenv
import shutil
import os
import logging
import traceback

# ---------- LOAD ENV ----------
load_dotenv()

# ---------- LOGGER ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ---------- CORS ORIGINS FROM ENV ----------
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
logger.info(f"Allowed origins: {ALLOWED_ORIGINS}")

# ---------- APP INSTANCE ----------
app = FastAPI(
    title="MediQuery API",
    description="Conversational Medical Document Assistant",
    version="1.0.0"
)

# ------------ MIDDLEWARE ------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- ROUTES ----------

@app.get("/")
async def root():
    logger.info("Health check hit")
    return {"message": "MediQuery API is running"}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...), session_id: str = ""):
    file_path = None
    logger.info(f"Upload request received | filename: {file.filename} | session: {session_id}")
    try:
        if not file.filename.endswith(".pdf"):
            logger.warning(f"Invalid file type: {file.filename}")
            raise HTTPException(status_code=400, detail="Only PDF files allowed")

        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")

        os.makedirs("data/uploads", exist_ok=True)

        file_path = f"data/uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved: {file_path}")

        # ✅ pass session_id to ingest
        result = ingest_pdf(file_path, session_id)
        logger.info(f"Ingestion result: {result}")

        os.remove(file_path)
        logger.info(f"Temp file deleted: {file_path}")

        chunks = int(result.split()[2])
        logger.info(f"Upload successful | chunks: {chunks}")

        return UploadResponse(message="PDF ingested successfully", chunks=chunks)

    except HTTPException:
        raise

    except Exception as e:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up file after error: {file_path}")
        logger.error(f"Upload failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    logger.info(f"Chat request | session: {request.session_id} | message: {request.message}")
    try:
        answer = run_memory_chain(
            question=request.message,
            session_id=request.session_id
        )
        logger.info(f"Chat response generated | session: {request.session_id}")
        return ChatResponse(
            session_id=request.session_id,
            reply=answer
        )

    except Exception as e:
        logger.error(f"Chat failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))