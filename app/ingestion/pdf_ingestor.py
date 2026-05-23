# app/ingestion/pdf_ingestor.py

# ---------- IMPORTS ----------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from app.config import settings
import logging
import os

os.environ["PINECONE_API_KEY"] = settings.pinecone_api_key

logger = logging.getLogger(__name__)


def ingest_pdf(file_path: str, session_id: str) -> str:

    # BLOCK 1: Load PDF
    logger.info(f"Loading PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    logger.info(f"Loaded {len(pages)} pages")

    # BLOCK 2: Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)
    logger.info(f"Split into {len(chunks)} chunks")

    # BLOCK 3: Create NVIDIA embeddings
    embeddings = NVIDIAEmbeddings(
        model=settings.embedding_model,
        api_key=settings.nvidia_api_key,
        base_url=settings.nvidia_base_url
    )

    # BLOCK 4: Save to Pinecone under session namespace
    logger.info(f"Saving to Pinecone | namespace: {session_id}")
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=settings.pinecone_index_name,
        pinecone_api_key=settings.pinecone_api_key,
        namespace=session_id  # ✅ each session isolated
    )
    logger.info(f"Ingestion complete | chunks: {len(chunks)}")

    return f"Successfully ingested {len(chunks)} chunks from PDF"


def delete_collection(session_id: str) -> None:
    logger.info(f"Deleting Pinecone namespace: {session_id}")
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)
    index.delete(delete_all=True, namespace=session_id)  # ✅ delete all vectors for this session
    logger.info(f"Namespace deleted: {session_id}")