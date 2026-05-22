# app/ingestion/pdf_ingestor.py

# ---------- IMPORTS ----------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.config import settings
import os
os.environ["PINECONE_API_KEY"] = settings.pinecone_api_key

def ingest_pdf(file_path: str) -> str:

    # BLOCK 1: Load PDF
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # BLOCK 2: Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)

    # BLOCK 3: Create NVIDIA embeddings
    embeddings = NVIDIAEmbeddings(
        model=settings.embedding_model,
        api_key=settings.nvidia_api_key,
        base_url=settings.nvidia_base_url
    )

    # BLOCK 4: Save to Pinecone
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=settings.pinecone_index_name,
        pinecone_api_key=settings.pinecone_api_key
    )

    return f"Successfully ingested {len(chunks)} chunks from PDF"