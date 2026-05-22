# app/retriever/pinecone_retriever.py

# ---------- IMPORTS ----------
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.config import settings
import os

# Set Pinecone API key in environment
os.environ["PINECONE_API_KEY"] = settings.pinecone_api_key



def get_retriever():

    # BLOCK 1: Connect to existing Pinecone index
    embeddings = NVIDIAEmbeddings(
        model=settings.embedding_model,
        api_key=settings.nvidia_api_key,
        base_url=settings.nvidia_base_url
    )

    vectorstore = PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=embeddings,
        pinecone_api_key=settings.pinecone_api_key
    )

    # BLOCK 2: Return retriever
    return vectorstore.as_retriever(search_kwargs={"k": 4})