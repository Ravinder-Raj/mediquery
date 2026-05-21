#app/ingestion/pdf_ingestor.py

#-----IMPORTS ----------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import Chroma
from app.config import settings

#----- INGEST FUNCTION --------------
def ingest_pdf(file_path: str) ->str:

    #BLOCK 1: Load PDF 
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    #BLOCK 2: split pages into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    chunks = splitter.split_documents(pages)

    #BLOCK 3: Create embeddings model using NVIDIA
    embeddings = NVIDIAEmbeddings(
        model = settings.embedding_model,
        api_key = settings.nvidia_api_key,
        base_url = settings.nvidia_base_url
    )

    #BLOCK 4: Save chunks + embeddings to chroma
    Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        collection_name = settings.collection_name,
        persist_directory = settings.chroma_path 
    )

    return f"Successfully ingested {len(chunks)} chunks from PDF"


