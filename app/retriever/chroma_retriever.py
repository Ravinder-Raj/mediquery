#app/retriever/chroma_retriever.py

#----------- IMPORTS ---------------------
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from app.config import settings

#------------ RETRIEVER FUNTION ----------
def get_retriever():

    # BLOCK 1: open existing ChromaDB
    emdeddings = NVIDIAEmbeddings(
        model = settings.embedding_model,
        api_key = settings.nvidia_api_key,
        base_url = settings.nvidia_base_url
    )

    vectorstore = Chroma(
        collection_name = settings.collection_name,
        persist_directory = settings.chroma_path,
        embedding_function = emdeddings
    )

    # BLOCK 2: Wrap as retriever and return
    return vectorstore.as_retriever(search_kwargs={"k" : 4})