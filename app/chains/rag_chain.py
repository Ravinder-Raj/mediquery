#app/chains/rag_chain.py

#--------- IMPORTS ---------------------
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.config import settings
from app.retriever.chroma_retriever import get_retriever

def run_rag_chain(question: str) -> str:

    # BLOCK 1: Create LLM
    llm = ChatNVIDIA(
        model = settings.llm_model,
        api_key = settings.nvidia_api_key,
        base_url = settings.nvidia_base_url
    )

    #BLOCK 2: Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a medical document assistant. 
         Answer the question based only on the provided context.
         If the answer is not in the context, say 'I cannot find this information in the document.'
         Be precise and professional."""),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    # BLOCK 3: Build LCEL chain
    chain = prompt | llm | StrOutputParser()

    retriever = get_retriever()
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return answer