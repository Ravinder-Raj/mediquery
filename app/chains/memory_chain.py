# app/chains/memory_chain.py

# ---------- IMPORTS ----------
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from app.config import settings
from app.retriever.chroma_retriever import get_retriever
from app.memory.session_store import get_memory


def run_memory_chain(question: str, session_id: str) -> str:

    # BLOCK 1: Create LLM
    llm = ChatNVIDIA(
        model=settings.llm_model,
        api_key=settings.nvidia_api_key,
        base_url=settings.nvidia_base_url
    )

    # BLOCK 2: Prompt with chat history
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a medical document assistant.
         Answer based only on the provided context.
         If answer is not in context, say 'I cannot find this in the document.'
         Be precise and professional."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    # BLOCK 3: LCEL chain
    chain = prompt | llm | StrOutputParser()

    # BLOCK 4: Get memory for this session
    memory = get_memory(session_id)

    # BLOCK 5: Retrieve relevant chunks
   # BLOCK 5: Build search query using history context
    if memory.messages:
        # combine last question + current question for better retrieval
        last_human = [m for m in memory.messages if m.type == "human"]
        if last_human:
            search_query = f"{last_human[-1].content} {question}"
        else:
            search_query = question
    else:
        search_query = question

    retriever = get_retriever()
    docs = retriever.invoke(search_query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # BLOCK 6: Run chain
    answer = chain.invoke({
        "context": context,
        "chat_history": memory.messages,
        "question": question
    })

    # BLOCK 7: Save to memory
    memory.add_user_message(question)
    memory.add_ai_message(answer)

    return answer