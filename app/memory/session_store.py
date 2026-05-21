#app/memory/session_store.py

#--------- IMPORT -------------------
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


# ---------- SESSION STORE ----------
# This dict lives at module level — persists as long as server is running
# Key = session_id, Value = chat history object
sessions: dict = {}

def get_memory(session_id: str) -> BaseChatMessageHistory:

    #if session doesn't exist -> create new empty history
    if session_id not in sessions:
        sessions[session_id] = ChatMessageHistory()

    #Retriun existing or newly created history
    return sessions[session_id]