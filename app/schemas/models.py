#app/schemas/models.py

#-------- IMPORTS --------------------
from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id:str
    message:str

class UploadResponse(BaseModel):
    message: str
    chunks: int

class ChatResponse(BaseModel):
    session_id: str
    reply: str