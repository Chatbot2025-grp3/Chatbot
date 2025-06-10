from fastapi import FastAPI
from pydantic import BaseModel
from radicalization_agent_7 import RadicalizationAgent

app = FastAPI()
sessions = {}

class Message(BaseModel):
    session_id: str
    message: str
    lang: str
    region: str

@app.post("/chat")
def chat_with_bot(payload: Message):
    if payload.session_id not in sessions:
        agent = RadicalizationAgent()
        agent.set_language(payload.lang)
        agent.set_region(payload.region)
        sessions[payload.session_id] = agent

    agent = sessions[payload.session_id]
    reply = agent.get_response(payload.message)

    return {"reply": reply}
