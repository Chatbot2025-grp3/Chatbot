from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi import status
from pydantic import BaseModel
from radicalization_agent_7 import RadicalizationBot
import uvicorn

app = FastAPI()
sessions = {}

class Message(BaseModel):
    session_id: str
    message: str
    lang: str
    region: str

@app.post("/chat")
def chat_with_bot(payload: Message):
   
    try:
        if payload.session_id not in sessions:
            agent = RadicalizationBot(session_id=payload.session_id)
            agent.set_language(payload.lang)
            # Validate and set region
            if not agent.validate_region(payload.region):
                print(f"[API WARNING] Region '{payload.region}' not found, using default")
            sessions[payload.session_id] = agent

        agent = sessions[payload.session_id]
        agent.set_language(payload.lang)
        # Ensure the latest region is applied
        if not agent.validate_region(payload.region):
            print(f"[API WARNING] Updated region '{payload.region}' invalid, keeping previous")

        reply = agent.get_response(payload.message)
        return {"reply": reply}
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
