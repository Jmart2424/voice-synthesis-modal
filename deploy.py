"""
Simple FastAPI wrapper for HTTP access to Modal functions
Run locally with: modal serve deploy.py
Deploy with: modal deploy deploy.py
"""

import modal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

stub = modal.Stub("voice-synthesis-api")

# Import the synthesis function from modal_app
from modal_app import synthesize_speech, upload_voice_model

# FastAPI app
web_app = FastAPI()

class SynthesisRequest(BaseModel):
    text: str
    voice_id: Optional[str] = "default"

class SynthesisResponse(BaseModel):
    audio_base64: str
    generation_time: float
    voice_id: str

@web_app.post("/synthesize", response_model=SynthesisResponse)
async def synthesize_endpoint(request: SynthesisRequest):
    """HTTP endpoint for voice synthesis"""
    try:
        result = synthesize_speech.remote(
            text=request.text,
            voice_id=request.voice_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@web_app.get("/health")
async def health_check():
    return {"status": "healthy", "gpu": "A10G", "service": "voice-synthesis"}

@stub.function()
@modal.asgi_app()
def fastapi_app():
    return web_app
