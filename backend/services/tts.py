import os
import httpx
from typing import Optional


ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"


async def text_to_speech(text: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM") -> Optional[bytes]:
    if not ELEVENLABS_API_KEY:
        return None
    url = f"{ELEVENLABS_BASE}/text-to-speech/{voice_id}"
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code == 200:
            return resp.content
    return None
