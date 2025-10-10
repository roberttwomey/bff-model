# server.py
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import httpx, os, tempfile
from pathlib import Path
from faster_whisper import WhisperModel

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

# Whisper config
WHISPER_SIZE   = os.environ.get("WHISPER_SIZE", "tiny.en")  # tiny.en/base.en/small.en/…
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")    # "cpu" or "cuda"
COMPUTE_TYPE   = "int8" if WHISPER_DEVICE == "cpu" else "float16"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend from ./static
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# ---------- Models ----------
class Message(BaseModel):
    role: str
    content: str

class ChatIn(BaseModel):
    messages: list[Message] = Field(default_factory=list)
    model: str | None = None

# ---------- Routes ----------
@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {"ok": True}

@app.post("/chat")
async def chat(payload: ChatIn):
    model = payload.model or DEFAULT_MODEL
    body = {"model": model, "messages": [m.dict() for m in payload.messages], "stream": False}
    async with httpx.AsyncClient(timeout=300.0) as client:
        r = await client.post(OLLAMA_URL, json=body)
        r.raise_for_status()
        data = r.json()
    reply = (data.get("message") or {}).get("content", "")
    return JSONResponse({"reply": reply})

# ---------- Whisper STT ----------
print(f"[STT] Loading faster-whisper: {WHISPER_SIZE} on {WHISPER_DEVICE} ({COMPUTE_TYPE})")
whisper_model = WhisperModel(WHISPER_SIZE, device=WHISPER_DEVICE, compute_type=COMPUTE_TYPE)

@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    """
    Accepts audio blobs from the browser (webm/ogg/wav).
    Returns: {"text": "...", "duration": <seconds>}
    """
    suffix = Path(file.filename or "").suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        raw = await file.read()
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        segments, info = whisper_model.transcribe(
            tmp_path,
            language="en",      # set None for auto-detect
            vad_filter=True     # slightly better punctuation
        )
        text = "".join(s.text for s in segments).strip()
        return {"text": text, "duration": getattr(info, "duration", None)}
    finally:
        try: os.remove(tmp_path)
        except: pass
