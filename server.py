# server.py
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import httpx, os, tempfile, json, asyncio
from pathlib import Path
from faster_whisper import WhisperModel

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
# DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")

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

@app.get("/models")
async def get_models():
    """Fetch available OLLAMA models"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(OLLAMA_URL.replace("/api/chat", "/api/tags"))
            r.raise_for_status()
            data = r.json()
            models = [model["name"] for model in data.get("models", [])]
            return {"models": models}
    except Exception as e:
        # Fallback to default model if OLLAMA is not available
        return {"models": [DEFAULT_MODEL]}

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

@app.post("/chat/stream")
async def chat_stream(payload: ChatIn):
    """Stream chat responses from Ollama as Server-Sent Events"""
    model = payload.model or DEFAULT_MODEL
    body = {"model": model, "messages": [m.dict() for m in payload.messages], "stream": True}
    
    async def generate():
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", OLLAMA_URL, json=body) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            # Parse each line as JSON
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            done = data.get("done", False)
                            
                            if content:
                                # Send content as SSE
                                yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"
                            
                            if done:
                                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                                break
                                
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
                            break
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

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

# ---------- HTTPS Server Startup ----------
if __name__ == "__main__":
    import uvicorn
    
    # Check if SSL files exist
    ssl_keyfile = "key.pem"
    ssl_certfile = "cert.pem"
    
    print("should be checking here")
    if os.path.exists(ssl_keyfile) and os.path.exists(ssl_certfile):
        print(f"[HTTPS] Starting server with SSL certificate")
        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=8443,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            reload=False  # Set to True for development
        )
    else:
        print(f"[HTTP] SSL files not found, starting without HTTPS")
        print(f"[HTTP] To enable HTTPS, generate certificates with:")
        print(f"[HTTP]   openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365")
        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=8000,
            reload=False
        )
