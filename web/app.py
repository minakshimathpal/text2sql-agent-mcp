import os
import httpx
import logging
import traceback
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

WORKER_URL = os.environ.get("WORKER_URL", "http://127.0.0.1:8700")
# Timeout (seconds) when calling the worker; re-execute and heavy queries can take longer.
# Make configurable via environment variable WORKER_TIMEOUT (default 300s).
WORKER_TIMEOUT = float(os.environ.get("WORKER_TIMEOUT", "450"))  # Increased to 450s to handle complex queries

# Reusable AsyncClient to avoid per-request connection setup overhead
client: httpx.AsyncClient | None = None

app = FastAPI(title="Text2SQL Web API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend under /static and provide a root route
static_dir = os.path.join(os.path.dirname(__file__), 'static')
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")


@app.get("/")
async def root_index():
    index_path = os.path.join(static_dir, 'index.html')
    return FileResponse(index_path)

class QueryReq(BaseModel):
    question: str
    # Optional doc_id produced by a prior /upload-image call
    doc_id: Optional[str] = None
    # Explicit mode hint from client: 'text2sql' or 'ocr_qa'. When omitted, worker may auto-detect.
    mode: Optional[str] = None

class OcrQAReq(BaseModel):
    question: str
    doc_id: str

class OcrQAResp(BaseModel):
    doc_id: str
    answer: str
    used_llm: bool

@app.get('/health')
async def health():
    return {"status": "ok"}

@app.post('/api/query')
async def api_query(req: QueryReq):
    # Use a shared AsyncClient created at startup to avoid TCP/TLS/HTTP handshakes per request.
    global client
    if client is None:
        # fallback to create a temporary client (shouldn't happen if startup ran)
        client = httpx.AsyncClient(timeout=WORKER_TIMEOUT)
    try:
        payload = {"question": req.question}
        # Forward explicit mode if provided so the worker doesn't infer from lingering doc_id
        if getattr(req, 'mode', None):
            payload['mode'] = req.mode
        if getattr(req, 'doc_id', None):
            payload['doc_id'] = req.doc_id
        r = await client.post(f"{WORKER_URL}/query", json=payload)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        # Log the full traceback and response body for debugging
        logging.error("Worker returned non-2xx status: %s", e.response.status_code)
        logging.error("Response text: %s", getattr(e.response, 'text', str(e)))
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=e.response.status_code, detail=getattr(e.response, 'text', str(e)))
    except httpx.ReadTimeout as e:
        # Worker took too long. Return a 504 gateway timeout with a helpful message.
        logging.error("Timeout contacting worker after %ss: %s", WORKER_TIMEOUT, str(e))
        logging.error(traceback.format_exc())
        return JSONResponse(status_code=504, content={
            "error": "worker_timeout",
            "message": f"Worker did not respond within {WORKER_TIMEOUT} seconds. The query may be long-running (re-execute) or the worker is busy. Try increasing WORKER_TIMEOUT or running the worker with more resources.",
        })
    except Exception as e:
        logging.error("Error contacting worker: %s", str(e))
        logging.error(traceback.format_exc())
        # Return a JSON response containing the error detail to help debugging from the client
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})


@app.post('/api/ocr-qa')
async def api_ocr_qa(req: OcrQAReq):
    global client
    if client is None:
        client = httpx.AsyncClient(timeout=WORKER_TIMEOUT)
    try:
        payload = {"question": req.question, "doc_id": req.doc_id}
        r = await client.post(f"{WORKER_URL}/ocr-qa", json=payload)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        logging.error("Worker OCR QA non-2xx: %s", e.response.status_code)
        logging.error("Response text: %s", getattr(e.response, 'text', str(e)))
        return JSONResponse(status_code=e.response.status_code, content={"error": getattr(e.response, 'text', str(e))})
    except Exception as e:
        logging.error("Error contacting worker OCR QA: %s", str(e))
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})


@app.on_event('startup')
async def startup_event():
    global client
    if client is None:
        client = httpx.AsyncClient(timeout=WORKER_TIMEOUT)


@app.on_event('shutdown')
async def shutdown_event():
    global client
    if client is not None:
        await client.aclose()


@app.post('/api/test')
async def api_test(req: QueryReq):
    # Simple echo endpoint to verify POST handling without contacting the worker
    return {"ok": True, "echo": req.question}


@app.post('/upload-image')
async def upload_image(file: bytes = None, request: Request = None):
    """Proxy an image upload to the worker's /upload-image endpoint.

    Accepts multipart/form-data file field 'file' from the browser.
    Returns the worker JSON (doc_id, text_preview, image_url) on success.
    """
    # Use the request.form() to access the uploaded file
    try:
        form = await request.form()
        f = form.get('file')
        if f is None:
            raise HTTPException(status_code=400, detail='file field is required')
        # forward the file to the worker
        files = {'file': (f.filename, await f.read(), f.content_type)}
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(f"{WORKER_URL}/upload-image", files=files)
            r.raise_for_status()
            return JSONResponse(status_code=200, content=r.json())
    except HTTPException:
        raise
    except Exception as e:
        logging.error('upload-image proxy failed: %s', str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/test')
async def api_test_get():
    return {"ok": True, "msg": "POST to /api/test with JSON {question: ...} to echo"}

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('WEB_PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)
