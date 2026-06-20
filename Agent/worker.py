import os
import asyncio
import io
import sys
import contextlib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi import File, UploadFile
import base64
import uuid
import uvicorn

# Worker that imports the agent and exposes a simple HTTP API
app = FastAPI(title="Agent Worker")

class QueryReq(BaseModel):
    question: str
    doc_id: Optional[str] = None
    mode: Optional[str] = None  # 'text2sql' or 'ocr_qa'


class UploadResp(BaseModel):
    doc_id: str
    text_preview: str
    image_url: str

class OcrQAReq(BaseModel):
    question: str
    doc_id: str

class OcrQAResp(BaseModel):
    doc_id: str
    answer: str
    used_llm: bool

# Lazy import agent module (do at startup so heavy init happens in worker)
agent = None
# Lock to prevent concurrent capture of stdout/stderr which would interleave logs
capture_lock = asyncio.Lock()

@app.on_event("startup")
async def startup_event():
    global agent
    # Import here so the LLM and DB init occur inside the worker process
    # Use package-relative import so we load Agent.agentic_workflow (not a top-level file)
    try:
        from . import agentic_workflow as agentic_workflow
    except Exception:
        # fallback to absolute package import
        import Agent.agentic_workflow as agentic_workflow
    agent = agentic_workflow
    # Initialize DB (best-effort)
    db_url = os.environ.get("DB_CONNECTION_URL")
    if db_url:
        try:
            await agent.initialize_database(uri=db_url)
        except Exception as e:
            # Log but continue; agentic_workflow has fallback behavior
            print(f"Worker: initialize_database failed: {e}")
    else:
        print("Worker: DB_CONNECTION_URL not set; skipping DB init")

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query")
async def query(req: QueryReq):
    global agent
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    q = req.question
    doc_id = getattr(req, 'doc_id', None)
    mode = getattr(req, 'mode', None)
    # Capture stdout/stderr produced by the agent for this request
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    async with capture_lock:
        try:
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                # Robust mode switch: if mode is 'ocr_qa' or doc_id is present, use OCR agent; else use SQL logic
                if (mode == 'ocr_qa') or (doc_id and (mode is None or mode == 'auto')):
                    # Use OCR Q&A agent
                    # ocr_agent_qa may be sync, so run in executor
                    import asyncio
                    answer = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: agent.ocr_agent_qa(q, doc_id)
                    )
                    result = answer
                    sql_out = None  # No SQL for OCR Q&A
                else:
                    # Use standard Text-to-SQL agentic workflow
                    result = await agent.agentic_query_process(q, None)
                    try:
                        sql_out = getattr(agent, 'LAST_GENERATED_SQL', None)
                    except Exception:
                        sql_out = None
        except ValueError as ve:
            # Treat known generation failures as normal answers so UI doesn't show nested JSON error
            err_msg = str(ve)
            if "Unable to generate SQL query" in err_msg:
                result = err_msg  # Provide user-visible plain text
                sql_out = None
            else:
                import traceback
                traceback.print_exc(file=buf_err)
                raise HTTPException(status_code=500, detail=err_msg)
        except Exception as e:
            import traceback
            traceback.print_exc(file=buf_err)
            raise HTTPException(status_code=500, detail=str(e))
    logs = buf_out.getvalue()
    errlogs = buf_err.getvalue()
    if errlogs:
        logs = logs + "\n" + errlogs
    return {"answer": result, "logs": logs, "sql": sql_out}


@app.post("/ocr-qa", response_model=OcrQAResp)
async def ocr_qa(req: OcrQAReq):
    """Answer a question about a previously uploaded document using its doc_id.

    This uses the agent's ocr_agent_qa helper which pulls stored OCR text and
    consults the LLM (or returns a low-memory fallback excerpt).
    """
    global agent
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    question = req.question
    doc_id = req.doc_id
    # Run potentially blocking OCR QA in thread pool if it's sync
    try:
        answer = await asyncio.get_event_loop().run_in_executor(
            None, lambda: agent.ocr_agent_qa(question, doc_id)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ocr_qa failed: {e}")
    used_llm = not answer.startswith("[Low memory]") and not answer.startswith("No OCR text") and not answer.startswith("Error while")
    return OcrQAResp(doc_id=doc_id, answer=answer, used_llm=used_llm)


@app.post("/upload-image", response_model=UploadResp)
async def upload_image(file: UploadFile = File(...)):
    """Accept an image, save locally, call MCP document_scanner.process eagerly,
    persist OCR to private chat store, and return a doc_id plus preview.
    """
    global agent
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    # Ensure uploads directory exists
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)

    # Save uploaded file
    filename = f"{uuid.uuid4()}_{file.filename}"
    path = os.path.join(uploads_dir, filename)
    try:
        contents = await file.read()
        with open(path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    # Call MCP document_scanner tool eagerly with base64 payload to avoid exposing file server
    b64 = base64.b64encode(contents).decode('utf-8')
    try:
        from .mcp_client import mcp_call_tool
        ctx = {"tool_call": {"name": "document_scanner.process", "args": {"image_bytes": b64}}}
        resp = mcp_call_tool(prompt="run document scanner", timeout=60.0, context=ctx)
        # Expect tool_result in response
        tool_result = resp.get("tool_result") if isinstance(resp, dict) else None
    except Exception:
        # Fall back to local direct call if MCP unavailable or tool failed
        tool_result = None
        try:
            if agent and hasattr(agent, 'tools') and hasattr(agent.tools, 'document_scanner'):
                try:
                    tool_result = agent.tools.document_scanner.process_image(image_bytes=b64)
                except Exception:
                    tool_result = None
        except Exception:
            tool_result = None
        if not tool_result:
            tool_result = {"doc_id": str(uuid.uuid4()), "text": "(ocr unavailable)", "metadata": {}}

    # Persist OCR to private chat store file for retrieval by doc_id
    doc_id = tool_result.get("doc_id") if isinstance(tool_result, dict) and tool_result.get("doc_id") else str(uuid.uuid4())
    ocr_text = tool_result.get("text", "") if isinstance(tool_result, dict) else str(tool_result)

    # Save to docs/ and also to chat_store_private for history
    docs_dir = os.path.join("chat_store", "docs")
    os.makedirs(docs_dir, exist_ok=True)
    doc_path = os.path.join(docs_dir, f"{doc_id}.txt")
    try:
        with open(doc_path, "w", encoding="utf-8") as df:
            df.write(ocr_text)
    except Exception as e:
        print(f"Failed to persist OCR text: {e}")

    # Also persist the original image path so Granite Vision can do direct visual Q&A
    img_path_file = os.path.join(docs_dir, f"{doc_id}.img_path")
    try:
        abs_img_path = os.path.abspath(path)
        with open(img_path_file, "w", encoding="utf-8") as ipf:
            ipf.write(abs_img_path)
        print(f"[upload] Saved image path for doc_id={doc_id}: {abs_img_path}")
    except Exception as e:
        print(f"[upload] Failed to persist image path: {e}")

    # Also add a short entry to chat_store_private for traceability
    try:
        try:
            from .agentic_workflow import chat_store_private, ChatMessage, MessageRole
        except ImportError:
            from Agent.agentic_workflow import chat_store_private, ChatMessage, MessageRole
        # Create minimal messages
        user_msg = ChatMessage(role=MessageRole.USER, content=f"Uploaded document {doc_id}")
        assistant_msg = ChatMessage(role=MessageRole.ASSISTANT, content=f"OCR stored: {ocr_text[:400]}")
        chat_store_private.add_message(key="conversation", message=user_msg)
        chat_store_private.add_message(key="conversation", message=assistant_msg)
        chat_store_private.persist(str("chat_store/chat_store_private.json"))
    except Exception:
        pass

    # Return doc id and small preview
    preview = ocr_text[:400]
    image_url = path
    return UploadResp(doc_id=doc_id, text_preview=preview, image_url=image_url)

if __name__ == '__main__':
    port = int(os.environ.get("WORKER_PORT", 8700))
    uvicorn.run(app, host="127.0.0.1", port=port)
