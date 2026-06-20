Web frontend integration notes

Flow:
- User uploads an image via the web page -> POST /upload-image (web) -> proxied to worker /upload-image.
- Worker saves the image, calls MCP tool `document_scanner.process` (via mcp_client or direct call), persists OCR text and returns `doc_id` and a small preview.
- The web UI stores `doc_id` (in JS `currentDocId`) and includes it in subsequent question POSTs to /api/query.
- Web server forwards the question + doc_id to the worker /query endpoint. The worker will load the OCR text (from chat_store/docs/<doc_id>.txt) and call the agent helper (ocr_agent_qa) to answer using the OCR + LLM.

Endpoints added/used:
- web: POST /upload-image  -> proxies upload to worker
- web: POST /api/query     -> accepts {question, doc_id?} and forwards to worker /query
- worker: POST /upload-image -> handles image saving and OCR via MCP tool
- worker: POST /query      -> handles question + optional doc_id and runs the agent

Notes:
- The OCR cleaning logic is in `Agent/tools/document_scanner.py` and returns cleaned text that is LLM-friendly.
- The worker persists cleaned OCR under `chat_store/docs/<doc_id>.txt` for later retrieval.

Security:
- This is a dev scaffold; consider authentication and size limits for uploads in production.
