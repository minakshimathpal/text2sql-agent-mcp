"""
Vision MCP tool — direct visual Q&A via Ollama (qwen3.5:2b).

Registered in mcp_server.py as 'granite_vision.qa'.
Called by the worker when an image file is available for a doc_id.
"""
import os
from typing import Optional

def qa(image_path: str, question: str, model_path: Optional[str] = None) -> dict:
    """Answer a question about an image using qwen3.5:2b via Ollama.

    Args:
        image_path:  Absolute path to the image file on disk.
        question:    Natural language question about the image content.
        model_path:  (Legacy) Ignored. Uses qwen3.5:2b via Ollama.

    Returns:
        dict with keys:
            "answer"  – the model's response string
            "model"   – model path that was used
            "error"   – present only on failure (answer will be an error string)
    """
    print(f"[vision.qa] image={image_path} target_model=qwen3.5:2b")

    try:
        import requests
        import base64
    except ImportError as e:
        msg = f"[ERROR] Missing dependency: {e}. Install: pip install requests"
        print(f"[granite_vision.qa] {msg}")
        return {"answer": msg, "model": "qwen3.5:2b", "error": str(e)}

    if not os.path.isfile(image_path):
        msg = f"[ERROR] Image file not found: {image_path}"
        print(f"[granite_vision.qa] {msg}")
        return {"answer": msg, "model": "qwen3.5:2b", "error": "file_not_found"}

    print(f"[granite_vision.qa] Sending image to Ollama (qwen3.5:2b)")

    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip('/')
        url = f"{base_url}/api/chat"
        # Qwen-VL performs better when the prompt explicitly asks it to look closely at the image
        enhanced_question = f"Carefully analyze this image. {question}"
        payload = {
            "model": "qwen3.5:2b",
            "messages": [
                {
                    "role": "user",
                    "content": enhanced_question,
                    "images": [img_b64]
                }
            ],
            "stream": False,
            "keep_alive": 60,
            "options": {
                "num_gpu": -1,
                "num_ctx": 1024,
                "temperature": 0.2
            }
        }
        
        r = requests.post(url, json=payload, timeout=180)
        if r.status_code == 200:
            answer = r.json().get("message", {}).get("content", "").strip()
            print(f"[granite_vision.qa] Answer ({len(answer)} chars): {answer[:200]}")
            return {"answer": answer, "model": "qwen3.5:2b (Ollama)"}
        else:
            msg = f"Ollama HTTP {r.status_code}: {r.text}"
            print(f"[granite_vision.qa] {msg}")
            return {"answer": msg, "model": "qwen3.5:2b (Ollama)", "error": "http_error"}
            
    except Exception as e:
        msg = f"[ERROR] Ollama Q&A failed: {e}"
        print(f"[granite_vision.qa] {msg}")
        return {"answer": msg, "model": "qwen3.5:2b (Ollama)", "error": str(e)}
