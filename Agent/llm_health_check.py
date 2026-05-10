import os, time, json, sys
from llama_index.llms.ollama import Ollama

BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.environ.get("FINETUNED_MODEL_NAME") or os.environ.get("OLLAMA_MODEL_NAME", "gemma3:1b")

print(f"Checking Ollama LLM connectivity: base_url={BASE_URL} model={MODEL}")

# Lightweight client check without llama_index first (raw HTTP) to see if daemon responds
try:
    import requests
    r = requests.get(f"{BASE_URL.rstrip('/')}/api/tags", timeout=5)
    print(f"/api/tags status: {r.status_code}")
    if r.ok:
        tags = r.json().get('models', [])
        names = [m.get('name') for m in tags]
        print(f"Available models: {names}")
        if MODEL not in names:
            print(f"Model '{MODEL}' not yet pulled. Run: ollama pull {MODEL}")
    else:
        print("Non-200 response from Ollama daemon")
except Exception as e:
    print(f"Raw HTTP check failed: {e}")

# LlamaIndex call
try:
    llm = Ollama(model=MODEL, base_url=BASE_URL, temperature=0.1, request_timeout=30)
    t0 = time.time()
    resp = llm.complete(prompt="SELECT 1;")
    dt = time.time() - t0
    print(f"LLM call success in {dt:.2f}s. Raw text: {resp.text[:120]!r}")
except Exception as e:
    print(f"LLM complete() failed: {type(e).__name__}: {e}")
    sys.exit(1)

print("Health check finished.")
