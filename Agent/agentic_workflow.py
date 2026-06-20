"""
Agentic Workflow Main Module
---------------------------
This file implements the core agentic workflow for text-to-SQL, including:
- LLM orchestration (Ollama, MCP, fallback)
- Memory management and safety checks
- SQL cleaning, validation, and artifact stripping
- Agentic orchestration (plan/micro/LLM phases)
- Robust fallback and error handling

All agent logic is preserved. Comments clarify key logic and utility functions.
"""

import os
import asyncio
import re
import gc  # Explicit garbage collection import
import signal  # For timeouts
import threading  # For thread management
import weakref  # For weak references to avoid memory leaks
import time  # For timing operations
from typing import Dict, Any, List, Optional
import json
import uuid
from datetime import datetime
from pathlib import Path
import psutil  # Import for memory monitoring
import requests

# Database-related imports
from sqlalchemy import create_engine, text, Table, MetaData, inspect
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import NoSuchTableError

# LLM imports
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.storage.chat_store import SimpleChatStore
try:
    from .invariants import enforce_invariants, classify_question  # package-relative
except ImportError:
    # Fallback when executed as a standalone script (no package context)
    # Try package-qualified import which works when running as a module (python -m)
    from Agent.invariants import enforce_invariants, classify_question

from dotenv import load_dotenv
# Load .env from project root first, then Agent/ subdirectory.
# This ensures OCR_QA_TIMEOUT, OLLAMA_OCR_QA_MODEL etc. are picked up
# regardless of which directory the server is started from.
import os as _os
_dotenv_candidates = [
    '.env',                                                       # CWD fallback
    _os.path.join(_os.path.dirname(__file__), '..', '.env'),    # project root .env
    _os.path.join(_os.path.dirname(__file__), '.env'),          # Agent/.env  (most specific)
]
for _dotenv_path in _dotenv_candidates:
    if _os.path.isfile(_dotenv_path):
        print(f"[DEBUG] Loading .env from: {_dotenv_path}")
        load_dotenv(dotenv_path=_dotenv_path, override=False)     # Environment variables (Docker) now win over .env files
del _os, _dotenv_candidates, _dotenv_path
print(f"[DEBUG] Before CWD load_dotenv, DB_CONNECTION_URL={os.environ.get('DB_CONNECTION_URL')}")
load_dotenv()  # also load CWD .env for any remaining vars
print(f"[DEBUG] After CWD load_dotenv, DB_CONNECTION_URL={os.environ.get('DB_CONNECTION_URL')}")

# Simple initialization
DB_CONNECTION_URL = os.environ.get("DB_CONNECTION_URL", None)
print(f"[DEBUG] Final DB_CONNECTION_URL={DB_CONNECTION_URL}")
if not DB_CONNECTION_URL:
    raise ValueError("DB_CONNECTION_URL environment variable is not set")

# Import our memory management utilities (robust import with fallbacks)
import importlib
import sys

memory_manager = None
safe_check_memory = None
TimeoutManager = None
_last_import_error = None

# Try several import strategies: package-relative, package-absolute, and top-level
try_names = []
if __package__:
    # e.g. when imported as Agent.agentic_workflow, __package__ == 'Agent'
    try_names.append(f"{__package__}.memory_manager")
# Also try the absolute package path
try_names.append("Agent.memory_manager")
# Finally try top-level fallback
try_names.append("memory_manager")

for modname in try_names:
    try:
        mod = importlib.import_module(modname)
        # Extract expected symbols if present
        memory_manager = getattr(mod, 'memory_manager', None)
        safe_check_memory = getattr(mod, 'safe_check_memory', None)
        TimeoutManager = getattr(mod, 'TimeoutManager', None)
        if memory_manager is not None and safe_check_memory is not None and TimeoutManager is not None:
            break
    except Exception as e:
        _last_import_error = e

if memory_manager is None or safe_check_memory is None or TimeoutManager is None:
    # Provide a clear diagnostic so startup logs show why import failed
    print("Failed to import Agent.memory_manager via tried paths:", try_names)
    print("cwd=", __import__('os').getcwd())
    print("sys.path[0]=", sys.path[0] if len(sys.path) > 0 else None)
    if _last_import_error:
        # Re-raise the last error to preserve traceback
        raise _last_import_error             
    else:
        raise ImportError("Could not locate memory_manager module; ensure Agent/memory_manager.py exists and package imports are correct")

# Ollama configuration
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


# --- Employee Database Schema for Prompt Injection ---
EMPLOYEE_DB_SCHEMA = (
    "Database schema:\n"
    "- employee(emp_no, birth_date, first_name, last_name, gender, hire_date)\n"
    "- department(dept_no, dept_name)\n"
    "- dept_emp(emp_no, dept_no, from_date, to_date)\n"
    "- dept_manager(emp_no, dept_no, from_date, to_date)\n"
    "- salary(emp_no, salary, from_date, to_date)  # column name is 'salary' in this dataset\n"
    "- title(emp_no, title, from_date, to_date)\n"
    "- views: current_dept_emp (current department per employee), dept_emp_latest_date\n"
    "\n"
    "Hint (for the LLM):\n"
    "- To get CURRENT records (managers, salaries, department assignments), ALWAYS filter by `to_date = '9999-01-01'`.\n"
    "- To get MANAGER NAMES, you MUST join `dept_manager` (dm) with `employee` (e) on `emp_no`.\n"
    "- To get DEPARTMENT NAMES for employees, join `dept_emp` (de) with `department` (d) on `dept_no`.\n"
    "- The 'employee' table does NOT have a 'dept_no' column. To find an employee's department name, you MUST join 'dept_emp' (de) with 'department' (d) on 'dept_no'.\n"
    "- Salaries are in table `salary` (column `salary`). Always use `s.to_date = '9999-01-01'` for current pay.\n"
)

# Allow a dedicated finetuned model name (merged LoRA) via FINETUNED_MODEL_NAME env.
# Resolution order:
#  1. FINETUNED_MODEL_NAME (e.g., gemma1b-text2sql)
#  2. OLLAMA_MODEL_NAME (defaults to base gemma3:1b)
#  3. Hard fallback 'gemma3:1b'
FINETUNED_MODEL_NAME = os.environ.get("FINETUNED_MODEL_NAME")
BASE_MODEL_FALLBACK = "qwen2.5-coder:1.5b" #"gemma3:1b-it-qat"
OLLAMA_MODEL_NAME = FINETUNED_MODEL_NAME or os.environ.get("OLLAMA_MODEL_NAME", BASE_MODEL_FALLBACK)
FALLBACK_MODEL_NAME = BASE_MODEL_FALLBACK  # Keep fallback stable
if FINETUNED_MODEL_NAME:
    print(f"🦙 Using finetuned Ollama model: {OLLAMA_MODEL_NAME} (base fallback: {FALLBACK_MODEL_NAME})")
else:
    print(f"🦙 Using base Ollama model: {OLLAMA_MODEL_NAME} (fallback: {FALLBACK_MODEL_NAME})")

# OCR Q&A model configuration
# PRIMARY: Granite Vision VLM (GRANITE_MODEL_PATH env) — used when image file is available
# FALLBACK: lightweight Ollama text model — used for old docs without a saved image
GRANITE_MODEL_PATH = (
    os.environ.get("GRANITE_MODEL_PATH") or
    os.environ.get("GRANITE_MODEL") or
    "qwen3.5:2b"
)
print(f"[GRANITE_VQA] Target Model: {GRANITE_MODEL_PATH} (via Ollama)")
# Text-only fallback: Ollama model used when no image file exists for the doc.
# NOT the same as GRANITE_MODEL_PATH — this runs via Ollama REST API, not HuggingFace.
OLLAMA_OCR_QA_MODEL = os.environ.get("OLLAMA_OCR_QA_MODEL", "gemma3:1b-it-qat")
print(f"[OCR_QA] Text fallback model (Ollama): {OLLAMA_OCR_QA_MODEL}")
OCR_QA_TIMEOUT = float(os.environ.get("OCR_QA_TIMEOUT", "45"))
print(f"[OCR_QA] Text fallback timeout={OCR_QA_TIMEOUT}s")

def check_memory_safety():
    """Check if there's enough memory to safely process a request"""
    # Use our improved memory checking system
    return safe_check_memory()

# Global variables for database connection
available_tables = []
inspector = None
sql_database = None
async_engine = None
async_session = None

# Initialize chat store directory and files
chat_store_dir = Path('chat_store')
chat_store_dir.mkdir(parents=True, exist_ok=True)

# Define chat store file paths
public_store_path = chat_store_dir/'chat_store_public.json'
private_store_path = chat_store_dir/'chat_store_private.json'
# Initialize empty chat store structure with conversation key
initial_store = {"conversation": []}

# Create JSON files if they don't exist
if not public_store_path.exists():   
    with open(public_store_path, 'w') as f:
        json.dump(initial_store, f, indent=2)

if not private_store_path.exists():    
    with open(private_store_path, 'w') as f:
        json.dump(initial_store, f, indent=2)

# Initialize chat stores
print("Debug: Initializing chat stores")
chat_store = SimpleChatStore()
chat_store_public = chat_store.from_persist_path(str(public_store_path))
chat_store_private = chat_store.from_persist_path(str(private_store_path))
print("Debug: Chat stores initialized")

# Utility: robustly extract clean SQL from LLM/artifact wrappers
def extract_clean_sql(raw: str) -> str:
    """Remove <<<SQL_START>>>, <<<SQL_END>>>, code fences, and similar wrappers from SQL text."""
    import re
    if not raw or not isinstance(raw, str):
        return raw
    s = raw.strip()
    # Remove <<<SQL_START>>> ... <<<SQL_END>>> blocks
    s = re.sub(r"<<<SQL_START>>>[\s\S]*?<<<SQL_END>>>", lambda m: m.group(0).replace('<<<SQL_START>>>','').replace('<<<SQL_END>>>','').strip(), s, flags=re.IGNORECASE)
    # Remove <<<SQL_REVISED_START>>> ... <<<SQL_REVISED_END>>> blocks
    s = re.sub(r"<<<SQL_REVISED_START>>>[\s\S]*?<<<SQL_REVISED_END>>>", lambda m: m.group(0).replace('<<<SQL_REVISED_START>>>','').replace('<<<SQL_REVISED_END>>>','').strip(), s, flags=re.IGNORECASE)
    # Remove code fences (```sql ... ``` or ```)
    s = re.sub(r"```(?:sql)?([\s\S]*?)```", lambda m: m.group(1).strip(), s, flags=re.IGNORECASE)
    # Remove any remaining <<<...>>> wrappers
    s = re.sub(r"<<<[A-Z_]+>>>", "", s)
    
    # Extract only the SQL portion before any explanations.
    # Handles both plain "Explanation:" and markdown "### Explanation:" headers.
    _explanation_pattern = r';\s*\n[\s\S]*?(?:#{1,3}\s*[Ee]xplanation|[Ee]xplanation)\s*:'
    if re.search(_explanation_pattern, s):
        s = re.split(_explanation_pattern, s)[0].rstrip() + ';'
    elif re.search(r';\s*\n\s*[Ee]xplanation:', s):
        s = re.split(r';\s*\n\s*[Ee]xplanation:', s)[0] + ';'


    # Remove leading/trailing whitespace and repeated newlines
    s = s.strip()
    # Remove any leading/trailing code fence lines
    s = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", s, flags=re.MULTILINE)
    
    # Remove any trailing explanation text that might still be there
    lines = s.split('\n')
    sql_lines = []
    for line in lines:
        line = line.strip()
        # Stop at explanation markers — including markdown ### headers
        if re.match(r'^#{1,3}\s*(explanation|note|this query)', line, re.IGNORECASE):
            break
        if re.match(r'^(explanation|note|this query):', line, re.IGNORECASE):
            break
        if line.lower().startswith('explanation:'):
            break
        if re.match(r'^\d+\.\s+', line):  # Numbered explanation steps
            break
        sql_lines.append(line)
    
    s = '\n'.join(sql_lines).strip()
    
    # NEW: Safety check for multiple commands. 
    # If the LLM outputted multiple queries (separated by semicolons), 
    # we only want the LAST valid one (which is usually the most refined).
    if s.count(';') > 1:
        # Split by semicolon, keep only those that look like queries
        statements = [stmt.strip() for stmt in s.split(';') if 'select' in stmt.lower()]
        if statements:
            # Use the last one as it's typically the final refinement
            s = statements[-1] + ';'
            print(f"[CLEAN_SQL] Multiple queries detected. Selected last valid statement.")
            
    return s.strip()

# -----------------------------
# OCR Q&A dedicated helpers
# -----------------------------
def _get_ocr_text_by_doc_id(doc_id: str) -> str:
    try:
        p = Path('chat_store') / 'docs' / f"{doc_id}.txt"
        if not p.exists():
            return ""
        return p.read_text(encoding='utf-8')
    except Exception:
        return ""

def _get_image_path_by_doc_id(doc_id: str) -> str:
    """Return the saved image file path for doc_id, or empty string if not found."""
    try:
        p = Path('chat_store') / 'docs' / f"{doc_id}.img_path"
        if not p.exists():
            return ""
        return p.read_text(encoding='utf-8').strip()
    except Exception:
        return ""

def _granite_vision_qa(image_path: str, question: str) -> str:
    """Answer a question about an image using IBM Granite Vision 3.1-2b (direct VLM Q&A).

    Loads the model from local HF cache with aggressive quantization to stay within
    the RAM budget (4-bit → 8-bit → fp16 fallback chain). The model is explicitly
    deleted after generation to free memory for the SQL model to reload.
    """
    import gc
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq
        from PIL import Image as PILImage
    except ImportError as e:
        return f"[ERROR] Granite Vision dependencies not installed: {e}"

    model_path = GRANITE_MODEL_PATH
    cache_dir  = os.environ.get("HF_HOME") or r"D:\hf_cache"
    offload_dir = os.path.join(cache_dir, "offload")
    os.makedirs(offload_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[GRANITE_VQA] Loading {model_path} on {device} from {cache_dir}")

    processor = vlm_model = None
    try:
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, cache_dir=cache_dir)

        # Try 4-bit → 8-bit → plain fp16 to fit within RAM budget
        load_kwargs = dict(local_files_only=True, cache_dir=cache_dir, low_cpu_mem_usage=True)
        loaded = False
        for attempt, extra in enumerate([
            {"load_in_4bit": True},
            {"load_in_8bit": True},
            {"device_map": "auto", "offload_folder": offload_dir},
        ]):
            try:
                vlm_model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs, **extra)
                print(f"[GRANITE_VQA] Loaded (attempt {attempt+1}: {list(extra.keys())[0]})")
                loaded = True
                break
            except Exception as le:
                print(f"[GRANITE_VQA] Load attempt {attempt+1} failed: {le}")
        if not loaded:
            return "[ERROR] Granite Vision could not be loaded — all quantisation attempts failed."

        img = PILImage.open(image_path).convert("RGB")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_path},
                    {"type": "text",  "text": (
                        f"{question}\n\n"
                        "Answer concisely using only information visible in the image. "
                        "If you see a table, read the correct row and column carefully."
                    )},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Move inputs to model device
        try:
            model_device = next(vlm_model.parameters()).device
            inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        except Exception:
            pass

        with torch.no_grad():
            output_ids = vlm_model.generate(**inputs, max_new_tokens=256, do_sample=False)
        answer = processor.decode(output_ids[0], skip_special_tokens=True)

        # Strip echoed prompt (Granite echoes conversation before the answer)
        if "ASSISTANT" in answer.upper():
            answer = answer.split("ASSISTANT")[-1].strip(" :").strip()
        elif question.lower()[:20] in answer.lower():
            idx = answer.lower().rfind(question.lower()[:20])
            answer = answer[idx + len(question):].strip(" :").strip() if idx != -1 else answer

        print(f"[GRANITE_VQA] Answer (chars={len(answer)}): {answer[:200]}")
        return answer.strip() or "[ERROR] Granite Vision returned an empty response."

    except Exception as e:
        print(f"[GRANITE_VQA] Generation failed: {e}")
        return f"[ERROR] Granite Vision Q&A failed: {e}"
    finally:
        # Aggressively free VLM memory so SQL model can reload
        try:
            del vlm_model, processor
        except Exception:
            pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        print("[GRANITE_VQA] Model unloaded; RAM released.")


def _structure_invoice_text(raw: str) -> str:
    """Convert flat Tesseract OCR output into a structured KEY: VALUE format.

    Tesseract merges field labels and values on the same line and squashes
    table rows together, causing a 1B LLM to confuse GST numbers with phone
    numbers, amounts with dates, etc.  This function:
    - Recognises common invoice field patterns via regex
    - Emits explicit 'FIELD_NAME: value' lines for the LLM
    - Preserves the rest of the text verbatim
    So the model receives unambiguous context like:
        GSTIN (Seller): 24HDE7487RE5RT4
        Customer Phone: 9372346666
        Customer GSTIN: 07AOLCC1206D126
    """
    import re
    lines = raw.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    structured = []
    seen_keys = set()

    def emit(key: str, val: str):
        tag = f"{key}: {val.strip()}"
        if tag not in seen_keys:
            seen_keys.add(tag)
            structured.append(tag)

    # Regex patterns for common Indian invoice fields
    patterns = [
        # GSTIN – 15-char alphanumeric starting with 2 digits
        (r'GSTIN\s*(?:No\.?|:)?\s*([0-9]{2}[A-Z0-9]{13})', 'GSTIN'),
        # Phone numbers (10-digit Indian mobile or landline with STD)
        (r'(?:PHONE|TEL|MOB(?:ILE)?)\s*[:\.]?\s*([0-9\-\+\s]{8,15})', 'Phone'),
        # Invoice number
        (r'Invoice\s*No\.?\s*[:\.]?\s*([A-Z0-9/\-]+)', 'Invoice Number'),
        # Challan number
        (r'Challan\s*No\.?\s*[:\.]?\s*([0-9]+)', 'Challan Number'),
        # Invoice / Challan dates
        (r'Invoice\s*Date\s*[:\.]?\s*(\d{1,2}[-/]\w{3,9}[-/]\d{2,4})', 'Invoice Date'),
        (r'Challan\s*Date\s*[:\.]?\s*(\d{1,2}[-/]\w{3,9}[-/]\d{2,4})', 'Challan Date'),
        (r'Due\s*Date\s*[:\.]?\s*(\d{1,2}[-/]\w{3,9}[-/]\d{2,4})', 'Due Date'),
        (r'Delivery\s*Date\s*[:\.]?\s*(\d{1,2}[-/]\w{3,9}[-/]\d{2,4})', 'Delivery Date'),
        # E-Way number
        (r'E-?Way\s*No\.?\s*[:\.]?\s*([A-Z0-9]+)', 'E-Way Number'),
        # LR / PO numbers
        (r'L\.?R\.?\s*No\.?\s*[:\.]?\s*([0-9]+)', 'LR Number'),
        (r'P\.?O\.?\s*No\.?\s*[:\.]?\s*([0-9]+)', 'PO Number'),
        # PAN
        (r'PAN\s*[:\.]?\s*([A-Z]{5}[0-9]{4}[A-Z])', 'PAN'),
        # Bank details
        (r'Bank\s*(?:Account|A/?C)\s*(?:No\.?|Number)?\s*[:\.]?\s*([0-9]+)', 'Bank Account Number'),
        (r'IFSC\s*[:\.]?\s*([A-Z]{4}0[A-Z0-9]{6})', 'Bank IFSC'),
        # Totals
        (r'Total\s+(?:Amount)?\s*After\s*Tax\s*[:\.]?\s*[\u20B9Rs\.]*\s*([\d,\.]+)', 'Total Amount After Tax'),
        (r'Taxable\s+(?:Amount|Value)\s*[:\.]?\s*([\d,\.]+)', 'Taxable Amount'),
        (r'Add\s*[:\.]?\s*IGST\s*[:\.]?\s*([\d,\.]+)', 'IGST Amount'),
    ]

    # Scan all lines for known field patterns
    full_text = '\n'.join(lines)
    # Track which GSTINs we've seen so we can label seller vs customer
    gstin_count = 0
    for line in lines:
        for pat, label in patterns:
            m = re.search(pat, line, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                if label == 'GSTIN':
                    gstin_count += 1
                    lbl = 'Seller GSTIN' if gstin_count == 1 else 'Customer GSTIN'
                    emit(lbl, val)
                elif label == 'Phone':
                    emit('Customer Phone', val)
                else:
                    emit(label, val)

    # Extract company name (typically first non-empty ALL-CAPS-ish line)
    for line in lines:
        stripped = line.strip()
        if len(stripped) > 4 and stripped.isupper() and not re.search(r'\d{5,}', stripped):
            emit('Company Name', stripped)
            break

    # Extract customer name after 'M/S' or 'Bill to'
    for i, line in enumerate(lines):
        if re.search(r'^(M/?S|Bill\s*to)\b', line.strip(), re.IGNORECASE):
            # Customer name is likely the next non-empty line or on same line after M/S
            rest = re.sub(r'^(M/?S|Bill\s*to)\s*', '', line.strip(), flags=re.IGNORECASE).strip()
            if rest:
                emit('Customer Name', rest)
            elif i + 1 < len(lines) and lines[i+1].strip():
                emit('Customer Name', lines[i+1].strip())
            break

    # --- Product line items ---
    # Scan for rows like: "2 | Stanley Hammer 8295 1.00 PCS 568.00 568.00 9.00 51.12 619.12"
    # OCR often squashes all rows onto one long line; use the item-number+pipe as separator.
    item_pat = re.compile(
        r'\b(\d{1,3})\s*[|Ill1]\s*'          # item number + pipe  (I/l/1 are common OCR artefacts for |)
        r'([A-Za-z][A-Za-z\s\-]{2,40}?)\s+'  # product name
        r'(\d{4,8})\s+'                        # HSN / SAC code
        r'(\d+(?:\.\d+)?)\s+'                  # quantity
        r'(?:PCS|NOS|KG|LT|MT|SET|BOX|PC|NO|EA)\.?\s+'  # unit (consumed but not captured)
        r'([\d,]+(?:\.\d+)?)'                  # rate (first numeric after unit)
        r'((?:\s+[\d,]+(?:\.\d+)?){1,4})',     # remaining numbers (taxable value, GST, total etc.)
        re.IGNORECASE
    )
    for m in item_pat.finditer(full_text):
        item_no    = m.group(1)
        name       = m.group(2).strip()
        hsn        = m.group(3)
        qty        = m.group(4)
        rate       = m.group(5)
        rest_nums  = re.findall(r'[\d,]+(?:\.\d+)?', m.group(6) or '')
        # rest_nums order depends on invoice: [taxable_value, gst%, gst_amount, total]
        taxable    = rest_nums[0] if len(rest_nums) > 0 else rate
        total      = rest_nums[-1] if len(rest_nums) > 1 else taxable
        label = f"Product {item_no} - {name}"
        emit(label, f"HSN={hsn}, Qty={qty}, Rate={rate}, Taxable Value={taxable}, Total={total}")

    # Build the final structured block
    header = "=== STRUCTURED DOCUMENT FIELDS ==="
    body = "=== RAW OCR TEXT ===\n" + raw.strip()
    if structured:
        return header + '\n' + '\n'.join(structured) + '\n\n' + body
    return body


def _ollama_unload_model(model: str) -> bool:
    """Ask Ollama to immediately unload a model from RAM (keep_alive=0).

    Uses /api/generate with a minimal prompt and keep_alive=0 to signal Ollama
    to release the model. Works for models loaded by any client (LlamaIndex etc).
    Returns True if the request succeeded, False otherwise (non-fatal).
    """
    base = OLLAMA_BASE_URL.rstrip('/')
    try:
        # A single-space prompt ensures Ollama actually processes the request
        # (empty string may be skipped); keep_alive=0 causes immediate unload.
        resp = requests.post(
            f"{base}/api/generate",
            json={"model": model, "prompt": " ", "keep_alive": 0, "stream": False},
            timeout=90.0,
        )
        print(f"[OCR_QA] Unloaded model '{model}' from Ollama RAM (status={resp.status_code})")
        return resp.status_code < 300
    except Exception as e:
        print(f"[OCR_QA][WARN] Failed to unload '{model}': {e}")
        return False


def _ollama_generate_text(model: str, prompt: str, timeout: float) -> str:
    """Minimal REST call to Ollama generate API to avoid llama_index routing.

    Uses OLLAMA_BASE_URL env and returns the concatenated response text.
    num_ctx is chosen adaptively based on available RAM so we don't cause
    excessive paging on memory-constrained machines.
    """

    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"

    # Pick num_ctx based on current available memory
    # ≥800 MB free  -> 2048  (full invoice fits comfortably)
    # 400–800 MB     -> 1024  (still fits most invoices)
    # <400 MB        -> 512   (minimal; avoid paging)
    try:
        stats = memory_manager.get_memory_stats()
        avail_mb = stats.get("available_mb", 999.0)
    except Exception:
        avail_mb = 999.0

    if avail_mb >= 800:
        num_ctx = 1024
        num_predict = 200
    elif avail_mb >= 400:
        num_ctx = 512
        num_predict = 150
    else:
        num_ctx = 256
        num_predict = 100

    payload = {
    "model": model,
    "prompt": prompt,
    "stream": False,
    "keep_alive": 0,   # unload immediately after response — prevents gemma3 squatting in RAM
    "options": {
        "temperature": 0.2,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "num_gpu": -1,      # keep on CPU same as Qwen; prevents GPU/CPU swap latency
        "num_thread": 2,   # 2 threads is sufficient for 1B model on CPU
        }
    }
    try:
        # Debug request meta (do not print full prompt to avoid huge logs here)
        try:
            print(f"[DEBUG][OLLAMA_REQUEST] url={url} model={model} prompt_chars={len(prompt)} timeout={timeout} num_ctx={num_ctx} avail_mb={avail_mb:.0f}")
        except Exception:
            pass
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        # Log only response length, not full body (can be very large)
        resp_text = data.get('response') or data.get('text') or str(data)
        print(f"[DEBUG][OLLAMA_RESPONSE] model={model} response_chars={len(resp_text)}")
        # Non-stream returns {response: "..."}
        txt = resp_text
        return txt
    except Exception as e:
        print(f"[ERROR][OLLAMA_API] Ollama generate failed: {e}")
        return f"Error: Ollama generate failed: {e}"

def ocr_agent_qa(question: str, doc_id: str, *, model: str | None = None, timeout: float | None = None) -> str:
    """Answer a question about a previously uploaded document.

    PRIMARY:  If the original image is available (.img_path), routes to Granite Vision
              via MCP (granite_vision.qa tool) for direct visual Q&A.
    FALLBACK: If only OCR text is available (.txt), builds a grounded prompt and
              calls OCR_TEXT_FALLBACK_MODEL (gemma3) via Ollama.
    """
    text = _get_ocr_text_by_doc_id(doc_id)

    # PRIMARY PATH: Granite Vision direct visual Q&A
    image_path = _get_image_path_by_doc_id(doc_id)

    # ── DIAGNOSTIC (temporary) ──────────────────────────────────────────
    _img_path_file = Path('chat_store') / 'docs' / f"{doc_id}.img_path"
    print(f"[OCR_QA_DIAG] doc_id          = {doc_id}")
    print(f"[OCR_QA_DIAG] .img_path file  = {_img_path_file.resolve()} | exists={_img_path_file.exists()}")
    print(f"[OCR_QA_DIAG] image_path      = {image_path!r}")
    print(f"[OCR_QA_DIAG] isfile(image)   = {os.path.isfile(image_path) if image_path else 'N/A (empty)'}")
    # ────────────────────────────────────────────────────────────────────

    if image_path and os.path.isfile(image_path):
        print(f"[OCR_QA] Using Granite Vision for doc_id={doc_id}, image={image_path}")
        # Unload Qwen first to free RAM for the VLM
        print(f"[OCR_QA] Swapping: unloading SQL model '{OLLAMA_MODEL_NAME}' to free RAM for Granite Vision")
        _ollama_unload_model(OLLAMA_MODEL_NAME)
        time.sleep(1.0)  # let OS reclaim pages
        # Route through MCP so Granite Vision runs as a proper registered agent tool
        try:
            from .mcp_client import mcp_call_tool
        except ImportError:
            from Agent.mcp_client import mcp_call_tool
        try:
            ctx = {
                "tool_call": {
                    "name": "granite_vision.qa",
                    "args": {
                        "image_path": image_path,
                        "question": question,
                        "model_path": GRANITE_MODEL_PATH,
                    },
                }
            }
            resp = mcp_call_tool(prompt="run granite vision qa", timeout=300.0, context=ctx)
            tool_result = resp.get("tool_result") if isinstance(resp, dict) else None
            if isinstance(tool_result, dict):
                return tool_result.get("answer", "[ERROR] No answer returned from granite_vision.qa")
            return str(tool_result) if tool_result else "[ERROR] granite_vision.qa returned no result"
        except Exception as e:
            print(f"[OCR_QA] MCP granite_vision.qa failed: {e}")
            return f"[ERROR] Vision Q&A timed out or failed. Please ensure Docker has enough RAM (6GB+) and try again. ({e})"

    # No image available — this doc was uploaded before the Granite Vision upgrade.
    # Returning a clear message instead of a wrong answer from the old gemma3 pipeline.
    print(f"[OCR_QA] No image found for doc_id={doc_id} — document predates Granite Vision upgrade")
    return (
        "⚠️ This document was uploaded before the Granite Vision upgrade. "
        "Please click 'Upload Image' to re-upload the image — "
        "Granite Vision will then answer your question directly from the image."
    )

# ---------------------------------------------------------------------------
# Conversation Memory Index (improves recall vs naive substring search)
# ---------------------------------------------------------------------------
question_memory_index = {}  # normalized_question -> last answer

def normalize_question(q: str) -> str:
    """Create a stable normalization of a user question for indexing.
    Steps: lowercase, remove punctuation except underscores, collapse whitespace.
    """
    import re
    q = q.lower()
    # remove any document snippet brackets if present (they make later repeats unmatchable)
    q = re.sub(r"\[document .*?excerpt end\]", "", q, flags=re.IGNORECASE|re.DOTALL)
    # strip bracketed metadata sections
    q = re.sub(r"\[[^\]]+\]", "", q)
    # punctuation -> space (keep underscores)
    q = re.sub(r"[^a-z0-9_]+", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q

def rebuild_memory_index():
    """Rebuild in-memory index from persisted chat history."""
    question_memory_index.clear()
    msgs = chat_store_private.get_messages(key="conversation")
    last_user = None
    for m in msgs:
        if m.role == MessageRole.USER:
            last_user = m.content
        elif m.role == MessageRole.ASSISTANT and last_user:
            key = normalize_question(last_user)
            question_memory_index[key] = m.content
            last_user = None
    print(f"[MEM-INDEX] Rebuilt memory index with {len(question_memory_index)} entries")

# Build initial index
try:
    rebuild_memory_index()
except Exception as e:
    print(f"[MEM-INDEX][WARN] Failed to build initial index: {e}")

# ---------------------------------------------------------------------------
# Feature flags / global controls
# ---------------------------------------------------------------------------
# Disable ALL template-based fallback logic to enforce pure LLM agentic path
DISABLE_TEMPLATE_FALLBACK = True
# Feature toggles
ENABLE_INVARIANTS = True  # can be turned off for debugging
USE_SIMPLE_LLM_CALL = True  # bypass TimeoutManager for main generation
# Configurable generation timeouts (seconds); can override via env vars
FIRST_ATTEMPT_TIMEOUT = float(os.environ.get("FIRST_ATTEMPT_TIMEOUT", 180))  # Increased to 180s for initial attempt
SECOND_ATTEMPT_TIMEOUT = float(os.environ.get("SECOND_ATTEMPT_TIMEOUT", 150))  # Increased to 150s for retry
THIRD_ATTEMPT_TIMEOUT = float(os.environ.get("THIRD_ATTEMPT_TIMEOUT", 120))   # Increased to 120s for final attempt
DEFAULT_LLM_REQUEST_TIMEOUT = float(os.environ.get("LLM_REQUEST_TIMEOUT", 150))

# Micro-verification timeouts
MICRO_VERIFY_MCP_TIMEOUT = float(os.environ.get("MICRO_VERIFY_MCP_TIMEOUT", 30))
MICRO_VERIFY_LLM_TIMEOUT = float(os.environ.get("MICRO_VERIFY_LLM_TIMEOUT", 60))

# Debug: Print timeout values to verify they're correct
print(f"[TIMEOUT_DEBUG] FIRST_ATTEMPT_TIMEOUT={FIRST_ATTEMPT_TIMEOUT}s")
print(f"[TIMEOUT_DEBUG] SECOND_ATTEMPT_TIMEOUT={SECOND_ATTEMPT_TIMEOUT}s")
print(f"[TIMEOUT_DEBUG] THIRD_ATTEMPT_TIMEOUT={THIRD_ATTEMPT_TIMEOUT}s")
STABLE_REUSE_LLM = True  # don't re-init on pure timeout, only on MemoryError
# Absolute safety thresholds to avoid system instability / BSOD risk
ABS_MIN_FREE_MB = float(os.environ.get("LLM_ABSOLUTE_MIN_FREE_MB", "200"))  # Lowered for GPU mode
SAFE_POST_QUERY_TARGET_MB = float(os.environ.get("LLM_SAFE_POST_QUERY_TARGET_MB", "300"))  # Lowered for GPU mode
# Placeholder tokens that should never appear in final executable SQL
PLACEHOLDER_TOKENS = {
    "your title", "some title", "sample title", "your table", "table_name",
    "column_name", "some_column", "placeholder", "value_here"
}
PLACEHOLDER_REGEXES = [
    r"your_[a-z0-9_]*",  # generic your_ placeholder tokens
]

# (Duplicate EMPLOYEE_DB_SCHEMA removed to prevent overwriting)

# Global readiness flag so we can re-attempt handshake before first real generation
LLM_HANDSHAKE_COMPLETE = False
LAST_GENERATED_SQL = None  # updated each time a final SQL statement is produced
# Cached real department names from DB — populated during initialize_database
# Fallback: known employee-DB departments (used if DB query fails)
_cached_dept_names: set = {
    'Customer Service', 'Development', 'Finance', 'Human Resources',
    'Marketing', 'Production', 'Quality Management', 'Research', 'Sales'
}

# Function to initialize LLM with better memory management
def initialize_llm(progressive_reduction=0):
    """
    Initialize the LLM model with Ollama with proper memory management.
    
    Args:
        progressive_reduction: Level of resource reduction (0=normal, 1=reduced, 2=minimal)
    
    Returns:
        An initialized Ollama LLM instance
    """
    # Make sure we have a clean slate
    gc.collect()
    
    # Get current memory stats
    stats = memory_manager.get_memory_stats()
    available_mb = stats["available_mb"]
    
    # Hard safety abort first
    if available_mb < ABS_MIN_FREE_MB:
        raise MemoryError(f"Refusing to initialize LLM: free {available_mb:.1f}MB < ABS_MIN_FREE_MB {ABS_MIN_FREE_MB}MB (system safety)")

    # Automatically determine if we need to increase resource reduction based on memory
    if available_mb < memory_manager.critical_mb * 1.5:
        # Force at least level 2 (minimal) if memory is near critical
        progressive_reduction = max(progressive_reduction, 2)
        print(f"Memory critically low ({available_mb:.2f}MB): Forcing minimal resource mode")
    elif available_mb < memory_manager.threshold_mb:
        # Force at least level 1 (reduced) if memory is below threshold
        progressive_reduction = max(progressive_reduction, 1)
        print(f"Memory low ({available_mb:.2f}MB): Using reduced resource mode")
    
    # Determine resource usage level based on available memory and progressive reduction
    # Allow environment overrides for mode thresholds
    try:
        env_ultra = float(os.environ.get("LLM_ULTRA_MB", "800"))
        env_min = float(os.environ.get("LLM_MIN_MB", "600"))
    except Exception:
        env_ultra, env_min = 400.0, 200.0
    ultra_low_memory_mode = available_mb < env_ultra or progressive_reduction >= 1
    minimal_mode = available_mb < env_min or progressive_reduction >= 2
    
    # Adjust parameters based on available memory
    # Allow explicit overrides so user can test larger context to reduce repeated token delays
    ctx_override = os.environ.get("LLM_CTX" )
    thread_override = os.environ.get("LLM_THREADS")
    try:
        # Set absolute floor at 512 to prevent "forgetting" the schema hints
        ctx_size = int(ctx_override) if ctx_override else (512 if minimal_mode else 768 if ultra_low_memory_mode else 1024)
    except Exception:
        ctx_size = 512
    
    try:
        num_thread = int(thread_override) if thread_override else 1
    except Exception:
        num_thread = 1
    timeout = 12.0 if minimal_mode else 18.0 if ultra_low_memory_mode else 25.0
    
    # Log mode based on memory constraints
    mode_desc = "minimal" if minimal_mode else "ultra-low" if ultra_low_memory_mode else "low"
    print(f"Initializing LLM with ctx_size={ctx_size}, num_thread={num_thread}, timeout={timeout:.1f}s in {mode_desc} memory mode")
    
    # Use smaller model if in minimal mode and memory is critically low
    # Permit explicit model override via env for experimentation (e.g. smaller model to prevent timeouts)
    model_name = os.environ.get("LLM_MODEL_OVERRIDE") or (
        FALLBACK_MODEL_NAME if minimal_mode and available_mb < memory_manager.critical_mb * 1.2 else OLLAMA_MODEL_NAME
    )
    
    # Create the LLM instance with optimized settings but do NOT register yet.
    # We perform a lightweight handshake first and then register only if memory allows.
    llm = None
    try:
        llm = Ollama(
            model=model_name,
            base_url=OLLAMA_BASE_URL,
            temperature=0.05,  # Restored from 7.7
            request_timeout=DEFAULT_LLM_REQUEST_TIMEOUT,
            additional_kwargs={
                # Optimized parameters for memory efficiency
                "num_ctx": ctx_size,
                "num_batch": 1,        # Restored from 7.7
                "num_gpu": -1,         # Keep GPU but with stable batching
                "f16_kv": True,       # Half-precision for key/value cache
                "mirostat": 0,        # Disable mirostat sampling
                "num_thread": num_thread,
                "seed": 42            # Consistent seed
            }
        )

        # Handshake strategy: if strict mode enabled, we block until success (within max timeout budget);
        # otherwise we warn and continue, with a later retry on first real query.
        strict_mode = os.environ.get("LLM_HANDSHAKE_STRICT", "0") not in ("0", "false", "False")
        base_timeout = float(os.environ.get("LLM_HANDSHAKE_TIMEOUT", "30"))  # single-attempt timeout
        max_total = float(os.environ.get("LLM_HANDSHAKE_MAX_TOTAL", "120"))  # total budget across attempts
        handshake_prompts = ["SELECT 1;", "-- warmup\nSELECT 1;"]
        attempt = 0
        start_all = time.time()
        handshake_ok = False
        while not handshake_ok and (time.time() - start_all) < max_total and attempt < len(handshake_prompts):
            hp = handshake_prompts[attempt]
            attempt += 1
            remaining_budget = max_total - (time.time() - start_all)
            this_timeout = min(base_timeout, remaining_budget)
            print(f"[HS] Handshake attempt {attempt} timeout={this_timeout:.1f}s remaining_budget={remaining_budget:.1f}s")
            try:
                # Use direct call bounded by TimeoutManager to avoid internal indefinite waits.
                resp = TimeoutManager.run_with_timeout(lambda: llm.complete(hp), timeout=this_timeout)
                # Some wrappers return object with .text, others raw string
                _txt = getattr(resp, 'text', None)
                if resp is not None and (_txt is None or 'error' not in str(_txt).lower()):
                    elapsed_all = time.time() - start_all
                    print(f"[HS] LLM handshake succeeded in {elapsed_all:.2f}s (model={model_name})")
                    handshake_ok = True
                    break
            except TimeoutError:
                print(f"[HS][WARN] Handshake attempt {attempt} timed out after {this_timeout:.1f}s")
            except Exception as e_hs:
                print(f"[HS][WARN] Handshake attempt {attempt} error: {e_hs}")
                # If model just started loading, give it a short pause before next attempt
                time.sleep(1.5)
        if not handshake_ok:
            msg = "[HS][WARN] Handshake not successful within budget; will retry on first real query." if not strict_mode else "[HS][ERROR] Strict handshake mode enabled and handshake failed." 
            print(msg)
            if strict_mode:
                # In strict mode, raise to stop startup (user explicitly wants guaranteed readiness)
                raise TimeoutError("Strict LLM handshake failed")
        else:
            # Mark readiness
            global LLM_HANDSHAKE_COMPLETE
            LLM_HANDSHAKE_COMPLETE = True

        # Check memory after initialization/handshake to ensure we didn't consume too much
        after_stats = memory_manager.get_memory_stats()
        memory_used = available_mb - after_stats.get("available_mb", 0)
        if memory_used > 100:  # If we used more than 100MB just for initialization
            print(f"⚠️ Warning: LLM initialization consumed {memory_used:.2f}MB")
        # Post-init safety check
        if after_stats.get('available_mb', 0) < ABS_MIN_FREE_MB:
            print(f"[SAFETY] Post-init free memory {after_stats.get('available_mb',0):.1f}MB < ABS_MIN_FREE_MB {ABS_MIN_FREE_MB}MB; tearing down instance")
            try:
                if hasattr(llm, 'close'): llm.close()
            except Exception:
                pass
            raise MemoryError("Post-init memory below absolute floor; aborted")

        # Attempt to register the instance with the memory manager now that it's healthy.
        try:
            memory_manager.register_llm_instance(llm)
        except MemoryError as me:
            # Registration refused due to low memory; leave the llm unregistered but return it
            print(f"MemoryManager refused to register LLM: {me}")
        except Exception as reg_err:
            # Non-fatal: log and continue returning the llm (unregistered)
            print(f"Warning: unexpected error registering LLM: {reg_err}")

        return llm

    except Exception:
        # Ensure we free any partial resources on failure
        try:
            if llm is not None:
                try:
                    memory_manager.unregister_llm_instance(llm)
                except Exception:
                    pass
        finally:
            # Best-effort GC and working set trim
            gc.collect()
            try:
                memory_manager.force_collect_garbage()
            except Exception:
                pass
        raise

# Create a function to run LLM safely with proper timeouts and cleanup
def run_llm_with_timeout(llm, prompt, timeout=30.0, retries=1):
    """
    Run an LLM with proper timeout handling and resource cleanup.
    
    Args:
        llm: The LLM instance to use
        prompt: The prompt to send to the LLM
        timeout: Timeout in seconds
        retries: Number of retries on timeout or memory error
        
    Returns:
        The LLM response
        
    Raises:
        TimeoutError: If all attempts time out
        MemoryError: If memory is insufficient after retries
    """
    last_error = None
    
    for attempt in range(retries + 1):
        print(f"[LLM][ATTEMPT] Attempt {attempt+1}/{retries+1} for prompt (len={len(prompt)})")
        # Check memory before attempt and log status
        stats = memory_manager.get_memory_stats()
        if stats['available_mb'] < ABS_MIN_FREE_MB:
            print(f"[LLM][MEMORY] Insufficient memory before attempt: {stats['available_mb']:.1f}MB < ABS_MIN_FREE_MB {ABS_MIN_FREE_MB}MB")
            raise MemoryError(f"Available memory {stats['available_mb']:.1f}MB < ABS_MIN_FREE_MB {ABS_MIN_FREE_MB}MB; aborting generation")
        print(f"Memory check before attempt {attempt+1}: {stats['available_mb']:.2f}MB available (ABS_MIN_FREE_MB={ABS_MIN_FREE_MB}MB)")
        
        if stats['available_mb'] < memory_manager.threshold_mb:
            # Try with more aggressive memory reduction
            reduction_level = attempt + 1
            print(f"[LLM][MEMORY] Memory low, creating LLM with reduction level {reduction_level}")
            
            # Release the current LLM instance if it exists
            if 'llm' in locals() and llm is not None:
                try:
                    memory_manager.unregister_llm_instance(llm)
                    llm = None
                except Exception as e:
                    print(f"Error unregistering LLM: {str(e)}")
                
                # Force garbage collection
                gc.collect()
                memory_manager.force_collect_garbage()
            
            # Check if memory is critically low
            stats = memory_manager.get_memory_stats()
            if stats['available_mb'] < memory_manager.critical_mb:
                print(f"🚨 CRITICAL MEMORY CONDITION: Only {stats['available_mb']:.2f}MB available!")
                raise MemoryError(f"System memory critically low: {stats['available_mb']:.2f}MB available")
            
            # Create a new LLM with more aggressive resource reduction
            llm = initialize_llm(progressive_reduction=reduction_level)
            
            # Register with memory manager
            memory_manager.register_llm_instance(llm)
        
        try:
            # Run the LLM with timeout using ThreadManager for safety
            print(f"[LLM][RUN] Running LLM (attempt {attempt+1}/{retries+1}) with {timeout:.1f}s timeout")

            # Start timing
            start_time = time.time()

            # First, try talking to an external MCP server if available. This centralizes
            # model & tool orchestration. If MCP is unreachable, fall back to local LLM.
            # Prefer local LLM first (preserves pre-MCP behaviour). If local LLM fails or times out,
            # attempt to use MCP as a fallback so an unavailable MCP won't break everything.
            with memory_manager.suspend_emergency("llm_generation"):
                result = None  # Initialize result to avoid UnboundLocalError
                try:
                    print("Attempting local LLM completion first...")
                    # Wrap the LLM call to ensure we pass prompt correctly and bound it with our TimeoutManager
                    result = TimeoutManager.run_with_timeout(lambda: llm.complete(prompt), timeout=timeout)
                except TimeoutError as local_to:
                    print(f"Local LLM completion timed out after {timeout}s: {local_to}; attempting MCP fallback...")
                    try:
                        from .mcp_client import mcp_complete
                        mcp_text = mcp_complete(prompt, timeout=min( max(5.0, timeout*0.5 ), timeout ))
                        class _Resp:
                            def __init__(self, text):
                                self.text = text
                        result = _Resp(mcp_text)
                        print(f"MCP fallback succeeded after local LLM timeout")
                    except Exception as mcp_err:
                        print(f"MCP completion failed or unavailable: {mcp_err}")
                        # If MCP is clearly unreachable (connection refused), don't treat as critical
                        if "connection refused" in str(mcp_err).lower() or "max retries exceeded" in str(mcp_err).lower():
                            print("[MCP] Server appears to be down - continuing without MCP fallback")
                        # Re-raise the original timeout error for retry logic
                        raise local_to
                except Exception as local_err:
                    print(f"Local LLM completion failed: {local_err}; attempting MCP fallback...")
                    try:
                        from .mcp_client import mcp_complete
                        mcp_text = mcp_complete(prompt, timeout=min( max(5.0, timeout*0.5 ), timeout ))
                        class _Resp:
                            def __init__(self, text):
                                self.text = text
                        result = _Resp(mcp_text)
                        print(f"MCP fallback succeeded after local LLM failure")
                    except Exception as mcp_err:
                        print(f"MCP completion failed or unavailable: {mcp_err}")
                        # If MCP is clearly unreachable (connection refused), don't treat as critical
                        if "connection refused" in str(mcp_err).lower() or "max retries exceeded" in str(mcp_err).lower():
                            print("[MCP] Server appears to be down - continuing without MCP fallback")
                        # For timeout errors, try a longer timeout retry instead of immediate failure
                        if isinstance(local_err, TimeoutError) and attempt < retries:
                            print(f"[RETRY] Will retry with longer timeout due to initial timeout")
                            # Don't re-raise immediately, let the retry logic handle it
                            raise local_err
                        else:
                            # Re-raise original local error to be handled by outer retry logic
                            raise local_err
                            
            # Log completion time
            elapsed = time.time() - start_time
            print(f"LLM completed in {elapsed:.2f}s")

            # Check memory after completion
            stats = memory_manager.get_memory_stats()
            print(f"Memory after LLM: {stats['available_mb']:.2f}MB available")
            # If after execution we are below safety target, proactively free (soft protection)
            if stats['available_mb'] < SAFE_POST_QUERY_TARGET_MB:
                try:
                    print(f"[SAFETY] Free memory {stats['available_mb']:.1f}MB < SAFE_POST_QUERY_TARGET_MB {SAFE_POST_QUERY_TARGET_MB}MB; releasing LLM to protect system")
                    memory_manager.unregister_llm_instance(llm)
                except Exception:
                    pass
                memory_manager.force_collect_garbage()

            # Force cleanup to prevent memory leaks
            memory_manager.force_collect_garbage()

            return result
            
        except (TimeoutError, MemoryError) as e:
            last_error = e
            print(f"[LLM][FAIL] Attempt {attempt+1} failed: {type(e).__name__}: {str(e)}")
            if isinstance(e, TimeoutError):
                print(f"[LLM][TIMEOUT] Timeout occurred for prompt (len={len(prompt)}). Consider simplifying the query or increasing system resources.")
            
            # Unregister LLM from memory manager
            try:
                memory_manager.unregister_llm_instance(llm)
            except Exception as unreg_error:
                print(f"Error unregistering LLM: {str(unreg_error)}")
            
            # Aggressive cleanup after error
            for _ in range(3):  # Multiple cleanup attempts
                gc.collect()
                time.sleep(0.1)  # Brief pause to allow OS to reclaim memory
            
            memory_manager.force_collect_garbage()
            
            if attempt < retries:
                # INCREASE timeout for subsequent attempts to handle initialization overhead
                timeout = min(180.0, timeout * 1.5)  # Increase by 50% each retry, cap at 3 minutes
                print(f"[LLM][RETRY] Retrying with increased timeout={timeout:.1f}s (attempt {attempt+2})")
                # Escalate LLM reduction after 2 timeouts
                if isinstance(e, TimeoutError) and attempt >= 1:
                    print(f"[LLM][ESCALATE] Escalating to more aggressive LLM reduction after repeated timeouts.")
                    try:
                        llm = initialize_llm(progressive_reduction=attempt+2)
                        memory_manager.register_llm_instance(llm)
                    except Exception as esc_err:
                        print(f"[LLM][ESCALATE][FAIL] Could not escalate LLM: {esc_err}")
            else:
                # Last attempt failed
                print("[LLM][FAIL] All LLM attempts failed for this prompt.")
                print(f"[LLM][USER] The LLM failed to respond after multiple attempts. Please try a simpler question, check your system resources, or increase the timeout settings if possible.")
                # Final cleanup before raising error
                memory_manager.force_collect_garbage()
                raise last_error
        
        except Exception as e:
            # For other exceptions, cleanup and don't retry
            print(f"LLM error: {type(e).__name__}: {str(e)}")
            
            # Unregister LLM from memory manager
            try:
                memory_manager.unregister_llm_instance(llm)
            except Exception as unreg_error:
                print(f"Error unregistering LLM: {str(unreg_error)}")
            
            # Cleanup
            memory_manager.force_collect_garbage()
            
            raise

# Initialize LLM — guard against double-import (Agent.agentic_workflow vs agentic_workflow)
# Both module paths share the same Ollama process, so re-running initialize_llm() causes
# redundant handshakes and wastes ~440MB of KV-cache allocation each time.
_LLM_INIT_SENTINEL = "Agent.__agentic_workflow_llm_initialized__"
import sys as _sys
if _sys.modules.get(_LLM_INIT_SENTINEL) is None:
    _sys.modules[_LLM_INIT_SENTINEL] = True  # mark as done before call (prevents re-entry)
    llm = initialize_llm()
    Settings.llm = llm
else:
    # Module already initialized in this process — reuse the existing LLM instance
    print("[LLM_INIT] Skipping duplicate initialize_llm() — module already initialized in this process")
    try:
        # Settings.llm should already be set; fall back to a fresh init only if truly missing
        if getattr(Settings, 'llm', None) is None:
            llm = initialize_llm()
            Settings.llm = llm
        else:
            llm = Settings.llm
    except Exception:
        llm = initialize_llm()
        Settings.llm = llm


def get_salary_column_name() -> str:
    """Helper to determine if the salary column is named 'salary' or 'amount'."""
    global inspector, available_tables
    if not inspector or 'salary' not in available_tables:
        return 'salary'  # default fallback
    try:
        cols = [c['name'].lower() for c in inspector.get_columns(table_name='salary')]
        if 'salary' in cols:
            return 'salary'
        if 'amount' in cols:
            return 'amount'
        return cols[0] if cols else 'salary'
    except Exception:
        return 'salary'


async def initialize_database(uri: Optional[str] = None) -> Dict[str, Any]:
    """Initialize the database connection and return metadata."""
    global available_tables, inspector, sql_database, async_engine, async_session
    metadata = {
        "uri": None,
        "tables": [],
        "sql_database": None,
        "inspector": None
    }
    
    try:
        # Use provided URI
        metadata["uri"] = uri
        
        # Remove sslmode parameter if present
        if 'postgresql' in metadata["uri"] and 'sslmode=' in metadata["uri"]:
            base_uri = metadata["uri"].split('?')[0]
            metadata["uri"] = base_uri
            print(f"Cleaned DB URL: {metadata['uri']}")
        
        # Create sync engine for inspector
        sync_uri = metadata["uri"]
        if 'postgresql' in sync_uri:
            sync_uri = sync_uri.replace('postgresql://', 'postgresql://')
        
        metadata["sql_database"] = create_engine(sync_uri)
        sql_database = metadata["sql_database"]
        
        # Create async engine for queries
        if 'postgresql' in metadata["uri"]:
            async_uri = metadata["uri"].replace('postgresql://', 'postgresql+asyncpg://')
        else:
            async_uri = metadata["uri"]
            
        async_engine = create_async_engine(async_uri)
        
        # Create async session factory
        async_session = sessionmaker(
            async_engine, class_=AsyncSession, expire_on_commit=False
        )
        
        # Create inspector
        metadata["inspector"] = inspect(metadata["sql_database"])
        inspector = metadata["inspector"]
        
        # Get available tables
        available_tables = metadata["inspector"].get_table_names() + metadata["inspector"].get_view_names()
        
        # Initialize tables
        for table in available_tables:
            metadata_obj = MetaData()
            try:
                table_obj = Table(table, metadata_obj, autoload_with=metadata["sql_database"])
                metadata["tables"].append(table_obj)
                print(f"Initialized table: {table}")
            except NoSuchTableError as e:
                print(f'"{e}" table not found!')
                continue
        
        return metadata
        
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        return metadata

def list_tables() -> str:
    """Returns a list of available tables in the database."""
    global available_tables, inspector
    try:
        table_list = []
        for table_name in available_tables:
            table_info = inspector.get_table_comment(table_name=table_name).get('text', '')
            table_info = f'Table Description: {table_info}\n' if table_info else ''
            table_list.append(f'Table Name: {table_name}\n{table_info}')
        
        result = '\n'.join(table_list)
        return result
    except Exception as e:
        return f"Error listing tables: {str(e)}"

def describe_tables(tables: Optional[List[str]] = None) -> str:
    """Describes the specified tables in the database."""
    global available_tables, inspector, sql_database
    try:
        table_names = tables or available_tables
        table_schemas = []

        for table_name in table_names:
            try:
                # Create table object
                metadata_obj = MetaData()
                table = Table(table_name, metadata_obj, autoload_with=sql_database)
                
                # Get schema
                schema = f"{table_name} columns:\n"
                
                # Get columns
                columns = inspector.get_columns(table_name=table_name)
                for column in columns:
                    col_name = column["name"]
                    col_type = str(column["type"])
                    schema += f"- {col_name}: {col_type}\n"
                
                table_schemas.append(f"{schema}")
            except NoSuchTableError as e:
                print(f"Table '{table_name}' not found: {str(e)}")
                continue

        if not table_schemas:
            return "No valid tables found to describe."

        result = "\n".join(table_schemas)
        return result
    except Exception as e:
        return f"Error describing tables: {str(e)}"

async def execute_query(query: str, skip_invariants: bool = False) -> List[Dict[str, Any]]:
    """Execute an SQL query and return the results as a list of dictionaries."""
    global async_session

    def sanitize_sql(raw: str) -> str:
        """Extract clean SQL, remove wrappers, and conservatively extract the first SQL statement (SELECT/WITH)."""
        import re
        s = extract_clean_sql(raw)
        if not s or not isinstance(s, str):
            return s
        s = s.strip()
        m = re.search(r"\b(select|with)\b", s, flags=re.IGNORECASE)
        if not m:
            return s
        candidate = s[m.start():]
        lines = candidate.splitlines()
        kept = []
        for ln in lines:
            stripped = ln.strip()
            if not stripped:
                kept.append(ln)
                continue
            # Stop at bullet/annotation lines or obvious assistant notes
            if re.match(r'^[-]{1,}', stripped):
                break
            if re.match(r'^(User Question:|Previous Attempt:|Use correct|Do NOT|Error executing|Result Presenter Agent)', stripped, flags=re.IGNORECASE):
                break
            kept.append(ln)
        cleaned = '\n'.join(kept).strip()
        # If the cleaned SQL looks like a bare SELECT of columns (no FROM), try to recover a FROM clause
        if cleaned and not re.search(r"\bfrom\b", cleaned, flags=re.IGNORECASE):
            fm = re.search(r"\bfrom\b[\s\S]*?(;|$)", candidate, flags=re.IGNORECASE)
            if fm:
                from_clause = fm.group(0).strip()
                if re.search(r"^\s*select\b", from_clause, flags=re.IGNORECASE):
                    subfm = re.search(r"\bfrom\b[\s\S]*?(;|$)", from_clause, flags=re.IGNORECASE)
                    if subfm:
                        from_clause = subfm.group(0).strip()
                cleaned = cleaned.rstrip(';').strip() + '\n' + from_clause
            else:
                return ""
        if cleaned and not cleaned.rstrip().endswith(';'):
            cleaned = cleaned + ';'
        return cleaned

    try:
        sanitized = sanitize_sql(query)
        if sanitized != query:
            print("Notice: sanitizing SQL before execution (removed LLM commentary)")
            print(f"Sanitized SQL: {sanitized}")
        print(f"Executing query: {sanitized}")
        async with async_session() as session:
            # Ensure we actually have a SELECT/WITH to run
            if not sanitized or not re.search(r"\b(select|with)\b", sanitized, flags=re.IGNORECASE):
                return [{"error": "No valid SQL found after sanitization"}]
            # Be conservative: SELECT statements that lack a FROM clause are likely
            # partial outputs from the LLM (e.g. column list only). Reject these
            # unless they are a simple literal/select expression (e.g. SELECT 1;)
            if re.search(r'^\s*select\b', sanitized, flags=re.IGNORECASE) and not re.search(r"\bfrom\b", sanitized, flags=re.IGNORECASE):
                # Allow very small literal selects like `SELECT 1;` or `SELECT 'x';`
                if not re.match(r"^\s*select\s+[0-9'\"]+\s*;\s*$", sanitized, flags=re.IGNORECASE):
                    return [{"error": "Sanitized SQL missing FROM clause; rejected to avoid invalid execution"}]
            result = await session.execute(text(sanitized))
            rows = result.fetchall()
            
            # Convert to list of dictionaries
            if rows and result.keys():
                column_names = result.keys()
                return [dict(zip(column_names, row)) for row in rows]
            elif result.keys():  # We have columns but no rows
                column_names = result.keys()
                print(f"Query executed successfully but returned no rows. Columns: {column_names}")
                return []
            else:  # No rows and potentially no columns
                print("Query executed but returned no results")
                return []
    except Exception as e:
        error_message = str(e)
        print(f"Error executing query: {error_message}")
        
        # Extract error details for better debugging
        error_type = type(e).__name__
        print(f"Error type: {error_type}")
        
        # Check for specific error types and provide more context
        if "column" in error_message.lower() and "does not exist" in error_message.lower():
            print("Column reference error detected - this may be a schema misunderstanding")
        elif "syntax error" in error_message.lower():
            print("SQL syntax error detected - the query structure is invalid")
        elif "relation" in error_message.lower() and "does not exist" in error_message.lower():
            print("Table reference error detected - the table may not exist or has a different name")

        # AUTO-FIX: Detect PostgreSQL GroupingError (missing GROUP BY) and repair inline
        if ("groupingerror" in error_type.lower() or
                "must appear in the group by clause" in error_message.lower()):
            print("[AUTO-FIX][GROUP_BY] Detected PostgreSQL GroupingError — attempting GROUP BY repair")
            try:
                col_match = re.search(
                    r'column\s+"?([^"]+)"?\s+must appear in the group by',
                    error_message, re.IGNORECASE
                )
                fixed_query = sanitized
                if col_match:
                    raw_col = col_match.group(1).strip()        # e.g. "title.title"
                    col_bare = raw_col.split('.')[-1]           # e.g. "title"
                    insertion = len(fixed_query)
                    for clause in [r'\bORDER\s+BY\b', r'\bLIMIT\b']:
                        m = re.search(clause, fixed_query, re.IGNORECASE)
                        if m and m.start() < insertion:
                            insertion = m.start()
                    semi = re.search(r'\s*;', fixed_query)
                    if semi and semi.start() < insertion:
                        insertion = semi.start()
                    if 'GROUP BY' in fixed_query.upper():
                        fixed_query = re.sub(
                            r'\b(GROUP\s+BY\b\s+\S+)',
                            rf'\1, {col_bare}',
                            fixed_query, count=1, flags=re.IGNORECASE
                        )
                    else:
                        fixed_query = fixed_query[:insertion] + f' GROUP BY {col_bare}' + fixed_query[insertion:]
                    print(f"[AUTO-FIX][GROUP_BY] Repaired: {fixed_query[:200]}")
                    async with async_session() as session:
                        result = await session.execute(text(fixed_query))
                        rows = result.fetchall()
                        if rows and result.keys():
                            return [dict(zip(result.keys(), row)) for row in rows]
                        elif result.keys():
                            return []
                        return []
            except Exception as gb_err:
                print(f"[AUTO-FIX][GROUP_BY] Repair failed: {gb_err}")

        # NEW: Detect missing FROM-clause/alias errors and trigger LLM regeneration
        if not skip_invariants and ("missing from-clause entry for table" in error_message.lower() or "undefinedtableerror" in error_type.lower() or "missing from-clause entry" in error_message.lower() or "no such table" in error_message.lower() or "invalid reference to FROM-clause entry" in error_message.lower() or "missing alias" in error_message.lower() or "ambiguous column" in error_message.lower() or "alias" in error_message.lower() and "does not exist" in error_message.lower()):
            print("[INVARIANTS] Detected alias/FROM-clause error. Triggering LLM regeneration for correct join order and alias usage.")
            try:
                from .invariants import build_invariants, evaluate_invariants, regenerate_with_constraints, classify_question
                # Use the last question from context if available, else fallback
                # (You may need to pass the question explicitly if not in scope)
                regen_question = getattr(execute_query, 'last_question', None)
                if not regen_question:
                    regen_question = "Please regenerate the SQL with correct join order and alias usage."
                cat = classify_question(regen_question)
                invs = build_invariants(cat, regen_question)
                missing = evaluate_invariants(query, invs)
                if missing:
                    regen_sql = regenerate_with_constraints(regen_question, query, missing, Settings.llm, run_llm_with_timeout, attempt=0)
                    if regen_sql:
                        print("[INVARIANTS] LLM regenerated SQL after alias/FROM-clause error. Re-executing.")
                        # Try to execute the regenerated SQL
                        return await execute_query(regen_sql)
            except Exception as regen_alias_err:
                print(f"[INVARIANTS] Alias/FROM-clause regeneration failed: {regen_alias_err}")
        # If error indicates a missing column, attempt a lightweight heuristic repair and re-run once
        col_match = re.search(r'column\s+"?([a-zA-Z0-9_]+)"?\s+does not exist', error_message, flags=re.IGNORECASE)
        if col_match:
            missing_col = col_match.group(1)
            print(f"Detected missing column '{missing_col}', attempting heuristic repair...")
            repaired = repair_missing_column(query, missing_col)
            if repaired:
                print(f"Repaired query to attempt: {repaired}")
                try:
                    async with async_session() as session:
                        result = await session.execute(text(repaired))
                        rows = result.fetchall()
                        if rows and result.keys():
                            column_names = result.keys()
                            return [dict(zip(column_names, row)) for row in rows]
                        elif result.keys():
                            return []
                        else:
                            return []
                except Exception as re_e:
                    print(f"Repaired query execution failed: {re_e}")
                    return [{"error": str(re_e)}]
        # Return the error for formatting
        return [{"error": error_message}]

def validate_sql_query(query: str) -> Dict[str, Any]:
    """Validate a SQL query without executing it."""
    try:
        # Always clean SQL before validation
        sql = extract_clean_sql(query)
        ql = sql.lower()
        # Use regex to check for SELECT and FROM as word boundaries (robust to newlines/indentation)
        if not re.search(r'\bselect\b', ql) and not re.search(r'\bwith\b', ql):
            return {
                "valid": False,
                "reason": "Query must start with SELECT or WITH"
            }
        if not re.search(r'\bfrom\b', ql):
            return {
                "valid": False,
                "reason": "Query must include FROM clause"
            }
        is_pure_aggregate = (
            ("count(" in ql or "avg(" in ql or "min(" in ql or "max(" in ql or "sum(" in ql)
            and " group by " not in ql
        )
        if ("limit" not in ql) and not is_pure_aggregate:
            return {"valid": False, "reason": "Query should include LIMIT clause to prevent excessive results"}
        return {
            "valid": True,
            "reason": "Query syntax appears valid"
        }
    except Exception as e:
        return {
            "valid": False,
            "reason": f"Validation exception: {str(e)}"
        }

def repair_missing_column(query: str, missing_col: str) -> Optional[str]:
    """Attempt a small, safe repair when a column is reported missing.
    Heuristic currently supports the common case where `salary` is referenced
    but belongs to the `salary` table rather than `employee`.
    Returns a repaired SQL string (including LIMIT) or None if no repair.
    """
    try:
        q = query.strip()
        q_lower = q.lower()
        # Find FROM clause and optional alias (e.g. FROM employee e)
        m = re.search(r'\bfrom\s+([a-zA-Z_][\w]*)(?:\s+([a-zA-Z_][\w]*))?', q, flags=re.IGNORECASE)
        if not m:
            return None
        base_table = m.group(1)
        base_alias = m.group(2)

        # Avoid treating SQL keywords as aliases (e.g., 'ORDER' accidentally captured)
        SQL_KEYWORDS = {"order","by","where","group","limit","join","on","having","union","except","intersect"}
        if base_alias and base_alias.lower() in SQL_KEYWORDS:
            base_alias = None

        alias = base_alias if base_alias else base_table

        # Only handle the common salary case conservatively
        if missing_col.lower() == 'salary':
            # Avoid double-injecting if salary already joined or qualified
            if re.search(r'\bjoin\s+salary\b', q_lower) or re.search(r's\.salary|salary\s+\w', q_lower):
                return None

            # Find end of the table name token to insert join before the next clause
            table_token_re = re.search(r'\bfrom\s+' + re.escape(base_table) + r'\b', q, flags=re.IGNORECASE)
            if not table_token_re:
                return None
            insert_pos = table_token_re.end()
            join_clause = f" JOIN salary s ON {alias}.emp_no = s.emp_no"
            repaired = q[:insert_pos] + join_clause + q[insert_pos:]

            # Qualify ORDER BY salary -> s.salary
            repaired = re.sub(r'order\s+by\s+salary\b', 'ORDER BY s.salary', repaired, flags=re.IGNORECASE)

            # Ensure terminating semicolon and LIMIT presence
            if 'limit' not in repaired.lower():
                if repaired.strip().endswith(';'):
                    repaired = repaired[:-1].strip() + ' LIMIT 100;'
                else:
                    repaired = repaired.strip() + ' LIMIT 100;'
            else:
                if not repaired.strip().endswith(';'):
                    repaired = repaired.strip() + ';'

            return repaired

        return None
    except Exception:
        return None

def enhanced_validate_sql(query: str) -> Dict[str, Any]:
    """Stricter validation to catch placeholders, unknown tables/columns.
    Returns dict(valid: bool, issues: List[str]). Lightweight heuristic (no full parse).
    """
    issues: List[str] = []
    q_lower = query.lower()

    # Reject obvious placeholders
    for token in PLACEHOLDER_TOKENS:
        if token in q_lower:
            issues.append(f"Contains placeholder token '{token}'")
    for rx in PLACEHOLDER_REGEXES:
        if re.search(rx, q_lower):
            issues.append("Contains generic placeholder pattern (your_*)")

    # Must contain SELECT ... FROM ... pattern (robust to newlines/indentation)
    if not re.search(r'\bselect\b', q_lower) or not re.search(r'\bfrom\b', q_lower):
        issues.append("Missing SELECT or FROM clause")

    # Extract referenced tables (rough pattern)
    referenced_tables = set()
    try:
        table_tokens = re.findall(r'(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\s+as)?\s*([a-zA-Z_][a-zA-Z0-9_]*)?', query, flags=re.IGNORECASE)
        for base, _alias in table_tokens:
            referenced_tables.add(base.lower())
    except Exception:
        pass

    avail_lower = [t.lower() for t in available_tables]
    for t in referenced_tables:
        if t and t not in avail_lower:
            issues.append(f"Unknown table '{t}'")

    # Build schema cache once
    schema_cache = {}
    try:
        for tbl in available_tables:
            try:
                cols = [c['name'].lower() for c in inspector.get_columns(table_name=tbl)]
                schema_cache[tbl.lower()] = set(cols)
            except Exception:
                continue
    except Exception:
        pass

    # Alias mapping
    alias_map = {}
    try:
        alias_defs = re.findall(r'(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:as\s+)?([a-zA-Z_][a-zA-Z0-9_]*)', query, flags=re.IGNORECASE)
        for real, al in alias_defs:
            alias_map[al.lower()] = real.lower()
    except Exception:
        pass

    # Column references alias.column
    try:
        col_refs = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)', query)
        # Skip common SQL functions/keywords that could appear before a dot (edge cases)
        skip_alias = {"select","where","and","or","on","group","order","limit","count","avg","sum","min","max"}
        for alias, col in col_refs:
            if alias.lower() in skip_alias:
                continue
            mapped_table = alias_map.get(alias.lower(), alias.lower())
            cols = schema_cache.get(mapped_table)
            if cols is not None and col.lower() not in cols:
                issues.append(f"Unknown column '{alias}.{col}'")
    except Exception:
        pass

    # Detect bare (unqualified) column names used in WHERE / SELECT that don't exist in any referenced table
    try:
        # Extract simple bare identifiers between SELECT and FROM and after WHERE
        bare_tokens = set()
        select_part = query.split('FROM')[0] if 'FROM' in query.upper() else query
        where_parts = re.split(r'WHERE', query, flags=re.IGNORECASE)
        if len(where_parts) > 1:
            where_clause = where_parts[1]
        else:
            where_clause = ''
            token_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
            keywords = {"select","from","where","and","or","on","group","by","order","limit","count","avg","sum","min","max","as","distinct","case","when","then","end"}
            for part in [select_part, where_clause]:
                for m in re.findall(token_pattern, part):
                    ml = m.lower()
                    if ml not in keywords and not any(ml == a or ml in schema_cache.get(a, set()) for a in alias_map.keys()):
                        # Check if token is a column in ANY referenced table
                        if not any(ml in cols for cols in schema_cache.values()):
                            bare_tokens.add(m)
        # Filter obviously numeric or alias tokens
        for tok in list(bare_tokens):
            if tok.isdigit():
                bare_tokens.remove(tok)
        # Report tokens that look like columns but unknown
        for tok in sorted(bare_tokens):
            # Skip if it is a table name
            if tok.lower() in avail_lower:
                continue
            # Skip placeholder patterns already captured
            if tok.lower().startswith('your_'):
                continue
            # Only flag if appears near '=' in query (likely used as column)
            if re.search(rf'\b{tok}\b\s*=|=\s*\b{tok}\b', query):
                issues.append(f"Unqualified unknown identifier '{tok}'")
    except Exception:
        pass

    # LIMIT optional for pure single-row aggregates (no GROUP BY)
    is_pure_aggregate = (
        ("count(" in q_lower or "avg(" in q_lower or "min(" in q_lower or "max(" in q_lower or "sum(" in q_lower)
        and " group by " not in q_lower
    )
    if "limit" not in q_lower and not is_pure_aggregate:
        issues.append("Missing LIMIT clause")
    if not q_lower.strip().endswith(';'):
        issues.append("Missing terminating semicolon")

    return {"valid": len(issues) == 0, "issues": issues}

def format_results(results: List[Dict[str, Any]], question: str) -> str:
    """Format query results into a user-friendly response."""
    if not results:
        return "I couldn't find any data matching your query in the employee database."
    
    if "error" in results[0]:
        error_msg = results[0]["error"]
        # Provide more specific error messages for common errors
        if "column" in error_msg.lower() and "does not exist" in error_msg.lower():
            column_match = re.search(r'column "([^"]+)" does not exist', error_msg.lower())
            if column_match:
                bad_column = column_match.group(1)
                return f"I encountered an error with the database query. The column '{bad_column}' doesn't exist in the database schema. This might be due to a typo or incorrect schema understanding."
        elif "relation" in error_msg.lower() and "does not exist" in error_msg.lower():
            table_match = re.search(r'relation "([^"]+)" does not exist', error_msg.lower())
            if table_match:
                bad_table = table_match.group(1)
                return f"I encountered an error with the database query. The table '{bad_table}' doesn't exist in the database. Available tables are: {', '.join(available_tables)}."
        
        # Generic error message
        return f"I encountered an error while querying the database: {error_msg}"
    
    # Convert any Decimal objects to float for JSON serialization
    processed_results = []
    for row in results:
        processed_row = {}
        for key, value in row.items():
            # Handle Decimal type for JSON serialization
            if hasattr(value, '__class__') and value.__class__.__name__ == 'Decimal':
                processed_row[key] = float(value)
            else:
                processed_row[key] = value
        processed_results.append(processed_row)
    
    # Handle all result sets directly without using LLM to save memory
    # First, check memory before potential LLM usage
    if not check_memory_safety():
        print("Low memory detected, using direct formatting instead of LLM")
        return format_results_direct(processed_results, question)
    
    # For result sets up to 50 rows, format directly (avoids LLM narrating from 3-sample rows)
    if len(processed_results) <= 50:
        return format_results_direct(processed_results, question)
    
    # For larger result sets, try to use LLM but have a fallback ready
    try:
        # Check memory before LLM call
        if not check_memory_safety():
            print("Low memory before LLM formatting, falling back to direct formatting")
            return format_results_direct(processed_results, question)
            
        # Extract just the first 3 rows to reduce context size
        sample_results = processed_results[:3]
        sample_str = str(sample_results)
        
        # Create a minimal prompt
        prompt = f"""Summarize database results for: "{question}"
Sample (first 3 of {len(results)} rows):
{sample_str}
Start with "Based on the employee database..."
"""
        
        # Generate the formatted response with a short timeout
        # Use a fresh LLM instance to avoid state issues
        format_llm = initialize_llm()
        response = format_llm.complete(prompt, timeout=5.0)
        return response.text.strip()
    except Exception as e:
        print(f"Error formatting results with LLM: {str(e)}")
        # Fallback to direct formatting
        return format_results_direct(processed_results, question)

def format_results_direct(processed_results: List[Dict[str, Any]], question: str) -> str:
    """
    Format results directly without using an LLM.
    This is a memory-efficient fallback when the LLM approach isn't possible.
    """
    question_lower = question.lower()
    
    # Start with a generic intro
    summary = "Based on the employee database, "
    
    # Determine the type of query to customize the response
    # Gender distribution queries — always show a table
    result_columns = list(processed_results[0].keys()) if processed_results else []
    has_gender_col = any('gender' in c.lower() for c in result_columns)
    if has_gender_col or 'gender' in question_lower or 'male' in question_lower or 'female' in question_lower:
        summary += "here is the gender breakdown:\n\n"
        headers = result_columns
        summary += "| " + " | ".join(headers) + " |\n"
        summary += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in processed_results:
            values = []
            for header in headers:
                val = row.get(header, "")
                if isinstance(val, (int, float)) and any(k in header.lower() for k in ["count", "num", "total"]):
                    values.append(f"{val:,}")
                else:
                    values.append(str(val))
            summary += "| " + " | ".join(values) + " |\n"
        return summary
    
    if "average salary" in question_lower and "department" in question_lower:
        summary += "here's the average salary by department:\n\n"
        
        # Create a markdown table for department salaries
        headers = list(processed_results[0].keys())
        summary += "| " + " | ".join(headers) + " |\n"
        summary += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        
        for row in processed_results:
            values = []
            for header in headers:
                val = row.get(header, "")
                # Format salary numbers with dollar signs and commas
                if isinstance(val, (int, float)) and "salary" in header.lower():
                    values.append(f"${val:,.2f}")
                else:
                    values.append(str(val))
            summary += "| " + " | ".join(values) + " |\n"
            
    elif "count" in question_lower or "how many" in question_lower or "number of" in question_lower:
        # For count queries
        if len(processed_results) == 1 and len(processed_results[0]) == 1:
            # Simple count result (like total number of employees)
            key = list(processed_results[0].keys())[0]
            count = processed_results[0][key]
            
            if "department" in question_lower and "employee" in question_lower:
                # Extract department name from question if possible
                dept_match = re.search(
                    r'(?:in|for|of|within)\s+(?:the\s+)?([A-Za-z\s&]+?)\s+department',
                    question_lower
                )
                dept_name = dept_match.group(1).strip().title() if dept_match else "that department"
                summary += f"there are {count:,} employees working in the {dept_name} department."
            elif "employee" in question_lower:
                summary += f"there are {count:,} employees in the database."
            else:
                summary += f"the count is {count:,}."
        else:
            # Multiple rows with counts (like employees per department)
            summary += "here are the counts:\n\n"
            
            # Create a markdown table
            headers = list(processed_results[0].keys())
            summary += "| " + " | ".join(headers) + " |\n"
            summary += "| " + " | ".join(["---"] * len(headers)) + " |\n"
            
            for row in processed_results:
                values = []
                for header in headers:
                    val = row.get(header, "")
                    # Format count numbers with commas
                    if isinstance(val, (int, float)) and ("count" in header.lower() or header.lower() == "count"):
                        values.append(f"{val:,}")
                    else:
                        values.append(str(val))
                summary += "| " + " | ".join(values) + " |\n"
    
    elif "manager" in question_lower or "managers" in question_lower:
        # For manager queries
        summary += "here are the department managers:\n\n"
        
        # Create a markdown table
        headers = list(processed_results[0].keys())
        summary += "| " + " | ".join(headers) + " |\n"
        summary += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        
        for row in processed_results:
            values = []
            for header in headers:
                val = row.get(header, "")
                values.append(str(val))
            summary += "| " + " | ".join(values) + " |\n"
    
    else:
        # Generic handling for other types of queries
        if len(processed_results) <= 10:
            summary += "here are the results:\n\n"
            
            # Create a markdown table
            headers = list(processed_results[0].keys())
            summary += "| " + " | ".join(headers) + " |\n"
            summary += "| " + " | ".join(["---"] * len(headers)) + " |\n"
            
            for row in processed_results:
                values = []
                for header in headers:
                    val = row.get(header, "")
                    # Format currency values
                    if isinstance(val, (int, float)) and "salary" in header.lower():
                        values.append(f"${val:,.2f}")
                    # Format count values
                    elif isinstance(val, (int, float)) and ("count" in header.lower() or header.lower() == "count"):
                        values.append(f"{val:,}")
                    else:
                        values.append(str(val))
                summary += "| " + " | ".join(values) + " |\n"
        else:
            # For large result sets
            summary += f"I found {len(processed_results):,} results. Here are the first 10:\n\n"
            
            # Create a markdown table with just the first 10 rows
            headers = list(processed_results[0].keys())
            summary += "| " + " | ".join(headers) + " |\n"
            summary += "| " + " | ".join(["---"] * len(headers)) + " |\n"
            
            for row in processed_results[:10]:
                values = []
                for header in headers:
                    val = row.get(header, "")
                    # Format currency values
                    if isinstance(val, (int, float)) and "salary" in header.lower():
                        values.append(f"${val:,.2f}")
                    # Format count values
                    elif isinstance(val, (int, float)) and ("count" in header.lower() or header.lower() == "count"):
                        values.append(f"{val:,}")
                    else:
                        values.append(str(val))
                summary += "| " + " | ".join(values) + " |\n"
    
    return summary

def check_history(question: str) -> Dict[str, Any]:
    """Check if a similar question has been asked before."""
    # 1. Direct normalized lookup
    norm = normalize_question(question)
    ans = question_memory_index.get(norm)
    if ans:
        return {"found": True, "previous_question": question, "previous_answer": ans}

    # 2. Lightweight fuzzy / token overlap search over recent history
    try:
        private_messages = chat_store_private.get_messages(key="conversation")
        qa_pairs = []
        cur_q = None
        for msg in private_messages[-400:]:  # limit scan window
            if msg.role == MessageRole.USER:
                cur_q = msg.content
            elif msg.role == MessageRole.ASSISTANT and cur_q:
                qa_pairs.append((cur_q, msg.content))
                cur_q = None
        qa_pairs.reverse()
        import re
        target_tokens = set(norm.split())
        best = None
        best_score = 0.0
        threshold = float(os.environ.get("MEMORY_FUZZY_THRESHOLD", "0.6"))
        for qtext, answer in qa_pairs:
            if any(k in qtext.lower() for k in ["re-execute","reexecute","regenerate","try again"]):
                continue
            q_norm = normalize_question(qtext)
            tokens = set(q_norm.split())
            if not tokens:
                continue
            overlap = len(tokens & target_tokens)
            union = len(tokens | target_tokens)
            jacc = overlap / union if union else 0.0
            if jacc > best_score:
                best_score = jacc
                best = (qtext, answer)
            if best_score >= 0.95:  # early exit on near-perfect match
                break
        if best and best_score >= threshold:
            print(f"[MEM-INDEX] Fuzzy match score={best_score:.2f} retrieved prior answer")
            return {"found": True, "previous_question": best[0], "previous_answer": best[1]}
    except Exception as e:
        print(f"[MEM-INDEX][WARN] Fuzzy history scan failed: {e}")
    return {"found": False}

def update_history(question: str, answer: str) -> None:
    """Update chat history with new question and answer."""
    # Check if this is a re-execution command
    is_reexecute = any(keyword in question.lower() for keyword in 
                      ["re-execute", "reexecute", "regenerate", "try again"])
    
    # If it's a re-execution command, also save a version without the command words
    if is_reexecute:
        # Create a clean version of the question (without re-execute commands)
        clean_question = question.lower()
        for keyword in ["re-execute", "reexecute", "regenerate", "try again", "force", "bypass memory"]:
            clean_question = clean_question.replace(keyword, "").strip()
            
        # Add the clean version to history as well
        clean_user_message = ChatMessage(
            role=MessageRole.USER,
            content=clean_question
        )
        
        clean_assistant_message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=answer
        )
        
        # Add clean version to chat store
        chat_store_private.add_message(key="conversation", message=clean_user_message)
        chat_store_private.add_message(key="conversation", message=clean_assistant_message)
    
    # Create messages for the original question
    user_message = ChatMessage(
        role=MessageRole.USER,
        content=question
    )
    
    assistant_message = ChatMessage(
        role=MessageRole.ASSISTANT,
        content=answer
    )
    
    # Skip storing if answer appears to be an error or pure system notice
    lowered = answer.lower()
    error_indicators = [
        "error while querying",
        "syntax error",
        "encountered an error",
        "critically low on available memory",
        "low on available memory",
        "all query generation methods failed",
        "cannot proceed due to low memory",
        "i'm sorry, but your system is",
        "failed to initialize"
    ]
    is_error_answer = any(tok in lowered for tok in error_indicators)
    if is_error_answer:
        print("[MEM-HISTORY] Skipping storage of error answer")
    else:
        chat_store_private.add_message(key="conversation", message=user_message)
        chat_store_private.add_message(key="conversation", message=assistant_message)
        # Maintain a cap of recent successful QA pairs (e.g., last 200 user+assistant pairs)
        try:
            msgs = chat_store_private.get_messages(key="conversation")
            # Count assistant messages to approximate pairs
            max_pairs = int(os.environ.get("MEMORY_MAX_PAIRS", "200"))
            # We'll rebuild keeping only last max_pairs*2 messages (user+assistant)
            if max_pairs > 0:
                trimmed = []
                # Walk backwards collecting pairs
                pair_msgs = []
                user_seen = 0
                for m in reversed(msgs):
                    pair_msgs.append(m)
                    if m.role == MessageRole.USER:
                        user_seen += 1
                    if user_seen >= max_pairs:
                        break
                trimmed = list(reversed(pair_msgs))
                if len(trimmed) < len(msgs):
                    chat_store_private._data["conversation"] = trimmed  # internal structure update
                    print(f"[MEM-HISTORY] Trimmed history to last {user_seen} user turns")
        except Exception as trim_err:
            print(f"[MEM-HISTORY][WARN] Failed to trim history: {trim_err}")
    
    # Persist only if stored
    if not is_error_answer:
        chat_store_private.persist(str(private_store_path))
        print("Chat history updated successfully")
        # Update memory index
        try:
            key = normalize_question(question)
            question_memory_index[key] = answer
        except Exception as e:
            print(f"[MEM-INDEX][WARN] Failed to index question: {e}")


def get_doc_text(doc_id: str) -> str:
    """Read persisted OCR text for a given doc_id from chat_store/docs/{doc_id}.txt.

    Returns the raw text or empty string if not found.
    """
    try:
        p = Path('chat_store') / 'docs' / f"{doc_id}.txt"
        if not p.exists():
            return ""
        return p.read_text(encoding='utf-8')
    except Exception:
        return ""


# Define the tool functions
def memory_check(question: str) -> str:
    """Check if this question has been asked before."""
    # Check if user explicitly requested to re-execute
    force_reexecute = any(keyword in question.lower() for keyword in 
                          ["re-execute", "reexecute", "regenerate", "try again", "force", "bypass memory"])
    
    # Remove re-execution keywords from the question for memory checking
    clean_question = question.lower()
    for keyword in ["re-execute", "reexecute", "regenerate", "try again", "force", "bypass memory"]:
        clean_question = clean_question.replace(keyword, "").strip()
    
    # If force re-execute, skip memory check and use the current question
    if force_reexecute:
        # Extract the actual question from the re-execute command
        # clean_question already has re-execute keywords removed
        if clean_question and len(clean_question.strip()) > 0:
            print(f"[MEMORY] Re-execute requested. Using current question: '{clean_question}'")
            return f"User requested to re-execute the query. Bypassing memory check."
        else:
            # If no question provided after re-execute, try to find previous question
            try:
                private_messages = chat_store_private.get_messages(key="conversation")
                if len(private_messages) >= 2:
                    # Find the last user question before this re-execute command
                    for i in range(len(private_messages) - 1, -1, -1):
                        if private_messages[i].role == "user":
                            last_user_msg = private_messages[i].content
                            # Skip if this is also a re-execute command or empty
                            if last_user_msg and len(last_user_msg.strip()) > 0 and not any(kw in last_user_msg.lower() for kw in ["re-execute", "reexecute", "regenerate", "try again"]):
                                print(f"[MEMORY] Found original question for re-execute: {last_user_msg}")
                                return f"RE_EXECUTE_ORIGINAL: {last_user_msg}"
            except Exception as e:
                print(f"Error retrieving original question for re-execute: {e}")
            
            # If we can't find a previous question, just bypass memory check
            print("[MEMORY] No previous question found, bypassing memory check for re-execute")
            return f"User requested to re-execute the query. Bypassing memory check."
    
    # Check memory with the cleaned question
    result = check_history(clean_question)
    
    if result["found"]:
        # Check if the previous answer was an error
        if "error" in result["previous_answer"].lower() or "encountered an error" in result["previous_answer"].lower():
            print("Previous answer contained an error. Suggesting to re-execute.")
            return f"FOUND_IN_MEMORY_WITH_ERROR: {result['previous_answer']}"
        else:
            return f"FOUND_IN_MEMORY: {result['previous_answer']}"
    else:
        return "No previous answer found. I'll create a SQL query for this."

##### Hardcoded SQL template fallback system removed to enforce pure LLM-only generation #####
def ensure_llm_ready():
    """Ensure the global LLM has completed a successful handshake. If not, attempt one now
    with an extended timeout (single attempt) so first user query doesn't fail silently.
    Non-raising on failure; query generation will still attempt but will log status.
    """
    global LLM_HANDSHAKE_COMPLETE, llm
    if LLM_HANDSHAKE_COMPLETE:
        return
    print("[HS][LAZY] Performing deferred handshake before first generation...")
    try:
        # Attempt a single prompt with generous timeout (can be overridden)
        timeout = float(os.environ.get("LLM_HANDSHAKE_TIMEOUT_FIRST_QUERY", os.environ.get("LLM_HANDSHAKE_TIMEOUT", "45")))
        resp = TimeoutManager.run_with_timeout(lambda: llm.complete("SELECT 1;"), timeout=timeout)
        if resp is not None:
            LLM_HANDSHAKE_COMPLETE = True
            print(f"[HS][LAZY] Deferred handshake succeeded (timeout={timeout}s)")
    except Exception as e:
        print(f"[HS][LAZY][WARN] Deferred handshake failed: {e}")

# ----------------------------------------------------------------------------
# Lightweight instrumentation utilities (no external deps) to diagnose timeouts
# ----------------------------------------------------------------------------
INSTRUMENT_GENERATION = os.environ.get("GEN_INSTRUMENT", "1") in ("1","true","True")
_GEN_EVENT_COUNTER = 0

def _approx_tokens(text: str) -> int:
    # Crude heuristic; avoids importing a tokenizer. Good enough for relative scale.
    if not text:
        return 0
    # Average 4 chars per token for English + SQL mix
    return max(1, len(text)//4)

def _log_gen(stage: str, **kv):
    if not INSTRUMENT_GENERATION:
        return
    global _GEN_EVENT_COUNTER
    _GEN_EVENT_COUNTER += 1
    # Build compact k=v list; truncate any large values
    parts = [f"{k}={str(v)[:160]}" for k,v in kv.items()]
    print(f"[GEN][INS][{_GEN_EVENT_COUNTER}][{stage}] " + " ".join(parts))

def _shrink_schema_if_needed(prompt: str, schema_text: str) -> str:
    """If prompt becomes very large (> ~9k chars) aggressively shrink schema section.
    Returns (maybe) new schema_text. Keep logic minimal & reversible."""
    if len(prompt) < 9000:
        return schema_text
    lines = schema_text.splitlines()
    # Keep only first 2 table lines if huge
    reduced = []
    kept_tables = 0
    for ln in lines:
        if ln.strip():
            reduced.append(ln)
            if ':' in ln:
                kept_tables += 1
            if kept_tables >= 2:
                break
    reduced.append("...schema trimmed for brevity...")
    new_schema = "\n".join(reduced)
    return new_schema


# ===============================================================================
# 3-LEVEL AGENTIC SQL GENERATION ARCHITECTURE
# ===============================================================================
# Helper functions for the 3-level SQL generation approach:
# - Level 1: Pure LLM intelligence (no assistance)
# - Level 2: LLM + SQL Fixer enhancement
# - Level 3: LLM + MCP hints + SQL Fixer
#
# Each level attempts to generate SQL, and if it fails, escalates to the next level.
# Loop prevention: Strict level progression, single attempt per level, no recursive calls.
# ===============================================================================

def classify_fast_sql_intent(q: str):
    """
    Classify user question to determine if it matches a fast SQL (MCP) intent pattern.
    Returns (tool_name, tool_args) or (None, None) if no match.
    
    This function extracts the intent classification logic from query_craft to make it reusable.
    """
    ql = q.lower()
    patterns = [
        # Put salary range FIRST to avoid conflicts with other patterns
        (r"salary range.*department|maximum and minimum salary.*department", ("fast_sql.salary_range_department", {})),
        # DISABLED - NO HARDCODED QUERIES: (r"gender pay gap|pay gap|salary gap.*(male|female)|salary gap.*department|salary.*difference.*male.*female|pay.*difference.*male.*female", ("fast_sql.gender_pay_gap", {})),
        (r"highest average salary|department .*highest average salary|which department has the highest average salary", ("fast_sql.department_highest_avg_salary", {})),
        # Make top paid pattern more specific to avoid false matches
        (r"top \d+ .*paid|highest[ -]paid employees|top paid employees", ("fast_sql.top_paid_employees", {})),
        (r"employee count by department|employees per department|how many employees .*department|count of employees .*department", ("fast_sql.employee_count_by_department", {})),
        (r"list departments|show departments|what departments", ("fast_sql.list_departments", {})),
        (r"how many employees|total employees|employee count", ("fast_sql.count_employees", {})),
        (r"highest and lowest salary|highest.*lowest salary|lowest.*highest salary", ("fast_sql.salary_extremes", {})),
        (r"department managers|list managers per department|managers of each department", ("fast_sql.department_manager_listing", {})),
    ]
    for pat, result in patterns:
        try:
            m = re.search(pat, ql)
        except Exception:
            m = None
        if m:
            if result[0] == 'fast_sql.top_paid_employees':
                nm = re.search(r"top\s+(\d+)", ql)
                top_n = int(nm.group(1)) if nm else 10
                return result[0], {"top_n": top_n, "question": q}
            if result[0] == 'fast_sql.salary_range_department':
                dm = re.search(r"salary range .*department (?:for|of) ([A-Za-z &'-]+)", ql)
                dept_name = dm.group(1).strip().title() if dm else 'Development'
                return result[0], {"department_name": dept_name, "question": q}
            return result[0], {"question": q}
    return None, None


def get_mcp_hint_if_available(question: str):
    """
    Attempt to retrieve a SQL hint from MCP server based on question classification.
    Returns SQL string as a hint (not to be used directly), or None if unavailable.
    
    This is Level 3 assistance - MCP provides hints to guide the LLM, not hardcoded SQL.
    """
    ENABLE_FAST_SQL = os.environ.get("ENABLE_MCP_FAST_TOOLS", "1") not in ("0", "false", "False")
    FAST_SQL_LOG = os.environ.get("FAST_SQL_LOG", "1") not in ("0", "false", "False")
    
    def _fast_log(stage: str, **kv):
        if FAST_SQL_LOG:
            parts = [f"{k}={str(v)[:140]}" for k, v in kv.items()]
            print(f"[FAST][{stage}] " + " ".join(parts))
    
    if not ENABLE_FAST_SQL:
        return None
    
    tool_name, tool_args = classify_fast_sql_intent(question)
    if not tool_name:
        return None
    
    _fast_log('INTENT', tool=tool_name, args=tool_args)
    try:
        from .mcp_client import mcp_call_tool
        ctx = {"tool_call": {"name": tool_name, "args": tool_args or {}}}
        start_fast = time.time()
        resp = mcp_call_tool(prompt=f"(fast sql) {question}", timeout=45.0, context=ctx)
        dur = round(time.time() - start_fast, 3)
        tool_res = resp.get('tool_result') if isinstance(resp, dict) else None
        if isinstance(tool_res, dict) and 'sql' in tool_res:
            sql = tool_res['sql']
            _fast_log('TOOL_SQL', ms=int(dur*1000), sql_preview=sql[:160])
            return sql  # Return as hint
        else:
            _fast_log('NO_SQL', raw=tool_res)
            return None
    except Exception as fast_err:
        _fast_log('FALLBACK', error=str(fast_err))
        return None


def is_sql_valid(sql: str, question: str = None) -> bool:
    """
    Validation check for generated SQL with both syntactic and semantic checks.
    
    Args:
        sql: The SQL query to validate
        question: Optional user question for semantic validation
        
    Returns:
        True if SQL appears valid both syntactically and semantically, False otherwise.
    """
    if not sql or not isinstance(sql, str):
        return False
    sql_lower = sql.lower().strip()
    
    # Must contain SELECT
    if 'select' not in sql_lower:
        return False
    
    # Quick placeholder check
    PLACEHOLDER_TOKENS = ["<", ">", "...", "___", "placeholder", "example", "your_"]
    PLACEHOLDER_REGEXES = [
        r"<[A-Z_]+>",  # <TABLE_NAME>, <COLUMN>
        r"\b(table_name|column_name|your_\w+)\b"
    ]
    
    for token in PLACEHOLDER_TOKENS:
        if token in sql_lower:
            return False
    
    for regex in PLACEHOLDER_REGEXES:
        if re.search(regex, sql, re.IGNORECASE):
            return False
    
    # Reject hardcoded department codes (D001-D009) - indicates incomplete query
    # These suggest LLM didn't join to department table to filter by name
    if re.search(r"'D00[0-9]'", sql, re.IGNORECASE):
        print("[VALIDATION] Rejected: Contains hardcoded department code (needs dept_name filter)")
        return False

    # STRICT SCHEMA CHECK: Reject common hallucinations before they hit the DB
    # employee (e) does NOT have dept_no or salary
    if re.search(r'\b[eE]\.dept_no\b|employee\.dept_no\b', sql, re.IGNORECASE):
        print("[VALIDATION] Rejected: Table 'employee' lacks 'dept_no'. Bridging through 'dept_emp' is required.")
        return False
    if re.search(r'\b[eE]\.salary\b|employee\.salary\b', sql, re.IGNORECASE):
        print("[VALIDATION] Rejected: Table 'employee' lacks 'salary'. Join with 'salary' table is required.")
        return False
    
    # MANAGER JOIN CHECK: Note that dept_manager should be used for manager queries.
    # We do NOT hard-reject here — adjust_manager_query() in apply_semantic_adjusters
    # will regenerate the SQL using generate_manager_list_sql() if dept_manager is absent.
    # Hard-rejecting here blocks all 3 levels before the adjuster ever gets a chance to fix it.
    if question and any(k in question.lower() for k in ["manager", "managers", "who manages"]):
        if 'dept_manager' not in sql.lower() and 'manager_name' not in sql.lower():
            print("[VALIDATION] Warning: Manager query missing dept_manager — adjuster will fix after this level succeeds.")


    # ===== SEMANTIC VALIDATION =====
    # Check if question asks for specific department but SQL doesn't filter by it
    if question:
        print(f"[VALIDATION DEBUG] Checking semantic validation for question: {question[:100]}")
        question_lower = question.lower()
        
        # Department-specific query patterns
        dept_patterns = [
            (r'\bin\s+(development|finance|sales|marketing|research|production|hr|human\s+resources|quality\s+management|customer\s+service)\s+(department|dept)\b', 1),
            (r'\b(development|finance|sales|marketing|research|production|hr|human\s+resources|quality\s+management|customer\s+service)\s+department\b', 1),
        ]
        
        for pattern, group_idx in dept_patterns:
            match = re.search(pattern, question_lower, re.IGNORECASE)
            if match:
                dept_name = match.group(group_idx).strip()
                print(f"[VALIDATION DEBUG] Found department mention: '{dept_name}'")
                # Normalization
                dept_map = {
                    'hr': 'Human Resources',
                    'human resources': 'Human Resources',
                    'quality management': 'Quality Management',
                    'customer service': 'Customer Service'
                }
                dept_name_proper = dept_map.get(dept_name.lower(), dept_name.title())
                print(f"[VALIDATION DEBUG] Normalized to: '{dept_name_proper}'")
                print(f"[VALIDATION DEBUG] Checking if '{dept_name_proper.lower()}' is in SQL")
                print(f"[VALIDATION DEBUG] SQL being checked (first 200 chars): {sql_lower[:200]}")
                
                # Check if SQL contains this department name as a filter
                if dept_name_proper.lower() not in sql_lower:
                    print(f"[VALIDATION] Rejected: Question asks for '{dept_name_proper}' but SQL doesn't filter by it")
                    print(f"[VALIDATION] This likely returns ALL departments instead of just '{dept_name_proper}'")
                    return False
                else:
                    print(f"[VALIDATION DEBUG] Department filter found in SQL - validation passed")
                break  # First matching pattern is sufficient — no need to check remaining patterns
    else:
        print("[VALIDATION DEBUG] No question provided for semantic validation")
    
    # ===== ALIAS VALIDATION =====
    # Check if SQL references aliases that aren't defined in FROM/JOIN clauses
    # Extract defined aliases from FROM and JOIN clauses
    defined_aliases = set()
    
    # Match FROM table alias and JOIN table alias patterns
    from_pattern = r'\bFROM\s+(\w+)\s+(?:AS\s+)?(\w+)\b'
    join_pattern = r'\bJOIN\s+(\w+)\s+(?:AS\s+)?(\w+)\b'
    
    # Exclude SQL keywords and common column names from alias matching
    _SQL_KEYWORDS = {'where', 'group', 'order', 'limit', 'having', 'join', 'on',
                     'and', 'or', 'not', 'in', 'as', 'by', 'select', 'from', 'set',
                     'emp_no', 'dept_no', 'from_date', 'to_date', 'first_name', 'last_name'}
    # Strip EXTRACT(... FROM ...) to avoid matching column names as tables
    _sql_for_alias = re.sub(r'\bEXTRACT\s*\([^)]+\)', 'EXTRACT_EXPR', sql, flags=re.IGNORECASE)
    _sql_for_alias_lower = _sql_for_alias.lower()

    for match in re.finditer(from_pattern, _sql_for_alias, re.IGNORECASE):
        table_name = match.group(1).lower()
        alias = match.group(2).lower()
        # Skip SQL keywords mistakenly matched as aliases
        if alias in _SQL_KEYWORDS or table_name in _SQL_KEYWORDS:
            continue
        if alias != table_name:
            defined_aliases.add(alias)
            print(f"[VALIDATION DEBUG] Found alias '{alias}' for table '{table_name}'")

    for match in re.finditer(join_pattern, _sql_for_alias, re.IGNORECASE):
        table_name = match.group(1).lower()
        alias = match.group(2).lower()
        if alias in _SQL_KEYWORDS or table_name in _SQL_KEYWORDS:
            continue
        if alias != table_name:
            defined_aliases.add(alias)
            print(f"[VALIDATION DEBUG] Found alias '{alias}' for table '{table_name}'")
    
    # NEW: Handle comma-separated joins like FROM table1 t1, table2 t2
    # Only look at the part BEFORE the first JOIN or WHERE to avoid condition noise
    comma_from_pattern = r'\bFROM\s+(.*?)(?=\bJOIN\b|\bWHERE\b|$)'
    comma_match = re.search(comma_from_pattern, _sql_for_alias, re.IGNORECASE | re.DOTALL)
    if comma_match:
        from_clause = comma_match.group(1)
        # Find all pairs of word+word (table alias)
        alias_pairs = re.findall(r'(\w+)\s+(?:AS\s+)?(\w+)', from_clause, re.IGNORECASE)
        for table_name, alias in alias_pairs:
            t_low = table_name.lower()
            a_low = alias.lower()
            if t_low in _SQL_KEYWORDS or a_low in _SQL_KEYWORDS:
                continue
            defined_aliases.add(a_low)
            print(f"[VALIDATION DEBUG] Found comma-join alias '{a_low}'")
    
    # Find all alias references (pattern: alias.column)
    alias_ref_pattern = r'\b([a-z_]\w*)\.(\w+)\b'
    used_aliases = set()
    
    for match in re.finditer(alias_ref_pattern, sql_lower, re.IGNORECASE):
        alias = match.group(1)
        # Skip if it's a table name (not an alias)
        if alias not in ['employee', 'department', 'salary', 'dept_emp', 'dept_manager', 'title', 'titles', 'titles']:
            used_aliases.add(alias)
    
    # Check for undefined aliases
    undefined_aliases = used_aliases - defined_aliases
    if undefined_aliases:
        print(f"[VALIDATION] Rejected: SQL references undefined aliases: {undefined_aliases}")
        print(f"[VALIDATION] Defined aliases: {defined_aliases}")
        print(f"[VALIDATION] Used aliases: {used_aliases}")
        return False
    
    # Basic structure check
    if not sql.strip().endswith(';'):
        return False
    
    return True


def level_1_pure_llm(question: str, schema_text: str, relevant_tables: list, 
                     base_llm, hints: str = "", timeout: float = 35.0):
    """
    Level 1: Pure LLM intelligence with no assistance.
    
    Args:
        question: User's natural language question
        schema_text: Formatted schema information
        relevant_tables: List of relevant table names
        base_llm: The LLM instance to use
        timeout: Timeout for LLM call in seconds
        
    Returns:
        Valid SQL string or None if generation fails
    """
    print("[LEVEL_1] Attempting pure LLM generation (no hints, no assistance)")
    
    prompt = (
        f"{EMPLOYEE_DB_SCHEMA}\n\n"
        f"You are an SQL expert. Convert the natural language question into a SINGLE valid PostgreSQL query.\n"
        f"Question: {question}\n"
        f"Available tables: {', '.join(relevant_tables)}\n"
        f"Table schemas (columns):\n{schema_text}\n"
        f"Rules: Do NOT invent column names. ALWAYS filter to_date = '9999-01-01' to get current records in salary, dept_emp, and dept_manager. "
        f"Append LIMIT 100 ONLY if a specific count (like 'top 10') is not requested. "
        f"The query must end with a single semicolon. Return ONLY raw SQL."
        f"Be precise with column names and table joins."
    )
    
    try:
        start_t = time.time()
        raw = base_llm.complete(prompt=prompt, timeout=timeout)
        gen_dur = time.time() - start_t
        text_out = raw.text if hasattr(raw, 'text') else str(raw)
        print(f"[LEVEL_1] LLM responded in {gen_dur:.2f}s")
        
        # Extract SQL from code fences if present
        sql_query = text_out.strip()
        if '```' in sql_query:
            parts = re.split(r"```(?:sql)?", sql_query, flags=re.IGNORECASE)
            sql_query = '\n'.join(p for p in parts if 'select' in p.lower())
        if sql_query.lower().startswith('sql'):
            sql_query = sql_query[3:].strip()
        
        # Clean and validate with semantic checks
        sql_query = extract_clean_sql(sql_query)
        if is_sql_valid(sql_query, question):
            print(f"[LEVEL_1] ✓ Generated valid SQL: {sql_query[:100]}...")
            return sql_query
        else:
            print(f"[LEVEL_1] ✗ Generated SQL failed validation. SQL: {sql_query}")
            return None
            
    except Exception as e:
        print(f"[LEVEL_1] ✗ Failed with error: {type(e).__name__}: {e}")
        return None


def generate_micro_gen_hint(question: str) -> str:
    """
    Generate micro-gen SQL hint based on question pattern.
    This is called LAZILY inside Level 2, only when Level 1 fails.
    
    Args:
        question: User's natural language question
        
    Returns:
        SQL hint string or None if no pattern matches
    """
    question_lower = question.lower()
    
    # Salary range by department pattern
    if re.search(r'\b(maximum and minimum|max.*min|min.*max|salary range)\b.*\b(development|dept|department)\b', question_lower):
        print("[MICRO-GEN][LAZY] Detected salary range in department question")
        try:
            # Extract department name from question
            dept_name = "Development"  # Default
            if "development" in question_lower:
                dept_name = "Development"
            elif "sales" in question_lower:
                dept_name = "Sales"
            elif "marketing" in question_lower:
                dept_name = "Marketing"
            elif "finance" in question_lower:
                dept_name = "Finance"
            elif "human resources" in question_lower:
                dept_name = "Human Resources"
            elif "research" in question_lower:
                dept_name = "Research"
            elif "production" in question_lower:
                dept_name = "Production"
            
            # Smart Hint: If question asks for EACH/ALL/Distribution, don't filter by a single department
            is_global = any(k in question_lower for k in ["each", "all", "every", "distribution", "per department", "by department"])
            
            # Build SQL hint
            where_clause = f"WHERE d.dept_name = '{dept_name}' " if not is_global else " "
            hint_sql = (
                f"SELECT d.dept_name, MAX(s.salary) AS max_salary, MIN(s.salary) AS min_salary "
                f"FROM salary s "
                f"JOIN employee e ON s.emp_no = e.emp_no "
                f"JOIN dept_emp de ON e.emp_no = de.emp_no "
                f"JOIN department d ON de.dept_no = d.dept_no "
                f"{where_clause}"
                f"AND s.to_date = '9999-01-01' "
                f"AND de.to_date = '9999-01-01' "
                f"{'GROUP BY d.dept_name' if is_global else ''};"
            )
            if is_global:
                print(f"[MICRO-GEN][LAZY] Generated GLOBAL hint (no department filter)")
            else:
                print(f"[MICRO-GEN][LAZY] Generated hint for {dept_name}")
            return hint_sql
        except Exception as e:
            print(f"[MICRO-GEN][LAZY] Failed: {e}")
            return None
    
    # Add more patterns here as needed
    # ...
    
    return None


def level_2_llm_with_fixer(question: str, schema_text: str, relevant_tables: list,
                           base_llm, hints: str = "", timeout: float = 45.0):
    """
    Level 2: LLM + Micro-gen hints + SQL Fixer enhancement.
    
    This level LAZILY generates schema-aware hints from micro-generators
    (only when Level 1 fails and Level 2 is invoked).
    Then applies fix_common_sql_errors for repair.
    
    Args:
        question: User's natural language question
        schema_text: Formatted schema information
        relevant_tables: List of relevant table names
        base_llm: The LLM instance to use
        timeout: Timeout for LLM call in seconds
        
    Returns:
        Valid SQL string or None if generation fails
    """
    print("[LEVEL_2] Attempting LLM + Micro-gen hints + SQL Fixer")
    
    # LAZY GENERATION: Only generate micro-hint NOW (not before Level 1)
    micro_gen_hint = generate_micro_gen_hint(question)
    
    # Build hint text if micro-gen provided one
    hint_text = ""
    if micro_gen_hint:
        hint_text = (
            f"\n💡 MICRO-GEN HINT (schema-aware pattern):\n{micro_gen_hint}\n\n"
            f"Use this as a guide but adapt column/table names to match the actual schema.\n"
        )
        print(f"[LEVEL_2] Micro-gen hint available: {micro_gen_hint[:100]}...")
    else:
        print("[LEVEL_2] No micro-gen hint available, using enhanced prompt only")
    
    # Enhanced prompt with optional micro-gen hint
    prompt = (
        f"{EMPLOYEE_DB_SCHEMA}\n\n"
        f"{hint_text}"
        f"Question: {question}\n"
        f"Tables: {', '.join(relevant_tables)}\n"
        f"Table schemas:\n{schema_text}\n"
        f"Rules: ALWAYS filter to_date = '9999-01-01' to get current records in salary, dept_emp, and dept_manager. "
        f"Append LIMIT 100 ONLY if a specific count (like 'top 10') is not requested. "
        f"The query must end with a single semicolon. Return ONLY SQL (PostgreSQL)."
        f"Be precise with column names and table joins."
    )
    
    try:
        start_t = time.time()
        raw = base_llm.complete(prompt=prompt, timeout=timeout)
        gen_dur = time.time() - start_t
        text_out = raw.text if hasattr(raw, 'text') else str(raw)
        print(f"[LEVEL_2] LLM responded in {gen_dur:.2f}s")
        
        # Extract SQL from code fences if present
        sql_query = text_out.strip()
        if '```' in sql_query:
            parts = re.split(r"```(?:sql)?", sql_query, flags=re.IGNORECASE)
            sql_query = '\n'.join(p for p in parts if 'select' in p.lower())
        if sql_query.lower().startswith('sql'):
            sql_query = sql_query[3:].strip()
        
        # Clean and apply fixer
        sql_query = extract_clean_sql(sql_query)
        sql_query = fix_common_sql_errors(sql_query)
        
        # Additional validation
        base_val = validate_sql_query(sql_query)
        if not base_val['valid']:
            sql_query = fix_query(sql_query, base_val['reason'])
        
        if is_sql_valid(sql_query, question):
            print(f"[LEVEL_2] ✓ Generated and fixed valid SQL: {sql_query[:100]}...")
            return sql_query
        else:
            print(f"[LEVEL_2] ✗ Generated SQL failed validation after fixes")
            return None
            
    except Exception as e:
        print(f"[LEVEL_2] ✗ Failed with error: {type(e).__name__}: {e}")
        return None


def level_3_mcp_hints_llm_fixer(question: str, schema_text: str, relevant_tables: list,
                                base_llm, hints: str = "", timeout: float = 55.0):
    """
    Level 3: LLM + MCP hints + SQL Fixer (MCP provides hints to guide the LLM).
    
    This is the final escalation level. MCP tools provide SQL patterns as hints,
    which are injected into the prompt to guide the LLM's generation.
    
    Args:
        question: User's natural language question
        schema_text: Formatted schema information
        relevant_tables: List of relevant table names
        base_llm: The LLM instance to use
        timeout: Timeout for LLM call in seconds
        
    Returns:
        Valid SQL string or None if generation fails
    """
    print("[LEVEL_3] Attempting LLM + MCP hints + SQL Fixer")
    
    # Try to get MCP hint
    mcp_hint = get_mcp_hint_if_available(question)
    
    # Construct hint text
    micro_hint_text = ""
    if mcp_hint:
        micro_hint_text = (
            f"\n🎯 HINT FROM MCP - SQL PATTERN SUGGESTION:\n{mcp_hint}\n\n"
            f"This is a suggested SQL pattern. Use it as guidance but adapt as needed for the schema.\n"
        )
        print(f"[LEVEL_3] MCP hint available: {mcp_hint[:100]}...")
    else:
        print("[LEVEL_3] No MCP hint available, proceeding with enhanced prompt")
    
    # Enhanced prompt with MCP hint if available
    prompt = (
        f"{EMPLOYEE_DB_SCHEMA}\n\n"
        f"{micro_hint_text}"
        f"You are an SQL expert. Convert the natural language question into a SINGLE valid PostgreSQL query.\n"
        f"Question: {question}\n"
        f"Available tables: {', '.join(relevant_tables)}\n"
        f"Table schemas (columns):\n{schema_text}\n"
        f"Rules: Do NOT invent column names. Use existing columns from provided schemas. "
        f"Use the LIMIT value requested in the question, or default to LIMIT 100; if none is mentioned. "
        f"End with a semicolon. Return ONLY raw SQL.\n"
        f"CRITICAL: If a hint is provided above, use it as a pattern guide but ensure it matches the actual schema."
    )
    
    try:
        start_t = time.time()
        raw = base_llm.complete(prompt=prompt, timeout=timeout)
        gen_dur = time.time() - start_t
        text_out = raw.text if hasattr(raw, 'text') else str(raw)
        print(f"[LEVEL_3] LLM responded in {gen_dur:.2f}s")
        
        # Extract SQL from code fences if present
        sql_query = text_out.strip()
        if '```' in sql_query:
            parts = re.split(r"```(?:sql)?", sql_query, flags=re.IGNORECASE)
            sql_query = '\n'.join(p for p in parts if 'select' in p.lower())
        if sql_query.lower().startswith('sql'):
            sql_query = sql_query[3:].strip()
        
        # Clean and apply comprehensive fixes
        sql_query = extract_clean_sql(sql_query)
        sql_query = fix_common_sql_errors(sql_query)
        
        # Additional validation and fixes
        base_val = validate_sql_query(sql_query)
        if not base_val['valid']:
            sql_query = fix_query(sql_query, base_val['reason'])
        
        if is_sql_valid(sql_query, question):
            print(f"[LEVEL_3] ✓ Generated valid SQL with MCP guidance: {sql_query[:100]}...")
            return sql_query
        else:
            print(f"[LEVEL_3] ✗ Generated SQL failed validation after all fixes")
            return None
            
    except Exception as e:
        print(f"[LEVEL_3] ✗ Failed with error: {type(e).__name__}: {e}")
        return None


def query_craft(question: str) -> str:
    """Generate SQL via LLM (no hardcoded templates) with minimal retries and semantic post-fixes.
    Reconstructed clean version after refactor issues.
    """
    # Make sure LLM handshake succeeded or attempt deferred handshake now
    ensure_llm_ready()
    print(f"[QUERY_CRAFT] Received question: {question}")
    question_lower = question.lower()

    # ---------------- DISABLED: Old MCP tool dispatch (replaced by 3-level architecture) -----------------
    # The 3-level architecture handles MCP hints in Level 3 via get_mcp_hint_if_available()
    # This old code was causing duplicate MCP calls and wasting ~31s
    ENABLE_FAST_SQL = False  # FORCE DISABLED to prevent duplicate MCP calls
    FAST_SQL_LOG = os.environ.get("FAST_SQL_LOG", "1") not in ("0","false","False")

    def _fast_log(stage: str, **kv):
        if FAST_SQL_LOG:
            parts = [f"{k}={str(v)[:140]}" for k,v in kv.items()]
            print(f"[FAST][{stage}] " + " ".join(parts))

    def classify_fast_sql_intent(q: str):
        ql = q.lower()
        patterns = [
            # Put salary range FIRST to avoid conflicts with other patterns
            (r"salary range.*department|maximum and minimum salary.*department", ("fast_sql.salary_range_department", {})),
            # DISABLED - NO HARDCODED QUERIES: (r"gender pay gap|pay gap|salary gap.*(male|female)|salary gap.*department|salary.*difference.*male.*female|pay.*difference.*male.*female", ("fast_sql.gender_pay_gap", {})),
            (r"highest average salary|department .*highest average salary|which department has the highest average salary", ("fast_sql.department_highest_avg_salary", {})),
            # Make top paid pattern more specific to avoid false matches
            (r"top \d+ .*paid|highest[- ]paid employees|top paid employees", ("fast_sql.top_paid_employees", {})),
            (r"employee count by department|employees per department|how many employees .*department|count of employees .*department", ("fast_sql.employee_count_by_department", {})),
            (r"list departments|show departments|what departments", ("fast_sql.list_departments", {})),
            (r"how many employees|total employees|employee count", ("fast_sql.count_employees", {})),
            (r"highest and lowest salary|highest.*lowest salary|lowest.*highest salary", ("fast_sql.salary_extremes", {})),
            (r"department managers|list managers per department|managers of each department", ("fast_sql.department_manager_listing", {})),
        ]
        for pat, result in patterns:
            try:
                m = re.search(pat, ql)
            except Exception:
                m = None
            if m:
                if result[0] == 'fast_sql.top_paid_employees':
                    nm = re.search(r"top\s+(\d+)", ql)
                    top_n = int(nm.group(1)) if nm else 10
                    return result[0], {"top_n": top_n, "question": q}
                if result[0] == 'fast_sql.salary_range_department':
                    dm = re.search(r"salary range .*department (?:for|of) ([A-Za-z &'-]+)", ql)
                    dept_name = dm.group(1).strip().title() if dm else 'Development'
                    return result[0], {"department_name": dept_name, "question": q}
                return result[0], {"question": q}
        return None, None

    micro_hint = None
    if ENABLE_FAST_SQL:
        tool_name, tool_args = classify_fast_sql_intent(question)
        if tool_name:
            _fast_log('INTENT', tool=tool_name, args=tool_args)
            try:
                from .mcp_client import mcp_call_tool
                ctx = {"tool_call": {"name": tool_name, "args": tool_args or {}}}
                start_fast = time.time()
                resp = mcp_call_tool(prompt=f"(fast sql) {question}", timeout=45.0, context=ctx)
                dur = round(time.time() - start_fast, 3)
                tool_res = resp.get('tool_result') if isinstance(resp, dict) else None
                if isinstance(tool_res, dict) and 'sql' in tool_res:
                    sql = tool_res['sql']
                    _fast_log('TOOL_SQL', ms=int(dur*1000), sql_preview=sql[:160])
                    # Use as a hint, not as a direct return
                    micro_hint = sql
                else:
                    _fast_log('NO_SQL', raw=tool_res)
            except Exception as fast_err:
                _fast_log('FALLBACK', error=str(fast_err))
                # EARLY SHORT-CIRCUIT for salary extremes: use as hint only
                if tool_name == 'fast_sql.salary_extremes':
                    try:
                        if 'salary' in available_tables:
                            try:
                                cols = [c['name'] for c in inspector.get_columns(table_name='salary')]
                                scol = 'salary' if 'salary' in cols else (cols[0] if cols else 'salary')
                            except Exception:
                                scol = 'salary'
                            parts = [f"SELECT MAX(s.{scol}) AS highest_salary, MIN(s.{scol}) AS lowest_salary", "FROM salary s"]
                            try:
                                if any(c['name'] == 'to_date' for c in inspector.get_columns(table_name='salary')):
                                    parts.append("WHERE s.to_date = '9999-01-01'")
                            except Exception:
                                pass
                            agg_sql = " ".join(parts)
                            if not agg_sql.strip().endswith(';'):
                                agg_sql += ';'
                            _fast_log('EARLY_EXTREMES', sql=agg_sql)
                            micro_hint = agg_sql
                    except Exception as early_err:
                        _fast_log('EARLY_EXTREMES_FAIL', err=str(early_err))
                # EARLY SHORT-CIRCUIT for salary range by department: use as hint only
                elif tool_name == 'fast_sql.salary_range_department':
                    try:
                        if 'salary' in available_tables and 'department' in available_tables:
                            dept_name = tool_args.get('department_name', 'Development')
                            try:
                                cols = [c['name'] for c in inspector.get_columns(table_name='salary')]
                                scol = 'salary' if 'salary' in cols else (cols[0] if cols else 'salary')
                            except Exception:
                                scol = 'salary'
                            parts = [
                                f"SELECT MAX(s.{scol}) AS max_salary, MIN(s.{scol}) AS min_salary",
                                "FROM salary s",
                                "JOIN employee e ON s.emp_no = e.emp_no",
                                "JOIN dept_emp de ON e.emp_no = de.emp_no",
                                "JOIN department d ON de.dept_no = d.dept_no",
                                f"WHERE d.dept_name = '{dept_name}'"
                            ]
                            try:
                                if any(c['name'] == 'to_date' for c in inspector.get_columns(table_name='salary')):
                                    parts.append("AND s.to_date = '9999-01-01'")
                                if any(c['name'] == 'to_date' for c in inspector.get_columns(table_name='dept_emp')):
                                    parts.append("AND de.to_date = '9999-01-01'")
                            except Exception:
                                pass
                            range_sql = " ".join(parts)
                            if not range_sql.strip().endswith(';'):
                                range_sql += ';'
                            _fast_log('EARLY_RANGE_DEPT', sql=range_sql, dept=dept_name)
                            micro_hint = range_sql
                    except Exception as early_err:
                        _fast_log('EARLY_RANGE_DEPT_FAIL', err=str(early_err))
                # proceed to normal path


    # Infer relevant tables heuristically from question
    table_set = set()
    if any(k in question_lower for k in ["manager", "managers"]):
        table_set.update(["department", "dept_manager", "employee"])
    if any(k in question_lower for k in ["average salary", "avg salary", "salary"]):
        table_set.update(["salary", "dept_emp", "department", "employee"])
    if "department" in question_lower:
        table_set.add("department")
    if any(k in question_lower for k in ["engineer", "title", "analyst", "developer", "staff"]):
        table_set.add("title")
    if not table_set:
        # default minimal context
        table_set.add("employee")
    # Keep only existing tables
    relevant_tables = [t for t in table_set if t in available_tables]
    if not relevant_tables:
        relevant_tables = available_tables[:4]  # small slice fallback

    # Build lightweight schema snippets for relevant tables (column names only) to reduce hallucinated columns
    def _collect_schema_snippets(tables: List[str], per_table_col_cap: int = 15) -> List[str]:
        out: List[str] = []
        try:
            for tbl in tables:
                try:
                    cols = [c['name'] for c in inspector.get_columns(table_name=tbl)]
                    if cols:
                        shown = cols[:per_table_col_cap]
                        out.append(f"{tbl}: {', '.join(shown)}")
                except Exception:
                    continue
        except Exception:
            pass
        return out

    schema_snippets = _collect_schema_snippets(relevant_tables)
    schema_text = "\n".join(schema_snippets) if schema_snippets else "(schema unavailable)"

    # ---------------- DISABLED: Plan Phase (not needed by 3-level architecture) ----------------
    # The 3-level architecture generates SQL directly without needing a separate planning step
    # This was adding ~25s overhead
    ENABLE_PLAN_PHASE = False  # FORCE DISABLED to eliminate redundant LLM call
    plan_dict: Optional[Dict[str, Any]] = None
    if ENABLE_PLAN_PHASE:
        plan_prompt = (
            "You are a planning assistant. Given a natural language question, output ONLY JSON describing the SQL plan.\n"
            "Keys: tables (list of table names), joins (list of join conditions like 'a.col = b.col'), filters (list of simple predicates),"
            " group_by (list of columns), metrics (list of aggregate expressions), order_by (optional string), limit (int or null).\n"
            "Do not include explanatory text. Return strict JSON only.\n"
            f"Question: {question}\nJSON:"
        )
        try:
            # Short timeout plan step to avoid long stall
            _plan_start = time.time()
            plan_resp = Settings.llm.complete(plan_prompt, timeout=10.0)
            _plan_dur = time.time() - _plan_start
            _log_gen("plan", dur_s=round(_plan_dur,2), prompt_chars=len(plan_prompt), est_tokens=_approx_tokens(plan_prompt))
            raw_plan = plan_resp.text if hasattr(plan_resp, 'text') else str(plan_resp)
            # Extract first JSON object
            import json as _json, re as _re
            m = _re.search(r"\{.*\}", raw_plan, flags=_re.DOTALL)
            if m:
                candidate = m.group(0)
                try:
                    plan_dict = _json.loads(candidate)
                except Exception:
                    plan_dict = None
            # Basic validation
            if plan_dict:
                ptables = plan_dict.get('tables') or []
                if not isinstance(ptables, list):
                    plan_dict = None
                else:
                    # Ensure tables exist
                    cleaned = [t for t in ptables if isinstance(t, str) and t in available_tables]
                    if not cleaned:
                        plan_dict = None
                    else:
                        plan_dict['tables'] = cleaned
                        # Replace relevant_tables with planned subset to shrink prompt
                        relevant_tables = cleaned
                        schema_snippets = _collect_schema_snippets(relevant_tables, per_table_col_cap=12)
                        schema_text = "\n".join(schema_snippets) if schema_snippets else "(schema unavailable)"
                        print(f"[PLAN] Accepted plan tables={cleaned} joins={plan_dict.get('joins')} metrics={plan_dict.get('metrics')}")
            else:
                print("[PLAN][WARN] Could not parse plan JSON; continuing without plan phase")
        except Exception as plan_err:
            print(f"[PLAN][WARN] Plan phase failed: {plan_err}")

    def _build_sql_prompt_from_plan(plan: Dict[str, Any]) -> str:
        """Construct a concise prompt using the validated plan dict."""
        return (
            f"You are an expert SQL generator. Use ONLY the provided plan and schema to write ONE PostgreSQL query.\n"
            f"Question: {question}\n"
            f"Plan tables: {', '.join(plan.get('tables', []))}\n"
            f"Plan joins: {plan.get('joins', [])}\n"
            f"Plan filters: {plan.get('filters', [])}\n"
            f"Plan group_by: {plan.get('group_by', [])}\n"
            f"Plan metrics: {plan.get('metrics', [])}\n"
            f"Plan order_by: {plan.get('order_by', '')}\n"
            f"Plan limit: {plan.get('limit', 100)}\n"
            f"Table schemas (columns):\n{schema_text}\n"
            "Rules: Do NOT invent columns. Apply all joins & filters. Ensure current rows by filtering to_date='9999-01-01' where relevant."
            " End with LIMIT 100; Return ONLY raw SQL."
        )

    # Lightweight hints (pure text, not SQL)
    hints = []
    if "manager" in question_lower:
        hints.append("Join department d, dept_manager dm, employee e")
    if "average" in question_lower and "salary" in question_lower:
        # Check if a specific department is mentioned — give a targeted hint
        _dept_names = ["marketing", "sales", "development", "research", "finance",
                       "human resources", "production", "quality management", "customer service"]
        _mentioned_dept = next((d.title() for d in _dept_names if d in question_lower), None)
        if _mentioned_dept:
            hints.append(
                f"Filter by specific department: FROM salary s "
                f"JOIN dept_emp de ON s.emp_no = de.emp_no "
                f"JOIN department d ON de.dept_no = d.dept_no "
                f"WHERE d.dept_name = '{_mentioned_dept}' AND s.to_date = '9999-01-01' AND de.to_date = '9999-01-01'. "
                f"dept_emp has NO dept_name column — you MUST join department d to filter by name."
            )
        else:
            hints.append("AVG(s.salary) per department; join salary s, dept_emp de, department d; dept_emp has NO dept_name — join department d to get name")
    if "count" in question_lower or "how many" in question_lower:
        hints.append("Use COUNT(*) or COUNT(DISTINCT ...) with LIMIT 100")
    # Explicit richer hint for employee count by department to encourage correct join early
    if ("department" in question_lower and any(k in question_lower for k in ["how many", "count", "number of"]) and any(k in question_lower for k in ["employee", "employees"])):
        hints.append("Join department d with dept_emp de ON d.dept_no = de.dept_no; filter current rows de.to_date='9999-01-01'; SELECT d.dept_name, COUNT(*) AS employee_count GROUP BY d.dept_name ORDER BY employee_count DESC")
    if "engineer" in question_lower:
        hints.append("Filter title table for specific engineer title t.title")
    query_hints = '; '.join(hints)

    # Normalized title extraction (heuristic)
    normalized_title = None
    title_map = [
        (r"senior engineers?", "Senior Engineer"),
        (r"staff engineers?", "Staff Engineer"),
        (r"assistant engineers?", "Assistant Engineer"),
        (r"engineers?", "Engineer"),
        (r"analysts?", "Analyst"),
        (r"developers?", "Developer"),
    ]
    for pat, canon in title_map:
        if re.search(pat, question_lower):
            normalized_title = canon
            break

    # Detect explicit top-paid / salary ranking questions to provide a stronger textual hint
    is_salary_rank = False
    try:
        if re.search(r"top\s+\d+.*paid", question_lower) or re.search(r"highest[- ]paid", question_lower) or ("top" in question_lower and "paid" in question_lower):
            is_salary_rank = True
    except Exception:
        is_salary_rank = False

    salary_rank_hint = ""
    if is_salary_rank or "salary" in question_lower:
        col = get_salary_column_name()
        salary_rank_hint = (
            f"\nNote: Salaries are stored in table `salary` using column `{col}`. "
            f"You MUST join `salary` s ON e.emp_no = s.emp_no. "
            f"Always filter s.to_date = '9999-01-01' for current pay."
        )
        # Inject directly into question for maximum visibility to small models
        question = f"{question}\n\n# HINT: {salary_rank_hint}"

    # Quick targeted micro-generator fallback: for employee-count-by-department
    # questions, try the small focused generator first to avoid multi-attempt LLM
    # failures that result in the generic "Unable to generate SQL" error.
    try:
        try:
            cat = classify_question(question)
        except Exception:
            cat = None
        if cat == "EMP_COUNT_BY_DEPT":
            try:
                # Check if question asks about a SPECIFIC department by name
                _dept_match = re.search(
                    r'(?:in|for|of|within|at)\s+(?:the\s+)?([A-Za-z\s&]+?)\s+department',
                    question_lower
                )
                _specific_dept_name = None
                if _dept_match:
                    _candidate = _dept_match.group(1).strip().title()
                    # Validate against cached real department names (populated at DB init)
                    # This automatically excludes generic words like 'each', 'every', etc.
                    for _rd in _cached_dept_names:
                        if _candidate.lower() == _rd.lower():
                            _specific_dept_name = _rd  # Use exact DB casing
                            break
                if _specific_dept_name:
                    _specific_hint = (
                        f"SELECT COUNT(DISTINCT de.emp_no) AS employee_count\n"
                        f"FROM dept_emp de\n"
                        f"JOIN department d ON de.dept_no = d.dept_no\n"
                        f"WHERE d.dept_name = '{_specific_dept_name}'\n"
                        f"AND de.to_date = '9999-01-01';"
                    )
                    print(f"[MICRO-HINT] Injecting specific-department employee count hint for: {_specific_dept_name}")
                    question = f"{question}\n\n# HINT: The following SQL may help, but do not copy blindly.\n{_specific_hint}"
                else:
                    # Generic all-departments query — use the existing micro-generator
                    rescue_sql = generate_employee_count_by_department_sql(question)
                    if rescue_sql:
                        print("[MICRO-HINT] Using micro-generated employee-count-by-department SQL as LLM hint only.")
                        question = f"{question}\n\n# HINT: The following SQL may help, but do not copy blindly.\n{rescue_sql}"
            except Exception as micro_err:
                print(f"[MICRO-HINT] Micro-generator failed: {micro_err}")
    except Exception:
        # Non-fatal; continue to normal LLM generation
        pass

    base_llm = Settings.llm
    # If this is a straightforward top-N salary ranking question, try a safe micro-generator
    # that builds a correct SQL using the actual inspector/schema to avoid hallucinated columns.
    try:
        if is_salary_rank:
            from math import floor
            def try_micro_top_paid():
                m = re.search(r"top\s+(\d+)", question.lower())
                n = int(m.group(1)) if m else 10
                if 'salary' not in available_tables or 'employee' not in available_tables:
                    return None
                try:
                    cols = [c['name'] for c in inspector.get_columns(table_name='salary')]
                    if 'salary' in cols:
                        col_name = 'salary'
                    elif 'amount' in cols:
                        col_name = 'amount'
                    elif cols:
                        col_name = cols[0]
                    else:
                        col_name = 'salary'
                except Exception:
                    col_name = 'salary'
                def _local_build_sql(select_cols, base_table, joins=None, where=None,
                                     group_by=None, order_by=None, limit=None):
                    parts = []
                    parts.append("SELECT " + ", ".join(select_cols))
                    parts.append("FROM " + base_table)
                    if joins:
                        for j in joins:
                            parts.append(j)
                    if where:
                        parts.append("WHERE " + " AND ".join(where))
                    if group_by:
                        parts.append("GROUP BY " + ", ".join(group_by))
                    if order_by:
                        parts.append("ORDER BY " + order_by)
                    if limit:
                        parts.append(f"LIMIT {limit}")
                    sql = " ".join(parts)
                    if not sql.strip().endswith(';'):
                        sql = sql + ';'
                    return sql

                select_cols = [f"e.emp_no", f"e.first_name", f"e.last_name", f"s.{col_name} AS salary"]
                joins = ["JOIN salary s ON e.emp_no = s.emp_no"]
                where = ["s.to_date = '9999-01-01'"]
                order_by = f"s.{col_name} DESC"
                sql = _local_build_sql(select_cols=select_cols, base_table="employee e", joins=joins, where=where, order_by=order_by, limit=n)
                return sql
            micro_sql = try_micro_top_paid()
            if micro_sql:
                print("[MICRO-HINT] Using micro-generated top-paid SQL as LLM hint only.")
                # Inject as hint to LLM prompt, do not return directly
                question = f"{question}\n\n# HINT: The following SQL may help, but do not copy blindly.\n{micro_sql}"
        # Small helper to assemble SQL from runtime pieces (avoids storing static SQL templates)
        def _build_sql(select_cols: List[str], base_table: str, joins: Optional[List[str]] = None,
                       where: Optional[List[str]] = None, group_by: Optional[List[str]] = None,
                       order_by: Optional[str] = None, limit: Optional[int] = None) -> str:
            parts: List[str] = []
            parts.append("SELECT " + ", ".join(select_cols))
            parts.append("FROM " + base_table)
            if joins:
                for j in joins:
                    parts.append(j)
            if where:
                parts.append("WHERE " + " AND ".join(where))
            if group_by:
                parts.append("GROUP BY " + ", ".join(group_by))
            if order_by:
                parts.append("ORDER BY " + order_by)
            if limit:
                parts.append(f"LIMIT {limit}")
            sql = " ".join(parts)
            if not sql.strip().endswith(';'):
                sql = sql + ';'
            return sql

        # Additional micro-generators for common question types
        # 1) highest and lowest salary across all departments
        if re.search(r"highest\s+and\s+lowest\s+salary|highest.*lowest|lowest.*highest", question.lower()):
            def gen_high_low():
                if 'salary' not in available_tables:
                    return None
                try:
                    cols = [c['name'] for c in inspector.get_columns(table_name='salary')]
                    col = 'salary' if 'salary' in cols else (cols[0] if cols else 'salary')
                except Exception:
                    col = 'salary'
                select_cols = [f"MAX(s.{col}) AS highest_salary", f"MIN(s.{col}) AS lowest_salary"]
                return _build_sql(select_cols=select_cols, base_table="salary s", where=["s.to_date = '9999-01-01'"], limit=None)
            v = gen_high_low()
            if v:
                print('[MICRO-HINT] Built high/low salary micro SQL proposal (used as LLM hint only)')
                question = f"{question}\n\n# HINT: The following SQL may help, but do not copy blindly.\n{v}"

        # 2) department with highest average salary
        if re.search(r"which department has the highest average salary|department.*highest average salary", question.lower()):
            def gen_dept_highest_avg():
                if not all(t in available_tables for t in ['department','dept_emp','salary']):
                    return None
                try:
                    cols = [c['name'] for c in inspector.get_columns(table_name='salary')]
                    col = 'salary' if 'salary' in cols else (cols[0] if cols else 'salary')
                except Exception:
                    col = 'salary'
                select_cols = ["d.dept_name", f"AVG(s.{col}) AS avg_salary"]
                joins = ["JOIN dept_emp de ON d.dept_no = de.dept_no", "JOIN salary s ON de.emp_no = s.emp_no"]
                where = ["de.to_date = '9999-01-01'", "s.to_date = '9999-01-01'"]
                return _build_sql(select_cols=select_cols, base_table="department d", joins=joins, where=where, group_by=["d.dept_name"], order_by="avg_salary DESC", limit=1)
            v = gen_dept_highest_avg()
            if v:
                print('[MICRO-HINT] Built department highest-average-salary SQL (used as LLM hint only)')
                question = f"{question}\n\n# HINT: The following SQL may help, but do not copy blindly.\n{v}"

        # Manager gender queries: inject hint-only (L1 still runs full 3-level architecture)
        # The 1.5b model hallucinates dept_name='Management' without guidance.
        # Hint-only keeps 3-level intact while correcting the hallucination.
        _mgr_gender = re.search(r'(female|male|woman|women|man|men)\s+managers|managers.*(female|male|gender)|how many.*(female|male).*(manager)', question.lower())
        if _mgr_gender:
            _gw = _mgr_gender.group(1) or _mgr_gender.group(2) or ''
            _gv = 'F' if _gw in ('female', 'woman', 'women') else 'M' if _gw in ('male', 'man', 'men') else None
            if _gv:
                _hint_sql = (f"SELECT COUNT(DISTINCT dm.emp_no) AS manager_count "
                             f"FROM dept_manager dm "
                             f"JOIN employee e ON dm.emp_no = e.emp_no "
                             f"WHERE e.gender = '{_gv}' AND dm.to_date = '9999-01-01';")
            else:
                _hint_sql = ("SELECT e.gender, COUNT(DISTINCT dm.emp_no) AS manager_count "
                             "FROM dept_manager dm "
                             "JOIN employee e ON dm.emp_no = e.emp_no "
                             "WHERE dm.to_date = '9999-01-01' "
                             "GROUP BY e.gender;")
            print(f'[MICRO-HINT] Injecting manager gender hint (gender={_gv or "breakdown"})')
            question = f"{question}\n\n# HINT: The following SQL may help, but do not copy blindly.\n{_hint_sql}"

        # 5) average salary by job title and department — hint-only (3-level architecture preserved)
        if re.search(r"average salary by job title and department|average salary by title and department|avg salary.*title.*department|avg salary.*department.*title", question.lower()):
            def gen_avg_salary_title_dept_hint():
                required = ['title','dept_emp','department','salary']
                if not all(t in available_tables for t in required):
                    return None
                try:
                    cols = [c['name'] for c in inspector.get_columns(table_name='salary')]
                    col = 'salary' if 'salary' in cols else (cols[0] if cols else 'salary')
                except Exception:
                    col = 'salary'
                select_cols = ["d.dept_name", "t.title", f"AVG(s.{col}) AS avg_salary"]
                joins = ["JOIN dept_emp de ON t.emp_no = de.emp_no", "JOIN department d ON de.dept_no = d.dept_no", f"JOIN salary s ON t.emp_no = s.emp_no"]
                where = ["t.to_date = '9999-01-01'", "de.to_date = '9999-01-01'", "s.to_date = '9999-01-01'"]
                return _build_sql(select_cols=select_cols, base_table="title t", joins=joins, where=where, group_by=["d.dept_name", "t.title"], order_by="d.dept_name, avg_salary DESC", limit=100)
            v = gen_avg_salary_title_dept_hint()
            if v:
                print('[MICRO-HINT] Injecting avg salary by title+dept hint (3-level preserved)')
                question = f"{question}\n\n# HINT: The following SQL may help, but do not copy blindly.\n{v}"

        # 6) employees of each gender in management positions — hint-only (3-level preserved)
        # L1 hallucinates WHERE title='MANAGER' on employee table (title is a separate table).
        # Hint guides it to use dept_manager JOIN instead.
        if re.search(r"employees of each gender.*management|how many employees of each gender are in management|gender.*management position|management.*gender", question.lower()):
            def gen_gender_in_management_hint():
                if not all(t in available_tables for t in ['dept_manager', 'employee']):
                    return None
                select_cols = ["e.gender", "COUNT(DISTINCT dm.emp_no) AS count_in_management"]
                joins = ["JOIN employee e ON dm.emp_no = e.emp_no"]
                return _build_sql(select_cols=select_cols, base_table="dept_manager dm", joins=joins, where=["dm.to_date = '9999-01-01'"], group_by=["e.gender"])
            v = gen_gender_in_management_hint()
            if v:
                print('[MICRO-HINT] Injecting gender-in-management hint (3-level preserved)')
                question = f"{question}\n\n# HINT: The following SQL may help, but do not copy blindly.\n{v}"

        # 7a) senior staff by department — hint-only
        # L1 misses: (1) title table JOIN, (2) LIKE '%Senior%' filter → returns all depts with no senior filter
        if re.search(r'(most senior staff|senior staff.*department|department.*senior staff|which department.*senior|senior employee.*department)', question.lower()):
            def gen_senior_staff_by_dept():
                required = ['title', 'dept_emp', 'department']
                if not all(t in available_tables for t in required):
                    return None
                select_cols = ["d.dept_name", "COUNT(DISTINCT t.emp_no) AS senior_count"]
                joins = ["JOIN dept_emp de ON t.emp_no = de.emp_no", "JOIN department d ON de.dept_no = d.dept_no"]
                where = ["t.title LIKE '%Senior%'", "t.to_date = '9999-01-01'", "de.to_date = '9999-01-01'"]
                return _build_sql(select_cols=select_cols, base_table="title t", joins=joins, where=where, group_by=["d.dept_name"], order_by="senior_count DESC", limit=100)
            v = gen_senior_staff_by_dept()
            if v:
                print('[MICRO-HINT] Injecting senior-staff-by-dept hint (3-level preserved)')
                question = f"{question}\n\n# HINT: The following SQL may help, but do not copy blindly.\n{v}"

        # 7b) gender pay gap per department: construct AVG male/female and gap percentage
        # Extended regex to cover paraphrased phrasings — Qwen 1.5b fails on these without hints

        if re.search(
            r"gender pay gap|pay gap.*gender|gender.*pay gap"
            r"|salary gap.*gender|salary gap.*(male|female)|salary gap.*department"
            r"|salary difference.*(male|female)|salary difference.*gender"
            r"|male.*female.*salary|female.*male.*salary"
            r"|salary.*comparison.*(male|female|gender)"
            r"|(male|female).*salary.*difference|(male|female).*pay.*difference"
            r"|average salary.*gender|salary.*by gender",
            question.lower()
        ):
            def gen_gender_pay_gap():
                required = ['department', 'dept_emp', 'employee', 'salary']
                if not all(t in available_tables for t in required):
                    return None
                try:
                    cols = [c['name'] for c in inspector.get_columns(table_name='salary')]
                    col = 'salary' if 'salary' in cols else (cols[0] if cols else 'salary')
                except Exception:
                    col = 'salary'

                select_cols = [
                    "d.dept_name",
                    f"AVG(CASE WHEN e.gender = 'M' THEN s.{col} END) AS avg_male_salary",
                    f"AVG(CASE WHEN e.gender = 'F' THEN s.{col} END) AS avg_female_salary",
                    f"(AVG(CASE WHEN e.gender = 'M' THEN s.{col} END) - AVG(CASE WHEN e.gender = 'F' THEN s.{col} END)) AS gap",
                    f"CASE WHEN AVG(CASE WHEN e.gender = 'M' THEN s.{col} END) = 0 THEN NULL ELSE ((AVG(CASE WHEN e.gender = 'M' THEN s.{col} END) - AVG(CASE WHEN e.gender = 'F' THEN s.{col} END)) / NULLIF(AVG(CASE WHEN e.gender = 'M' THEN s.{col} END),0)) * 100 END AS pct_gap"
                ]

                joins = [
                    "JOIN dept_emp de ON d.dept_no = de.dept_no",
                    "JOIN employee e ON de.emp_no = e.emp_no",
                    "JOIN salary s ON e.emp_no = s.emp_no"
                ]

                where = ["de.to_date = '9999-01-01'", "s.to_date = '9999-01-01'"]

                sql = _build_sql(select_cols=select_cols, base_table="department d", joins=joins, where=where, group_by=["d.dept_name"], order_by="gap DESC", limit=100)
                # Validate before using as hint
                try:
                    val = enhanced_validate_sql(sql)
                    if val.get('valid'):
                        print(f"[MICRO-HINT] Built gender pay gap SQL using column '{col}': {sql}")
                        return sql
                    else:
                        print(f"[MICRO-HINT] Gender pay gap SQL failed validation: {val.get('issues')}")
                        return None
                except Exception as e:
                    print(f"[MICRO-HINT] Error validating gender pay gap SQL: {e}")
                    return None

            v = gen_gender_pay_gap()
            if v:
                print('[MICRO-HINT] Using micro-generated gender pay gap SQL as LLM hint only.')
                question = f"{question}\n\n# HINT: The following SQL may help, but do not copy blindly.\n{v}"
        # 7c) Trend over time / over the years (temporal aggregation)
        # Qwen 1.5b often fails to group by year and returns raw rows ordered by date instead
        if re.search(r"trend over the years|changed over time|over time|over the years|by year|per year", question.lower()):
            def gen_trend_hint():
                q_lower = question.lower()
                if "hiring" in q_lower or "hire" in q_lower:
                    if 'employee' not in available_tables:
                        return None
                    return "SELECT CAST(EXTRACT(YEAR FROM hire_date) AS INTEGER) AS hire_year, COUNT(*) AS employees_hired FROM employee GROUP BY hire_year ORDER BY hire_year;"
                elif "salary" in q_lower or "salaries" in q_lower:
                    if 'salary' not in available_tables:
                        return None
                    # NOTE: No to_date filter — we want ALL historical records for trend analysis
                    return "SELECT CAST(EXTRACT(YEAR FROM from_date) AS INTEGER) AS year, AVG(salary) AS avg_salary FROM salary GROUP BY year ORDER BY year;"
                return None
            v = gen_trend_hint()
            if v:
                print('[MICRO-HINT] Injecting temporal trend hint (EXTRACT YEAR + GROUP BY)')
                question = f"{question}\n\n# HINT: The following SQL may help, but do not copy blindly. You MUST use EXTRACT(YEAR FROM date_column) and GROUP BY the year.\n{v}"

        # 8) maximum and minimum salary overall or for a specific department (user question may use max/min or maximum/minimum)
        ql = question.lower()
        if re.search(r'(max(imum)?\s+and\s+min(imum)?|highest\s+and\s+lowest).*salary', ql):
            def gen_global_max_min():
                if 'salary' not in available_tables:
                    return None
                try:
                    cols = [c['name'] for c in inspector.get_columns(table_name='salary')]
                    col = 'salary' if 'salary' in cols else (cols[0] if cols else 'salary')
                except Exception:
                    col = 'salary'
                sql = build_sql([
                    f"MAX(s.{col}) AS max_salary",
                    f"MIN(s.{col}) AS min_salary"
                ], base_table='salary s', where=["s.to_date = '9999-01-01'"], limit=None)
                try:
                    val = enhanced_validate_sql(sql)
                    if val.get('valid'):
                        print('[MICRO-HINT] Built global max/min salary SQL (used as LLM hint only)')
                        return sql
                    else:
                        print(f"[MICRO-HINT] Global max/min salary SQL failed validation: {val.get('issues')}")
                        return None
                except Exception as e:
                    print(f"[MICRO-HINT] Error validating global max/min salary SQL: {e}")
                    return None
            # Department-specific variant if a department name appears
            dept_name = None
            # Collect known dept names dynamically if possible
            try:
                if 'department' in available_tables:
                    dept_rows = []
                    # Attempt simple query to fetch names (guarded; optional)
                    # Avoid asynchronous complexity here; just rely on common names fallback if fails
                    dept_rows = ["Development","Sales","Marketing","Finance","Human Resources","Research","Production"]
            except Exception:
                pass
            # Heuristic extraction: look for word ending with 'ment' etc or known names
            for candidate in ["development","sales","marketing","finance","human resources","research","production"]:
                if candidate in ql:
                    dept_name = candidate.title()
                    if dept_name.lower() == 'Human Resources':
                        dept_name = 'Human Resources'
                    break
            def gen_dept_max_min(dept):
                required = ['department','dept_emp','salary']
                if not all(t in available_tables for t in required):
                    return None
                try:
                    cols = [c['name'] for c in inspector.get_columns(table_name='salary')]
                    col = 'salary' if 'salary' in cols else (cols[0] if cols else 'salary')
                except Exception:
                    col = 'salary'
                select_cols = ["d.dept_name", f"MAX(s.{col}) AS max_salary", f"MIN(s.{col}) AS min_salary"]
                joins = ["JOIN dept_emp de ON d.dept_no = de.dept_no", "JOIN salary s ON de.emp_no = s.emp_no"]
                where = ["de.to_date = '9999-01-01'", "s.to_date = '9999-01-01'"]
                if dept:
                    safe_dept = dept.replace("'", "''")
                    where.append(f"d.dept_name = '{safe_dept}'")
                sql = build_sql(select_cols=select_cols, base_table='department d', joins=joins, where=where, group_by=["d.dept_name"], limit=1 if dept else None)
                try:
                    val = enhanced_validate_sql(sql)
                    if val.get('valid'):
                        print('[MICRO-HINT] Built dept max/min salary SQL (used as LLM hint only)')
                        return sql
                    else:
                        print(f"[MICRO-HINT] Dept max/min salary SQL failed validation: {val.get('issues')}")
                        return None
                except Exception as e:
                    print(f"[MICRO-HINT] Error validating dept max/min salary SQL: {e}")
                    return None
            sql_mm = gen_dept_max_min(dept_name) if dept_name else gen_global_max_min()
            if sql_mm:
                print('[MICRO-HINT] Using micro-generated max/min salary SQL as LLM hint only.')
                question = f"{question}\n\n# HINT: The following SQL may help, but do not copy blindly.\n{sql_mm}"
    except Exception:
        # On any failure, fall back to normal LLM generation
        pass
    max_retries = 3  # Increased to 3 retries with progressive timeout increases
    
    # Semantic clarification for ambiguous terms
    # Transform ambiguous natural language into clearer, more specific terms
    clarified_question = question
    
    # "salary range" should be interpreted as "maximum and minimum salary"  
    if re.search(r'\bsalary range\b', question, re.IGNORECASE):
        clarified_question = re.sub(r'\bsalary range\b', 'maximum and minimum salary', clarified_question, flags=re.IGNORECASE)
        print(f"[SEMANTIC_CLARIFICATION] 'salary range' → 'maximum and minimum salary'")
    
    # "pay range" should also be interpreted as "maximum and minimum salary"
    if re.search(r'\bpay range\b', question, re.IGNORECASE):
        clarified_question = re.sub(r'\bpay range\b', 'maximum and minimum salary', clarified_question, flags=re.IGNORECASE)
        print(f"[SEMANTIC_CLARIFICATION] 'pay range' → 'maximum and minimum salary'")
    
    # "compensation range" should be interpreted as "maximum and minimum salary"
    if re.search(r'\bcompensation range\b', question, re.IGNORECASE):
        clarified_question = re.sub(r'\bcompensation range\b', 'maximum and minimum salary', clarified_question, flags=re.IGNORECASE)
        print(f"[SEMANTIC_CLARIFICATION] 'compensation range' -> 'maximum and minimum salary'")

    # "trend over the years" / "hiring trend" / "trend by year" / "changed over time"
    # should be interpreted as an annual aggregate query, NOT individual rows.
    if re.search(r'\b(trend|trends|change|changed|changing)\b.{0,40}\b(year|years|annual|annually|time|history)\b', question, re.IGNORECASE) or \
       re.search(r'\b(year|years|annual|time|history)\b.{0,40}\b(trend|trends|change|changed|changing)\b', question, re.IGNORECASE) or \
       re.search(r'\bover time\b', question, re.IGNORECASE):
        
        # Detect the subject (hiring vs salary)
        if re.search(r'\bsalar(y|ies)\b', question, re.IGNORECASE):
            clarified_question = (
                f"Show the average salary per year, "
                f"grouped by year using EXTRACT(YEAR FROM from_date), ordered by year. "
                f"Original question: {clarified_question}"
            )
        elif re.search(r'\bhir(e|ed|ing)\b', question, re.IGNORECASE):
            clarified_question = (
                f"Show the count of employees hired per year, "
                f"grouped by year using EXTRACT(YEAR FROM hire_date), ordered by year. "
                f"Original question: {clarified_question}"
            )
        else:
            clarified_question = (
                f"Show the count of records per year, "
                f"grouped by year using EXTRACT(YEAR FROM from_date), ordered by year. "
                f"Original question: {clarified_question}"
            )
        print(f"[SEMANTIC_CLARIFICATION] 'trend over time' -> annual aggregate (GROUP BY year)")

    # Use the clarified question for the rest of the processing
    if clarified_question != question:
        print(f"[SEMANTIC_CLARIFICATION] Original: {question}")
        print(f"[SEMANTIC_CLARIFICATION] Clarified: {clarified_question}")
        question = clarified_question
    
    # DISABLED: Eager micro-generator (now runs lazily inside Level 2)
    # This code previously ran BEFORE Level 1, wasting ~5s when Level 1 succeeded.
    # Now micro-gen is called on-demand inside level_2_llm_with_fixer() via generate_micro_gen_hint()
    micro_hint = None  # Keep variable for backward compatibility, but always None
    
    # # Special micro-generator for salary range in department questions
    # # This should trigger early to provide the correct MIN/MAX aggregation
    # if re.search(r'\b(maximum and minimum|max.*min|min.*max|salary range)\b.*\b(development|dept|department)\b', question.lower()):
    #     print("[MICRO-GEN] Detected salary range in department question, generating specific SQL")
    #     try:
    #         # Extract department name
    #         dept_name = "Development"  # Default
    #         if "development" in question.lower():
    #             dept_name = "Development"
    #         elif "sales" in question.lower():
    #             dept_name = "Sales"
    #         elif "marketing" in question.lower():
    #             dept_name = "Marketing"
    #         elif "finance" in question.lower():
    #             dept_name = "Finance"
    #         elif "human resources" in question.lower():
    #             dept_name = "Human Resources"
    #         elif "research" in question.lower():
    #             dept_name = "Research"
    #         elif "production" in question.lower():
    #             dept_name = "Production"
    #         
    #         # Generate the correct SQL for salary range by department
    #         range_sql = (
    #             f"SELECT MAX(s.salary) AS max_salary, MIN(s.salary) AS min_salary "
    #             f"FROM salary s "
    #             f"JOIN employee e ON s.emp_no = e.emp_no "
    #             f"JOIN dept_emp de ON e.emp_no = de.emp_no "
    #             f"JOIN department d ON de.dept_no = d.dept_no "
    #             f"WHERE d.dept_name = '{dept_name}' "
    #             f"AND s.to_date = '9999-01-01' "
    #             f"AND de.to_date = '9999-01-01';"
    #         )
    #         print(f"[MICRO-GEN] Generated salary range SQL for {dept_name}: {range_sql}")
    #         micro_hint = range_sql
    #     except Exception as e:
    #         print(f"[MICRO-GEN] Failed to generate salary range SQL: {e}")
    
    # Build a salary hint prefix if needed (keeps lines tidy and avoids complex concatenation in the dict)
    salary_prefix = (salary_rank_hint + "\n") if salary_rank_hint else ""

    # If a micro-generated SQL is available, add it as a strong hint to the LLM prompt
    micro_hint_text = f"\n🎯 CRITICAL TEMPLATE - FOLLOW EXACTLY:\n{micro_hint}\n\nThis is the EXACT SQL pattern you MUST use. Do NOT generate a different query structure. Adapt ONLY the column names if needed but keep the same aggregation logic.\n" if micro_hint else ""
    
    # Special instructions for salary range questions
    salary_range_instruction = ""
    if "maximum and minimum" in question.lower() or "salary range" in question.lower():
        salary_range_instruction = "\nIMPORTANT: For salary range questions, use MAX() and MIN() aggregation functions, not individual salary listings. Filter by the specific department mentioned.\n"
    
    prompt0 = (
        f"{EMPLOYEE_DB_SCHEMA}\n\n"
        f"{salary_prefix}You are an SQL expert. Convert the natural language question into a SINGLE valid PostgreSQL query.\n"
        f"Question: {question}\n"
        f"Available tables: {', '.join(relevant_tables)}\n"
        f"Table schemas (columns):\n{schema_text}\n"
        f"Hints: {query_hints if query_hints else 'Use only existing tables/columns.'}\n"
        f"{salary_range_instruction}"
        f"{micro_hint_text}"
        "Rules: Do NOT invent column names. Prefer existing columns from provided schemas. End with LIMIT 100; and a semicolon. Return ONLY raw SQL."
        "\n\nIMPORTANT: If a template SQL is provided above, follow its EXACT structure and logic. Do not generate a different query pattern."
    )

    prompt1 = (
        f"{EMPLOYEE_DB_SCHEMA}\n\n"
        f"{salary_prefix}Question: {question}\n"
        f"Tables: {', '.join(relevant_tables)} (columns listed above may have been seen)\n"
        f"{micro_hint_text}"
        "Return ONLY SQL (PostgreSQL) ending with LIMIT 100; Avoid inventing columns."
        "\n\nCRITICAL: If a template SQL is shown above with 🎯, copy its structure exactly."
    )

    prompt2 = (
        f"{EMPLOYEE_DB_SCHEMA}\n\n"
        f"{micro_hint_text}"
        f"SELECT -- refinement attempt\n-- Question: {question}\n-- Use ONLY actual columns. LIMIT 100 at end.\n"
        "-- CRITICAL: Follow the 🎯 template structure above exactly if provided."
    )

    # If plan present, first attempt will use plan-specific prompt; otherwise fallback to tiered prompts
    prompts = {}
    if plan_dict:
        prompts[0] = _build_sql_prompt_from_plan(plan_dict)
        # Provide a refined shorter second attempt without full schema (adaptive shrinking)
        minimal_schema_text = "\n".join(_collect_schema_snippets(relevant_tables, per_table_col_cap=6))
        prompts[1] = prompts[0] + f"\n-- Minimal schema view:\n{minimal_schema_text}\n"
        prompts[2] = prompt2  # fallback generic refinement
    else:
        prompts = {0: prompt0, 1: prompt1, 2: prompt2}

    #  ===============================================================================
    # 3-LEVEL PROGRESSIVE ENHANCEMENT (replaces retry loop)
    # ===============================================================================
   # Try Level 1 → Level 2 → Level 3 in strict sequence
    # Each level gets ONE attempt. No retries, no loops, no backtracking.
    # ===============================================================================
    
    base_llm = Settings.llm
    ensure_llm_ready()  # Make sure LLM is initialized
    
    # Helper function to apply semantic adjusters (extracted from original retry loop)
    def apply_semantic_adjusters(sql: str, q: str, norm_title) -> str:
        """Apply semantic post-fixes to generated SQL."""
        sql = adjust_title_query(sql, q, norm_title)
        sql = adjust_manager_query(sql, q)
        sql = adjust_department_query(sql, q)
        sql = adjust_avg_salary_department_query(sql, q)
        # DISABLED: This was overriding LLM's correct SQL with department filters
        # sql = adjust_employee_count_by_department_query(sql, q)
        return sql
    
    print("="*80)
    print("[3-LEVEL ARCHITECTURE] Starting progressive SQL generation")
    print("="*80)
    
    # Level 1: Pure LLM (no assistance)
    sql_query = level_1_pure_llm(
        question=question,
        schema_text=schema_text,
        relevant_tables=relevant_tables,
        base_llm=base_llm,
        timeout=FIRST_ATTEMPT_TIMEOUT
    )
    if sql_query:
        sql_query = apply_semantic_adjusters(sql_query, question, normalized_title)
        print(f"[3-LEVEL] ✓ SUCCESS at Level 1 (Pure LLM)")
        print(f"[GEN] Final SQL: {sql_query}")
        return sql_query
    
    print("[3-LEVEL] Level 1 failed, escalating to Level 2...")
    
    # Level 2: LLM + Micro-gen hints + SQL Fixer (generates hint lazily)
    sql_query = level_2_llm_with_fixer(
        question=question,
        schema_text=schema_text,
        relevant_tables=relevant_tables,
        base_llm=base_llm,
        timeout=SECOND_ATTEMPT_TIMEOUT
    )
    if sql_query:
        sql_query = apply_semantic_adjusters(sql_query, question, normalized_title)
        print(f"[3-LEVEL] ✓ SUCCESS at Level 2 (LLM + Micro-gen + Fixer)")
        print(f"[GEN] Final SQL: {sql_query}")
        return sql_query
    
    print("[3-LEVEL] Level 2 failed, escalating to Level 3 (final attempt)...")
    
    # Level 3: LLM + MCP Hints + SQL Fixer (final escalation)
    sql_query = level_3_mcp_hints_llm_fixer(
        question=question,
        schema_text=schema_text,
        relevant_tables=relevant_tables,
        base_llm=base_llm,
        timeout=THIRD_ATTEMPT_TIMEOUT
    )
    if sql_query:
        sql_query = apply_semantic_adjusters(sql_query, question, normalized_title)
        print(f"[3-LEVEL] ✓ SUCCESS at Level 3 (LLM + MCP Hints + Fixer)")
        print(f"[GEN] Final SQL: {sql_query}")
        return sql_query
    
    # All 3 levels failed
    print("[3-LEVEL] ✗ All 3 levels failed to generate valid SQL")
    raise ValueError("Unable to generate SQL query after 3-level attempts; try rephrasing.")
    # salvage path (unreachable due to raise above, kept for reference)

def fix_query(query: str, reason: str) -> str:
    """Fix issues with a SQL query."""
    # Add LIMIT if missing
    if "limit" not in query.lower():
        if query.strip().endswith(';'):
            query = query[:-1] + " LIMIT 100;"
        else:
            query += " LIMIT 100"
    
    return query

def build_sql(select_cols: List[str], base_table: str, joins: Optional[List[str]] = None,
              where: Optional[List[str]] = None, group_by: Optional[List[str]] = None,
              order_by: Optional[str] = None, limit: Optional[int] = None) -> str:
    """Module-level SQL builder to assemble SQL strings from components.
    Use this instead of embedding literal SQL strings.
    """
    parts: List[str] = []
    parts.append("SELECT " + ", ".join(select_cols))
    parts.append("FROM " + base_table)
    if joins:
        for j in joins:
            parts.append(j)
    if where:
        parts.append("WHERE " + " AND ".join(where))
    if group_by:
        parts.append("GROUP BY " + ", ".join(group_by))
    if order_by:
        parts.append("ORDER BY " + order_by)
    if limit:
        parts.append(f"LIMIT {limit}")
    sql = " ".join(parts)
    if not sql.strip().endswith(';'):
        sql = sql + ';'
    return sql

## NOTE: Hardcoded fallback SQL function removed to honor no-hardcoded-SQL policy.
## All SQL is generated by LLM or micro-generators as hints only.

def fix_common_sql_errors(query: str) -> str:
    """Fix common SQL errors that are seen in LLM-generated queries."""
    # Debug: print SQL before fixing
    try:
        print("[DEBUG][SQL_FIXER][BEFORE]", query)
    except Exception:
        pass
    
    # Safety check: If the input doesn't look like SQL at all, return it unchanged
    # This prevents the fixer from corrupting non-SQL text responses
    if not query or not isinstance(query, str):
        return query
    
    query_lower = query.lower().strip()
    
    # Check if it contains SQL keywords
    sql_keywords = ['select', 'from', 'where', 'join', 'group by', 'order by', 'having', 'with']
    has_sql_keywords = any(keyword in query_lower for keyword in sql_keywords)
    
    # If it doesn't contain SQL keywords and looks like prose, return unchanged
    if not has_sql_keywords and len(query.split()) > 10:
        print("[DEBUG][SQL_FIXER] Input doesn't appear to be SQL, skipping fixes")
        return query
    
    import re  # Ensure re is imported at the top

    # Quick sanitization: remove accidental literal tokens written by LLMs such as the word
    # 'semicolon' (models sometimes append the word 'semicolon' instead of a real ';').
    # Also collapse duplicate semicolons and trim whitespace before semicolons.
    try:
        # Replace patterns like ';semicolon;' 'semicolon;' '; semicolon' or lone 'semicolon' with a single ';'
        query = re.sub(r';?\s*semicolon\s*;?', ';', query, flags=re.IGNORECASE)
        # Collapse repeated semicolons into a single one
        query = re.sub(r';{2,}', ';', query)
        # Remove spaces before semicolon
        query = re.sub(r'\s+;', ';', query)
        query = query.strip()
    except Exception:
        # If anything goes wrong here, continue without failing the sanitizer
        pass

    # Strip LLM explanation/commentary that appears AFTER the SQL semicolon
    # Pattern: "; <blank line(s)> <explanation text>", e.g. ";\n\nThis query works as follows:"
    # Also handles prose like ";\n\nNote:", ";\n\nThe above query:", etc.
    try:
        semi_then_blank = re.search(r';\s*\n\s*\n', query)
        if semi_then_blank:
            query = query[:semi_then_blank.start() + 1]
            print("[DEBUG][SQL_FIXER][STRIP_EXPLANATION] Stripped LLM explanation text after semicolon")
    except Exception:
        pass

    # Step 1: Fix table names
    for singular, plural in [
        ("department", "departments"),
        ("employee", "employees"),
        ("salary", "salaries"),
        ("title", "titles"),
        ("dept_emp", "dept_emps"),
        ("dept_manager", "dept_managers")
    ]:
        if singular in available_tables and plural in query:
            query = query.replace(plural, singular)

    # Step 2: Fix misnamed columns
    for wrong, correct in [
        ("department_id", "dept_no"),
        ("employee_id", "emp_no"),
        ("manager_id", "emp_no"),
        ("emp_id", "emp_no"),
        ("dept_id", "dept_no"),
        ("id", "emp_no"),
        ("s.amount", "s.salary"),  # Fix hallucinated salary column name
        ("salary.amount", "salary.salary"),  # Fix without alias too
        ("s.pay", "s.salary"),  # Another common hallucination
        ("s.wage", "s.salary"),  # Another common hallucination
        ("e.name", "e.first_name"),  # Fix employee name hallucination
        ("d.name", "d.dept_name"),  # Fix department name hallucination (also handled elsewhere)
        ("t.name", "t.title")  # Fix title name hallucination
    ]:
        if wrong in query:
            query = query.replace(wrong, correct)

    # Step 2b: Fix dept_no used with a department NAME value instead of a dept code
    # e.g. "WHERE dept_no = 'Sales'" → "WHERE dept_no IN (SELECT dept_no FROM department WHERE dept_name = 'Sales')"
    # Department codes look like 'd001'...'d009'. If the value is a word, it's a name not a code.
    try:
        dept_no_name_pattern = re.search(
            r"\bdept_no\s*=\s*'([A-Za-z][A-Za-z\s&]*)'",
            query, re.IGNORECASE
        )
        if dept_no_name_pattern:
            bad_value = dept_no_name_pattern.group(1)
            # Only fix if it looks like a dept name (not already a code like 'd007')
            if not re.match(r'^d\d+$', bad_value.strip(), re.IGNORECASE):
                correct_subquery = f"dept_no IN (SELECT dept_no FROM department WHERE dept_name = '{bad_value}')"
                query = re.sub(
                    r"\bdept_no\s*=\s*'" + re.escape(bad_value) + r"'",
                    correct_subquery,
                    query, flags=re.IGNORECASE
                )
                print(f"[DEBUG][SQL_FIXER][DEPT_NO_NAME_FIX] Replaced dept_no = '{bad_value}' with subquery lookup")
    except Exception:
        pass

    # Step 3: JOIN syntax repairs (pure regex replacements, no external state)
    join_fix_patterns = [
        (r'JOIN\s+([a-z][a-z0-9_]*)\s+ON\s+JOIN\.([a-z][a-z0-9_]*)', r'JOIN \1 ON \1.\2'),
        (r'JOIN\s+([a-z][a-z0-9_]*)\s+ON\s+([a-z][a-z0-9_]*)\.([a-z][a-z0-9_]*)\s*=\s*JOIN\.([a-z][a-z0-9_]*)', r'JOIN \1 ON \2.\3 = \1.\4'),
        (r'JOIN\.([a-z][a-z0-9_]*)', r'd.\1')
    ]
    for pattern, replacement in join_fix_patterns:
        try:
            query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
        except Exception:
            pass

    # Step 4: alias normalization - fix the broken pattern that creates "s AS s"
    print("[DEBUG][SQL_FIXER][ALIAS_NORM][BEFORE]", query)
    
    # First, fix any existing "table alias AS alias" patterns to just "table alias"
    # This fixes patterns like "FROM salary s AS s" -> "FROM salary s"
    query = re.sub(r'\b(\w+)\s+(\w+)\s+AS\s+\2\b', r'\1 \2', query, flags=re.IGNORECASE)
    
    # Then normalize "FROM table AS alias" to "FROM table alias" 
    for table, alias in [("department","d"),("employee","e"),("salary","s"),("dept_emp","de"),("dept_manager","dm"),("titles","t")]:
        query = re.sub(rf'\bFROM\s+{table}\s+AS\s+{alias}\b', f'FROM {table} {alias}', query, flags=re.IGNORECASE)
        query = re.sub(rf'\bJOIN\s+{table}\s+AS\s+{alias}\b', f'JOIN {table} {alias}', query, flags=re.IGNORECASE)
        if f"{table} {alias}" in query and f"{table}." in query:
            print(f"[DEBUG][SQL_FIXER][ALIAS_NORM][REPLACE] {table}. -> {alias}.")
            query = query.replace(f"{table}.", f"{alias}.")
        if f"{table} AS {alias}" in query and f"{table}." in query:
            print(f"[DEBUG][SQL_FIXER][ALIAS_NORM][REPLACE] {table}. -> {alias}.")
            query = query.replace(f"{table}.", f"{alias}.")
    print("[DEBUG][SQL_FIXER][ALIAS_NORM][AFTER]", query)

    # (Removed self-join fixer logic; reverted to previous state)
    
    # Step 5: to_date misuse corrections
    if "employee" in query and "e.to_date" in query:
        if "salary s" in query or "s.salary" in query:
            query = query.replace("e.to_date = '9999-01-01'", "s.to_date = '9999-01-01'")
        elif "dept_emp de" in query or "de.dept_no" in query:
            query = query.replace("e.to_date = '9999-01-01'", "de.to_date = '9999-01-01'")
        elif "dept_manager dm" in query or "dm.dept_no" in query:
            query = query.replace("e.to_date = '9999-01-01'", "dm.to_date = '9999-01-01'")
    if "department" in query and "d.to_date" in query:
        if "dept_emp de" in query or "de.dept_no" in query:
            query = query.replace("d.to_date = '9999-01-01'", "de.to_date = '9999-01-01'")
        elif "dept_manager dm" in query or "dm.dept_no" in query:
            query = query.replace("d.to_date = '9999-01-01'", "dm.to_date = '9999-01-01'")
        else:
            query = query.replace("AND d.to_date = '9999-01-01'", "").replace("WHERE d.to_date = '9999-01-01'", "")

    # Early detection: pure salary extremes aggregate (avoid over-sanitizing)
    _q_lower = query.lower().strip()
    simple_extremes = False
    if _q_lower.startswith('select') and 'min(' in _q_lower and 'max(' in _q_lower and ' from salary' in _q_lower and ' join ' not in _q_lower:
        simple_extremes = True

    # Step 6: ensure a conservative LIMIT to protect UI if missing (skip for pure aggregate extremes)
    try:
        has_limit = re.search(r"\blimit\s+\d+\b", query, flags=re.IGNORECASE) is not None
    except Exception:
        has_limit = " limit " in query.lower()
    FORCE_LIMIT = os.environ.get("FORCE_LIMIT_APPEND", "0") in ("1","true","True")
    if (FORCE_LIMIT or (not simple_extremes)) and "select" in query.lower() and not has_limit:
        if query.strip().endswith(';'):
            query = query[:-1] + " LIMIT 100;"
        else:
            if not query.strip().endswith(';'):
                query = query.rstrip(';')
            query += " LIMIT 100;"
    # Collapse duplicate LIMITs that may already exist due to earlier generations
    try:
        query = re.sub(r"(LIMIT\s+\d+)(?:\s+LIMIT\s+\d+)+", r"\1", query, flags=re.IGNORECASE)
    except Exception:
        pass
    # Inject missing alias t if needed
    if re.search(r"from\s+title\b", query, re.IGNORECASE) and re.search(r"t\.(?:emp_no|title|to_date)", query, re.IGNORECASE) and not re.search(r"from\s+title\s+(?:as\s+)?t\b", query, re.IGNORECASE):
        query = re.sub(r"from\s+title\b", "FROM title t", query, flags=re.IGNORECASE)

    if not query.strip().endswith(';'):
        query += ";"
    # Secondary duplicate LIMIT cleanup in case edits added a new one before semicolon
    try:
        query = re.sub(r"(LIMIT\s+\d+)(?:\s+LIMIT\s+\d+)+", r"\1", query, flags=re.IGNORECASE)
    except Exception:
        pass

    print("[DEBUG][SQL_FIXER][ARTIFACT_CLEANUP][BEFORE]", query)
    if "dde." in query:
        print("[DEBUG][SQL_FIXER][ARTIFACT_CLEANUP][REPLACE] dde. -> de.")
        query = query.replace("dde.", "de.")
    for original, typo in [("de.","dee."),("de.","dp."),("dm.","ddm."),("dm.","dmm."),("e.","ee."),("e.","emp."),("d.","dd."),("d.","dep."),("s.","ss."),("s.","sal.")]:
        if typo in query:
            print(f"[DEBUG][SQL_FIXER][ARTIFACT_CLEANUP][REPLACE] {typo} -> {original}")
            query = query.replace(typo, original)
    for wrong, right in [("department_id","dept_no"),("employee_id","emp_no"),("manager_id","emp_no")]:
        if wrong in query:
            print(f"[DEBUG][SQL_FIXER][ARTIFACT_CLEANUP][REPLACE] {wrong} -> {right}")
            query = query.replace(wrong, right)
    print("[DEBUG][SQL_FIXER][ARTIFACT_CLEANUP][AFTER]", query)
    
    # Step 7: JOIN/ON artifact cleanup
    if " ON." in query:
        query = re.sub(r'([a-z][a-z0-9_]*)\s+ON\s+([a-z][a-z0-9_]*)\.([a-z][a-z0-9_]*)\s*=\s*ON\.([a-z][a-z0-9_]*)', r'\1 ON \2.\3 = \1.\4', query, flags=re.IGNORECASE)
        if " ON." in query:
            for alias in ["dm","de","e","d","s","t"]:
                if f" {alias} ON" in query and f" ON.{alias}" not in query:
                    query = query.replace(" ON.", f" {alias}.")
    if "JOIN." in query:
        if "JOIN dept_manager dm ON JOIN.dept_no = dm.dept_no" in query:
            query = query.replace("JOIN dept_manager dm ON JOIN.dept_no = dm.dept_no", "JOIN dept_manager dm ON d.dept_no = dm.dept_no")
        if "department d\nJOIN\n    dept_manager dm ON JOIN.dept_no = dm.dept_no" in query:
            query = query.replace("department d\nJOIN\n    dept_manager dm ON JOIN.dept_no = dm.dept_no", "department d\nJOIN\n    dept_manager dm ON d.dept_no = dm.dept_no")
        if "department d\nJOIN\n    dept_manager dm ON JOIN." in query:
            query = query.replace("department d\nJOIN\n    dept_manager dm ON JOIN.", "department d\nJOIN\n    dept_manager dm ON d.")
        query = re.sub(r'([a-z][a-z0-9_]*)\s+ON\s+JOIN\.([a-z][a-z0-9_]*)\s*=\s*([a-z][a-z0-9_]*)\.([a-z][a-z0-9_]*)', r'\1 ON \1.\2 = \3.\4', query, flags=re.IGNORECASE)
        query = re.sub(r'([a-z][a-z0-9_]*)\s+ON\s+([a-z][a-z0-9_]*)\.([a-z][a-z0-9_]*)\s*=\s*JOIN\.([a-z][a-z0-9_]*)', r'\1 ON \2.\3 = \1.\4', query, flags=re.IGNORECASE)
        if "JOIN." in query:
            tables_with_aliases = re.findall(r'([a-z][a-z0-9_]*)\s+([a-z][a-z0-9_]*)(?=\s+ON\s+JOIN\.)', query, flags=re.IGNORECASE)
            if tables_with_aliases:
                last_alias = tables_with_aliases[-1][1]
                query = query.replace("JOIN.", f"{last_alias}.")
            else:
                for alias in ["dm","de","e","d","s","t"]:
                    if f"JOIN {alias}" in query:
                        query = query.replace("JOIN.", f"{alias}.")
            if "JOIN." in query:
                query = query.replace("JOIN.", "d.")
    
    # REMOVED: Duplicate to_date correction logic (already handled in Step 5 above)
    print("[DEBUG][SQL_FIXER][AFTER_TODATE_CORRECTION]", query)
    
    # Add to_date conditions where appropriate based on the schema knowledge
    # For any salary-related query that's missing the to_date condition
    if "salary" in query.lower() and "s." in query and "s.to_date" not in query:
        # If we're joining with employee table
        if "employee e" in query or "e.emp_no" in query:
            # Replace any incorrect e.to_date reference
            if "e.to_date" in query:
                query = query.replace("e.to_date = '9999-01-01'", "s.to_date = '9999-01-01'")
            # Add missing to_date condition based on query structure
            elif "WHERE" in query.upper():
                query = re.sub(r'\bWHERE\b', "WHERE s.to_date = '9999-01-01' AND ", query, count=1, flags=re.IGNORECASE)
            elif "LIMIT" in query.upper():
                query = re.sub(r'\bLIMIT\b', "WHERE s.to_date = '9999-01-01' LIMIT", query, count=1, flags=re.IGNORECASE)
            else:
                query += " WHERE s.to_date = '9999-01-01'"
    
    # Similarly for dept_emp table with 'de' alias
    if "dept_emp" in query.lower() and "de." in query and "de.to_date" not in query:
        if "WHERE" in query.upper():
            query = re.sub(r'\bWHERE\b', "WHERE de.to_date = '9999-01-01' AND ", query, count=1, flags=re.IGNORECASE)
        elif "LIMIT" in query.upper():
            query = re.sub(r'\bLIMIT\b', "WHERE de.to_date = '9999-01-01' LIMIT", query, count=1, flags=re.IGNORECASE)
        else:
            query += " WHERE de.to_date = '9999-01-01'"
    print("[DEBUG][SQL_FIXER][AFTER_DEPTEMP_TODATE]", query)
    
    # Similarly for dept_manager table with 'dm' alias
    if "dept_manager" in query.lower() and "dm." in query and "dm.to_date" not in query:
        if "WHERE" in query.upper():
            query = re.sub(r'\bWHERE\b', "WHERE dm.to_date = '9999-01-01' AND ", query, count=1, flags=re.IGNORECASE)
        elif "LIMIT" in query.upper():
            query = re.sub(r'\bLIMIT\b', "WHERE dm.to_date = '9999-01-01' LIMIT", query, count=1, flags=re.IGNORECASE)
        else:
            query += " WHERE dm.to_date = '9999-01-01'"

    # Deduplicate repeated to_date conditions (e.g. AND s.to_date = '9999-01-01' AND s.to_date = '9999-01-01')
    for alias in ['s', 'de', 'dm', 'e', 'd']:
        dup_pattern = rf"(AND\s+{alias}\.to_date\s*=\s*'9999-01-01'\s*)(?:AND\s+{alias}\.to_date\s*=\s*'9999-01-01'\s*)+"
        query = re.sub(dup_pattern, rf"AND {alias}.to_date = '9999-01-01' ", query, flags=re.IGNORECASE)
    
    # Check for and fix unclosed quotes in date conditions
    if "to_date=" in query:
        # Look for unclosed date quotes with LIMIT inside the string
        if "'9999-01-01 LIMIT" in query:
            query = query.replace("'9999-01-01 LIMIT", "'9999-01-01' LIMIT")
        
        # General check for other variations
        for date_format in ["'9999-01-01", "'9999-01-01 AND", "'9999-01-01 WHERE", "'9999-01-01 GROUP", "'9999-01-01 ORDER"]:
            if date_format in query and date_format + "'" not in query:
                query = query.replace(date_format, date_format + "'")
                
        # If there are multiple to_date conditions, ensure they all have closing quotes
        if "to_date='9999-01-01 AND" in query:
            query = query.replace("to_date='9999-01-01 AND", "to_date='9999-01-01' AND")
    
    # Fix inconsistent table aliases (e.g., "dept_emp" used after "dept_emp de" was defined)
    if "dept_emp de" in query and "dept_emp." in query:
        query = query.replace("dept_emp.", "de.")
    
    if "employee e" in query and "employee." in query:
        query = query.replace("employee.", "e.")
    
    if "department d" in query and "department." in query:
        query = query.replace("department.", "d.")
    
    if "salary s" in query and "salary." in query:
        query = query.replace("salary.", "s.")
    
    if "dept_manager dm" in query and "dept_manager." in query:
        query = query.replace("dept_manager.", "dm.")
    
    if "titles t" in query and "titles." in query:
        query = query.replace("titles.", "t.")

    # Self-join salary alias fix: only if detected, otherwise leave correct joins untouched
    import re
    salary_self_join_pattern = r'JOIN\s+salary\s+s\s+ON\s+s\.emp_no\s*=\s*s\.emp_no'
    deptemp_self_join_pattern = r'JOIN\s+dept_emp\s+(?:AS\s+)?de\s+ON\s+de\.emp_no\s*=\s*de\.emp_no'
    
    print(f'[DEBUG][SQL_FIXER][SELF_JOIN_CHECK] Input query: {query[:200]}...')
    
    # Only apply fix if self-join is present
    if re.search(salary_self_join_pattern, query, re.IGNORECASE):
        print('[DEBUG][SQL_FIXER][SELF_JOIN_SALARY] Found salary self-join pattern')
        emp_alias_match = re.search(r'FROM\s+employee\s+(\w+)|JOIN\s+employee\s+(\w+)', query, re.IGNORECASE)
        emp_alias = (emp_alias_match.group(1) or emp_alias_match.group(2)) if emp_alias_match else None
        if emp_alias:
            query = re.sub(salary_self_join_pattern, f'JOIN salary s ON s.emp_no = {emp_alias}.emp_no', query, flags=re.IGNORECASE)
            print('[DEBUG][SQL_FIXER][SELF_JOIN_SALARY] Fixed self-join: replaced with join to employee alias')
    else:
        print('[DEBUG][SQL_FIXER][SELF_JOIN_SALARY] No salary self-join pattern found, skipping fix')
        
    # Only fix if the join is exactly de.emp_no = de.emp_no and s.emp_no = de.emp_no is not already present
    if re.search(deptemp_self_join_pattern, query, re.IGNORECASE):
        print('[DEBUG][SQL_FIXER][SELF_JOIN_DEPTEMP] Found dept_emp self-join pattern')
        # Check if a correct join already exists
        correct_join_exists = re.search(r'JOIN\s+dept_emp\s+(?:AS\s+)?de\s+ON\s+s\.emp_no\s*=\s*de\.emp_no', query, re.IGNORECASE)
        print(f'[DEBUG][SQL_FIXER][SELF_JOIN_DEPTEMP] Correct join exists: {bool(correct_join_exists)}')
        if not correct_join_exists:
            print('[DEBUG][SQL_FIXER][SELF_JOIN_DEPTEMP] No correct join found, applying fix')
            sal_alias_match = re.search(r'FROM\s+salary\s+(\w+)|JOIN\s+salary\s+(\w+)', query, re.IGNORECASE)
            sal_alias = (sal_alias_match.group(1) or sal_alias_match.group(2)) if sal_alias_match else None
            print(f'[DEBUG][SQL_FIXER][SELF_JOIN_DEPTEMP] Found salary alias: {sal_alias}')
            if sal_alias:
                old_query = query
                query = re.sub(deptemp_self_join_pattern, f'JOIN dept_emp de ON {sal_alias}.emp_no = de.emp_no', query, flags=re.IGNORECASE)
                print('[DEBUG][SQL_FIXER][SELF_JOIN_DEPTEMP] Fixed self-join: replaced with join to salary alias')
                print(f'[DEBUG][SQL_FIXER][SELF_JOIN_DEPTEMP] BEFORE: {old_query}')
                print(f'[DEBUG][SQL_FIXER][SELF_JOIN_DEPTEMP] AFTER: {query}')
        else:
            print('[DEBUG][SQL_FIXER][SELF_JOIN_DEPTEMP] Correct join already present, no fix applied')
    else:
        print('[DEBUG][SQL_FIXER][SELF_JOIN_DEPTEMP] No self-join pattern found, skipping fix')
    
    print(f'[DEBUG][SQL_FIXER][POST_SELF_JOIN] {query[:100]}...')
    
    # If no self-join, do not touch correct joins
            
    # Fix incorrect MIN/MAX alias syntax errors
    if "MIN(" in query or "MAX(" in query:
        # Fix syntax error with column = aggregate instead of AS alias
        if "=" in query and ("MIN" in query or "MAX" in query):
            import re
            # Pattern: column_name = MIN(...)
            query = re.sub(r'([a-zA-Z0-9_\.]+)\s*=\s*MIN\(([^)]+)\)', r'MIN(\2) AS \1_min', query)
            query = re.sub(r'([a-zA-Z0-9_\.]+)\s*=\s*MAX\(([^)]+)\)', r'MAX(\2) AS \1_max', query)
            
            # Pattern: table.column_name = MIN(...)
            query = re.sub(r'([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\s*=\s*MIN\(([^)]+)\)', r'MIN(\3) AS \2_min', query)
            query = re.sub(r'([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\s*=\s*MAX\(([^)]+)\)', r'MAX(\3) AS \2_max', query)
    
    # Fix incorrect MIN/MAX alias syntax (column = aggregate_function instead of alias = aggregate_function)
    if "MIN(" in query or "MAX(" in query:
        # Look for pattern: table.column = MIN(...)
        import re
        query = re.sub(r'([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\s*=\s*(MIN|MAX)\(', r'\3(\1.\2) AS \2_\3', query)
        
        # Also catch: column = MIN(...) without table prefix
        query = re.sub(r'([a-zA-Z0-9_]+)\s*=\s*(MIN|MAX)\(', r'\2(\1) AS \1_\2', query)
    
    # Fix GROUP BY queries that have COUNT in ORDER BY but not in SELECT
    if "GROUP BY" in query.upper() and "ORDER BY" in query.upper() and "COUNT" in query.upper():
        import re
        
        # Check if COUNT is in ORDER BY but not in SELECT
        if "ORDER BY COUNT" in query.upper() and "COUNT" not in query.upper().split("FROM")[0]:
            # Extract the COUNT expression from ORDER BY
            count_match = re.search(r'ORDER BY\s+COUNT\s*\(\s*([^)]+)\s*\)', query, re.IGNORECASE)
            
            if count_match:
                count_expr = count_match.group(1)
                # Add the count to the SELECT clause
                query = re.sub(
                    r'SELECT\s+(.+?)\s+FROM', 
                    rf'SELECT \1, COUNT({count_expr}) AS employee_count FROM', 
                    query, 
                    flags=re.IGNORECASE
                )
                # Update the ORDER BY to use the alias
                query = re.sub(
                    r'ORDER BY\s+COUNT\s*\([^)]+\)', 
                    r'ORDER BY employee_count', 
                    query, 
                    flags=re.IGNORECASE
                )
    
    # Fix missing GROUP BY for aggregate queries (PostgreSQL strict mode)
    if any(agg in query.upper() for agg in ['AVG(', 'SUM(', 'COUNT(', 'MAX(', 'MIN(']):
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', query, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_clause = select_match.group(1)
            # Remove aggregate functions to find non-aggregated columns
            remaining_select = re.sub(r'(COUNT|SUM|AVG|MAX|MIN)\s*\([^)]*\)', '', select_clause, flags=re.IGNORECASE)
            remaining_select = re.sub(r'\s*AS\s+[a-z][a-z0-9_]*', '', remaining_select, flags=re.IGNORECASE)
            # Find alias.column patterns first (e.g. d.dept_name, e.first_name)
            dot_columns = re.findall(r'([a-z][a-z0-9_]*\.[a-z][a-z0-9_]*)', remaining_select, re.IGNORECASE)
            # Also find bare column names (e.g. title, dept_name) for queries without alias prefix
            _GBY_KW = {'where','group','order','limit','having','join','on','and','or','not',
                       'in','as','by','select','from','set','distinct','case','when','then',
                       'else','end','null','true','false','is','like'}
            bare_columns = [t for t in re.findall(r'\b([a-z][a-z0-9_]*)\b', remaining_select, re.IGNORECASE)
                            if t.lower() not in _GBY_KW and not t.isdigit()]
            columns = dot_columns if dot_columns else bare_columns
            if columns and 'GROUP BY' not in query.upper():
                # Deduplicate preserving order
                seen = []
                for c in columns:
                    if c.upper() not in [x.upper() for x in seen]:
                        seen.append(c)
                group_cols = ', '.join(seen)
                # Find insertion point: before ORDER BY, LIMIT, or semicolon
                insertion_point = len(query)
                for clause in [r'\bORDER\s+BY\b', r'\bLIMIT\b']:
                    m = re.search(clause, query, re.IGNORECASE)
                    if m and m.start() < insertion_point:
                        insertion_point = m.start()
                semi = re.search(r'\s*;', query)
                if semi and semi.start() < insertion_point:
                    insertion_point = semi.start()
                query = query[:insertion_point] + f' GROUP BY {group_cols} ' + query[insertion_point:]
                print(f"[DEBUG][SQL_FIXER][GROUP_BY_ADDED] Added GROUP BY {group_cols}")
    
    # Fix incorrect date formatting in conditions
    date_formats = [
        (r"'(\d{4}/\d{2}/\d{2})'", lambda m: f"'{m.group(1).replace('/', '-')}'"),  # YYYY/MM/DD to YYYY-MM-DD
        (r"'(\d{2}/\d{2}/\d{4})'", lambda m: f"'{m.group(1)[6:]}-{m.group(1)[0:2]}-{m.group(1)[3:5]}'"),  # MM/DD/YYYY to YYYY-MM-DD
    ]
    
    for pattern, replacement in date_formats:
        query = re.sub(pattern, lambda m: replacement(m), query)
    
    # Extract all table alias definitions early to avoid UnboundLocalError
    # Pattern matches table aliases only in FROM and JOIN clauses
    alias_pattern = r'(?:FROM|JOIN)\s+([a-z][a-z0-9_]*)\s+(?:AS\s+)?([a-z][a-z0-9_]*)'
    aliases = {}
    for match in re.finditer(alias_pattern, query, re.IGNORECASE):
        table_name, alias = match.groups()
        if alias.lower() not in ['where', 'group', 'order', 'limit', 'on', 'join', 'from']:
            aliases[table_name.lower()] = alias.lower()
    
    # Extract all table alias definitions early to avoid UnboundLocalError
    # Pattern matches table aliases only in FROM and JOIN clauses
    alias_pattern = r'(?:FROM|JOIN)\s+([a-z][a-z0-9_]*)\s+(?:AS\s+)?([a-z][a-z0-9_]*)'
    aliases = {}
    for match in re.finditer(alias_pattern, query, re.IGNORECASE):
        table_name, alias = match.groups()
        if alias.lower() not in ['where', 'group', 'order', 'limit', 'on', 'join', 'from']:
            aliases[table_name.lower()] = alias.lower()
            
    # Fix wrong table aliases in JOIN conditions
    if "JOIN" in query.upper():
        print(f'[DEBUG][SQL_FIXER][ALIAS_FIX_START] {query[:150]}...')
        print(f'[DEBUG][SQL_FIXER][ALIAS_FIX] Found aliases: {aliases}')
        
        # Now check for wrong JOIN conditions
        for table, alias in aliases.items():
            print(f'[DEBUG][SQL_FIXER][ALIAS_FIX] Processing table={table}, alias={alias}')
            # Pattern for incorrect join conditions like: ON table.column = other_alias.column
            incorrect_join = rf'ON\s+{table}\.([a-z][a-z0-9_]*)\s*=\s*([a-z][a-z0-9_]*)\.([a-z][a-z0-9_]*)'
            old_query = query
            query = re.sub(incorrect_join, f'ON {alias}.\\1 = \\2.\\3', query, flags=re.IGNORECASE)
            if old_query != query:
                print(f'[DEBUG][SQL_FIXER][ALIAS_FIX] Applied incorrect_join fix: {table} -> {alias}')
                print(f'[DEBUG][SQL_FIXER][ALIAS_FIX] BEFORE: {old_query[old_query.find("JOIN"):old_query.find("JOIN")+80]}...')
                print(f'[DEBUG][SQL_FIXER][ALIAS_FIX] AFTER:  {query[query.find("JOIN"):query.find("JOIN")+80]}...')
            
            # Pattern for the reverse direction
            incorrect_join_reverse = rf'ON\s+([a-z][a-z0-9_]*)\.([a-z][a-z0-9_]*)\s*=\s*{table}\.([a-z][a-z0-9_]*)'
            old_query2 = query
            query = re.sub(incorrect_join_reverse, f'ON \\1.\\2 = {alias}.\\3', query, flags=re.IGNORECASE)
            if old_query2 != query:
                print(f'[DEBUG][SQL_FIXER][ALIAS_FIX] Applied incorrect_join_reverse fix: {table} -> {alias}')
                print(f'[DEBUG][SQL_FIXER][ALIAS_FIX] BEFORE: {old_query2[old_query2.find("JOIN"):old_query2.find("JOIN")+80]}...')
                print(f'[DEBUG][SQL_FIXER][ALIAS_FIX] AFTER:  {query[query.find("JOIN"):query.find("JOIN")+80]}...')
        
        print(f'[DEBUG][SQL_FIXER][ALIAS_FIX_END] {query[:150]}...')
        
        # Fix common table name/alias confusion in JOIN clauses (like "dept_manager.dept_no" instead of "dm.dept_no")
        if "JOIN" in query and any(full_table in query for full_table in ["department.", "dept_manager.", "dept_emp.", "employee.", "salary.", "titles."]):
            # Replace full table names with their aliases in JOIN conditions
            # Use found aliases from aliases dict, falling back to defaults if not found
            mapping = {
                "department": aliases.get("department", "d"),
                "dept_manager": aliases.get("dept_manager", "dm"), 
                "dept_emp": aliases.get("dept_emp", "de"),
                "employee": aliases.get("employee", "e"),
                "salary": aliases.get("salary", "s"),
                "titles": aliases.get("titles", "t")
            }
            
            for table, alias in mapping.items():
                if f"{table}." in query:
                    # Don't replace in the FROM or JOIN clauses where the alias is defined
                    parts = re.split(r'(FROM\s+|JOIN\s+)', query, flags=re.IGNORECASE)
                    result = []
                    
                    for i, part in enumerate(parts):
                        if i > 0 and (parts[i-1].upper() == "FROM " or parts[i-1].upper() == "JOIN "):
                            # Check if this part contains the definition of an alias for this table
                            # e.g. "employee AS T1" or "employee T1"
                            alias_def = re.match(rf'^{table}\s+(?:AS\s+)?(\w+)', part, re.IGNORECASE)
                            if alias_def:
                                # This is where the alias is defined, don't replace here
                                result.append(part)
                            else:
                                result.append(part.replace(f"{table}.", f"{alias}."))
                        else:
                            result.append(part.replace(f"{table}.", f"{alias}."))
                    query = ''.join(result)
    
    # Optional: allow disabling aggressive artifact heuristics (some regressions observed)
    if os.environ.get("DISABLE_SANITIZER_ARTIFACT_HEURISTICS", "0") in ("1","true","True"):
        # Skip the more invasive JOIN./ON. artifact rewrites
        pass
    else:
        # Final cleanup - ensure no remaining JOIN. syntax errors with smarter mapping
        if "JOIN." in query or " ON." in query:
            try:
                # Build alias preference ordering
                alias_candidates = []
                if re.search(r'\bdept_emp\s+de\b', query, re.IGNORECASE):
                    alias_candidates.append('de')
                if re.search(r'\bdept_manager\s+dm\b', query, re.IGNORECASE):
                    alias_candidates.append('dm')
                if re.search(r'\bemployee\s+e\b', query, re.IGNORECASE):
                    alias_candidates.append('e')
                if re.search(r'\bdepartment\s+d\b', query, re.IGNORECASE):
                    alias_candidates.append('d')
                if re.search(r'\bsalary\s+s\b', query, re.IGNORECASE):
                    alias_candidates.append('s')
                # Fallback default alias if nothing else
                if not alias_candidates:
                    alias_candidates = ['d','e','s','de','dm']

                def _map_join_artifact(m):
                    col = m.group(1)
                    return f"{alias_candidates[0]}.{col}"

                # Replace join.col or ON.col artifacts
                query = re.sub(r'\bjoin\.([a-z][a-z0-9_]*)', lambda m: _map_join_artifact(m), query, flags=re.IGNORECASE)
                query = re.sub(r'\bON\.([a-z][a-z0-9_]*)', lambda m: _map_join_artifact(m), query, flags=re.IGNORECASE)
                # Fix pathological ON alias self-compare: ON join.emp_no = join.emp_no -> try map to distinct aliases
                query = re.sub(r'ON\s+join\.([a-z0-9_]+)\s*=\s*join\.([a-z0-9_]+)', lambda m: f"ON {alias_candidates[0]}.{m.group(1)} = {alias_candidates[1] if len(alias_candidates)>1 else alias_candidates[0]}.{m.group(2)}", query, flags=re.IGNORECASE)
            except Exception:
                pass
        # Smart alias injection: only if table is present WITHOUT ANY ALIAS
        # Fix for the "de AS T2" bug: ensure we don't inject 'de' if 'T2' or any other alias is already there
        for table, default_alias in [("salary", "s"), ("dept_emp", "de"), ("employee", "e"), ("department", "d")]:
            if table in query.lower() and table.lower() not in aliases:
                # If table is mentioned but not in our alias map, it might be unaliased
                # Use a regex that checks if the table name is followed by something that looks like an alias
                # We only want to inject if it's "FROM table" or "JOIN table" followed by a keyword or punctuation
                pattern = rf'(FROM|JOIN)\s+{table}\b(?!\s+(?:AS\s+)?\w+)'
                query = re.sub(pattern, rf'\1 {table} {default_alias}', query, flags=re.IGNORECASE)

        # Fix the specific T1.salary / employee.salary hallucination
        # If 'salary' is in SELECT but NOT in FROM/JOIN, we MUST join it.
        if re.search(r'SELECT\b.*?\.salary\b', query, re.IGNORECASE | re.DOTALL) and 'salary' not in query.lower():
            # Find the alias used for salary, e.g. T1.salary -> T1
            sal_alias_match = re.search(r'([a-zA-Z0-9_]+)\.salary', query, re.IGNORECASE)
            if sal_alias_match:
                bad_alias = sal_alias_match.group(1)
                # If this alias is actually mapped to 'employee', we need to fix it
                if aliases.get('employee') == bad_alias or bad_alias.lower() == 'e':
                    # Find a good place to inject the join
                    if 'FROM employee' in query:
                        query = query.replace('FROM employee', f'FROM employee {bad_alias} JOIN salary s ON {bad_alias}.emp_no = s.emp_no')
                    elif f'FROM employee {bad_alias}' in query:
                        query = query.replace(f'FROM employee {bad_alias}', f'FROM employee {bad_alias} JOIN salary s ON {bad_alias}.emp_no = s.emp_no')
                    query = query.replace(f'{bad_alias}.salary', 's.salary')
                    print(f"[DEBUG][SQL_FIXER][SALARY_HALLUCINATION] Fixed {bad_alias}.salary by joining salary table")

        # Fix the specific d.dept_name / department.dept_name hallucination
        # If 'd.' or 'dept_name' is used but 'department' is NOT in FROM/JOIN, we MUST join it.
        if re.search(r'\bd\.dept_name\b', query, re.IGNORECASE) and 'department' not in query.lower():
            # If we already have a join to dept_emp, we can join department d
            if 'dept_emp' in query.lower():
                de_alias = aliases.get('dept_emp', 'de')
                if f'JOIN department d' not in query:
                    query = query.replace(f'LIMIT', f'JOIN department d ON {de_alias}.dept_no = d.dept_no LIMIT')
                    print(f"[DEBUG][SQL_FIXER][DEPT_HALLUCINATION] Fixed d.dept_name by joining department table")
            else:
                # If no dept_emp, we need a cross join or similar? 
                # Better to just add a standard join sequence if it's a global query
                pass
        # Only repair salary.emp_no = dept_emp.emp_no if NOT inside an ON clause (avoid rewriting valid joins)
        # This is a very conservative fix: only replace if the pattern is NOT inside an ON ... = ... clause
        # Do NOT replace all salary.emp_no or dept_emp.emp_no globally!
        # Instead, only fix the specific case of a direct, unaliased join condition outside ON clauses
        # (This disables the destructive rewrite for all other cases)
        # Simple extremes pattern: SELECT MIN(salary), MAX(salary) FROM salary -> ensure alias usage
        if re.search(r'SELECT\s+MIN\(salary\)\s*,\s*MAX\(salary\)\s+FROM\s+salary(?!\s+(?:AS\s+)?s\b)', query, re.IGNORECASE):
            query = re.sub(r'FROM\s+salary(?!\s+(?:AS\s+)?s\b)', 'FROM salary s', query, flags=re.IGNORECASE)
            query = re.sub(r'MIN\(salary\)', 'MIN(s.salary)', query, flags=re.IGNORECASE)
            query = re.sub(r'MAX\(salary\)', 'MAX(s.salary)', query, flags=re.IGNORECASE)
        # Fix leftover join.emp_no occurrences after mapping (only if not part of a valid ON clause)
        # This is now a no-op unless a clear artifact remains, so we keep it for legacy cases only
        if re.search(r'\bjoin\.emp_no\b', query, re.IGNORECASE):
            query = re.sub(r'\bjoin\.emp_no\b', 's.emp_no', query, flags=re.IGNORECASE)

    # Department naming normalization & column correction
    # Normalize capitalized Department references
    query = re.sub(r'FROM\s+Department\b', 'FROM department', query, flags=re.IGNORECASE)
    query = re.sub(r'JOIN\s+Department\b', 'JOIN department', query, flags=re.IGNORECASE)
    # Standardize alias to d (if uppercase D used)
    query = re.sub(r'from\s+department\s+D\b', 'FROM department d', query, flags=re.IGNORECASE)
    query = re.sub(r'join\s+department\s+D\b', 'JOIN department d', query, flags=re.IGNORECASE)
    # Replace d.name (nonexistent) with d.dept_name
    query = re.sub(r'\b[dD]\.name\b', 'd.dept_name', query)
    # Fix hallucinated column department.department_name
    query = re.sub(r'department\.department_name', 'department.dept_name', query, flags=re.IGNORECASE)
    query = re.sub(r'd\.department_name', 'd.dept_name', query, flags=re.IGNORECASE)
    # Repair broken pattern 'JOIN dept ON d.dept_no = ON.dept_no'
    if re.search(r'JOIN\s+dept\s+ON\s+d\.dept_no\s*=\s*ON\.dept_no', query, flags=re.IGNORECASE):
        query = re.sub(r'JOIN\s+dept\s+ON\s+d\.dept_no\s*=\s*ON\.dept_no', '', query, flags=re.IGNORECASE)
    # Replace stray ' ON.dept_no' with proper alias if department alias d exists
    if ' ON.dept_no' in query and re.search(r'FROM\s+department\s+(?:AS\s+)?d\b', query, flags=re.IGNORECASE):
        query = query.replace(' ON.dept_no', ' d.dept_no')

    # General mapping: convert stray ON.<col> (e.g. ON.emp_no, ON.dept_no) to a likely alias.<col>
    # This handles cases where the LLM accidentally emitted 'ON.col' instead of 'alias.col'
    try:
        def _map_on_col(m):
            col = m.group(1).lower()
            # Prefer department alias 'd' for dept_no
            if col == 'dept_no' and re.search(r'FROM\s+department\s+(?:AS\s+)?d\b', query, flags=re.IGNORECASE):
                return 'd.dept_no'
            # Prefer dept_emp alias 'de' for dept_emp-related columns
            if col in ('dept_no', 'emp_no') and re.search(r'\bdept_emp\s+(?:as\s+)?de\b', query, flags=re.IGNORECASE):
                return f'de.{col}'
            # dept_manager alias
            if col in ('dept_no', 'emp_no') and re.search(r'\bdept_manager\s+(?:as\s+)?dm\b', query, flags=re.IGNORECASE):
                return f'dm.{col}'
            # salary/employee preferences for emp_no
            if col == 'emp_no' and re.search(r'\bsalary\s+(?:as\s+)?s\b', query, flags=re.IGNORECASE):
                return 's.emp_no'
            if col == 'emp_no' and re.search(r'\bemployee\s+(?:as\s+)?e\b', query, flags=re.IGNORECASE):
                return 'e.emp_no'
            # If an explicit alias for the column exists anywhere, prefer that alias
            for alias in ['d', 'de', 'dm', 'e', 's', 't']:
                if re.search(rf'\b{alias}\.', query, flags=re.IGNORECASE) or re.search(rf'\b{alias}\b', query, flags=re.IGNORECASE):
                    return f"{alias}.{col}"
            # Fallback to department alias if present, otherwise 'd'
            if re.search(r'FROM\s+department', query, flags=re.IGNORECASE):
                return f'd.{col}'
            return f'd.{col}'

        query = re.sub(r'\bON\.([a-z][a-z0-9_]*)\b', lambda m: _map_on_col(m), query, flags=re.IGNORECASE)
    except Exception:
        pass

    # Fix common LLM alias typos like 'dept_e' or 'dept_emp' misuse and stray 'ON.dept_no'
    # e.g. "JOIN department d ON dept_e.dept_no = ON.dept_no" -> "JOIN department d ON de.dept_no = d.dept_no"
    try:
        if re.search(r'join\s+department\s+d\s+on\s+[a-zA-Z0-9_]+\.dept_no\s*=\s*on\.dept_no', query, flags=re.IGNORECASE):
            # Normalize left alias to 'de' (dept_emp) and right side to 'd.dept_no'
            query = re.sub(r'join\s+department\s+d\s+on\s+[a-zA-Z0-9_]+\.dept_no\s*=\s*on\.dept_no',
                           'JOIN department d ON de.dept_no = d.dept_no',
                           query, flags=re.IGNORECASE)

        # Replace 'dept_e.' (a common truncated alias) with 'de.'
        if re.search(r'\bdept_e\.', query, flags=re.IGNORECASE):
            query = re.sub(r'\bdept_e\.', 'de.', query, flags=re.IGNORECASE)

        # Replace stray 'ON.dept_no' with 'd.dept_no' if department alias d is present
        if re.search(r'\bON\.dept_no\b', query, flags=re.IGNORECASE) and re.search(r'from\s+department\s+(?:as\s+)?d\b', query, flags=re.IGNORECASE):
            query = re.sub(r'\bON\.dept_no\b', 'd.dept_no', query, flags=re.IGNORECASE)

        # Normalize 'dept_emp' full name followed by dot to 'de.' when alias de exists
        if re.search(r'\bdept_emp\.', query, flags=re.IGNORECASE) and re.search(r'\bdept_emp\s+de\b', query, flags=re.IGNORECASE):
            query = re.sub(r'\bdept_emp\.', 'de.', query, flags=re.IGNORECASE)
    except Exception:
        pass

    # Final debug print to show the output after all fixer logic
    try:
        print("[DEBUG][SQL_FIXER][AFTER]", query)
    except Exception:
        pass

    # Fix common mistake: stray semicolon before WHERE (e.g. "; WHERE ...")
    # This often breaks SQL (syntax error at or near "WHERE"). Remove the stray semicolon.
    if re.search(r';\s*WHERE', query, flags=re.IGNORECASE):
        query = re.sub(r';\s*WHERE', ' WHERE', query, flags=re.IGNORECASE)

    # Fix mistaken alias references where LLM used 'd' but salary table is aliased as 's'.
    # Example: "salary s JOIN dept_emp de ON d.emp_no = de.emp_no" -> use s.emp_no
    try:
        if re.search(r'\bsalary\s+s\b', query, flags=re.IGNORECASE) and re.search(r'\bd\.emp_no\b', query, flags=re.IGNORECASE):
            query = re.sub(r'\bd\.emp_no\b', 's.emp_no', query, flags=re.IGNORECASE)
    except Exception:
        pass
    # Safe auto-insert: if the SQL references alias `e.` but there's no FROM/JOIN for employee
    # inject a conservative join using dept_emp alias `de` when present. This addresses errors
    # like "missing FROM-clause entry for table \"e\"" for queries such as employee counts per dept.
    try:
        has_e_alias_usage = re.search(r'\be\.', query, flags=re.IGNORECASE)
        has_employee_join = re.search(r'\bjoin\s+employee\b|\bfrom\s+employee\b|\bemployee\s+e\b', query, flags=re.IGNORECASE)
        has_de_alias = re.search(r'\b(dept_emp\s+de|\bde\.)', query, flags=re.IGNORECASE)
        if has_e_alias_usage and not has_employee_join and has_de_alias:
            # Insert join before WHERE/GROUP/ORDER/LIMIT/HAVING if present, otherwise append before semicolon/end
            insert_point = None
            m = re.search(r'\b(WHERE|GROUP\s+BY|ORDER\s+BY|LIMIT|HAVING)\b', query, flags=re.IGNORECASE)
            join_clause = ' JOIN employee e ON de.emp_no = e.emp_no'
            if m:
                idx = m.start()
                query = query[:idx] + join_clause + ' ' + query[idx:]
            else:
                # append safely before final semicolon if present
                if query.strip().endswith(';'):
                    query = query.rstrip()[:-1] + join_clause + ';'
                else:
                    query = query + join_clause
            # Normalize spacing
            query = re.sub(r'\s{2,}', ' ', query).strip()
    except Exception:
        pass
    
    # Missing table join detection and auto-fix
    # Fix missing department table join when d.dept_name is referenced
    try:
        # Fix for e.salary error: LLM trying to get salary from employee table
        has_employee_salary = re.search(r'\be\.salary\b', query, re.IGNORECASE)
        if has_employee_salary:
            print("[DEBUG][SQL_FIXER][MISSING_JOIN] e.salary referenced - need salary table join")
            # Add salary table join
            if not re.search(r'salary\s+s\b', query, re.IGNORECASE):
                # Add salary join after employee table
                employee_join_pattern = r'(FROM\s+employee\s+e)'
                if re.search(employee_join_pattern, query, re.IGNORECASE):
                    query = re.sub(employee_join_pattern, r'\1\nJOIN salary s ON e.emp_no = s.emp_no', query, flags=re.IGNORECASE)
                    print("[DEBUG][SQL_FIXER][MISSING_JOIN] Added salary join")
            
            # Fix the column references: e.salary -> s.salary
            query = re.sub(r'\be\.salary\b', 's.salary', query, flags=re.IGNORECASE)
            print("[DEBUG][SQL_FIXER][MISSING_JOIN] Fixed column references from e.salary to s.salary")
            
            # Add salary table to WHERE clause for current records
            if 's.to_date' not in query.lower():
                where_match = re.search(r'(\bWHERE\b)', query, re.IGNORECASE)
                if where_match:
                    query = re.sub(r'(\bWHERE\b)', r'\1 s.to_date = \'9999-01-01\' AND', query, flags=re.IGNORECASE)
                else:
                    # Add WHERE clause before GROUP BY
                    group_match = re.search(r'(\bGROUP BY\b)', query, re.IGNORECASE)
                    if group_match:
                        query = re.sub(r'(\bGROUP BY\b)', r'WHERE s.to_date = \'9999-01-01\'\n\1', query, flags=re.IGNORECASE)
                print("[DEBUG][SQL_FIXER][MISSING_JOIN] Added salary to_date filter")
            print(f"[DEBUG][SQL_FIXER][MISSING_JOIN_AFTER] {query}")
        
        # Fix for missing e.gender in SELECT: LLM puts e.gender in GROUP BY but forgets to SELECT it
        # This produces 2-column results instead of 3-column (dept, gender, count)
        # Use SELECT-anchored search to avoid matching schema text or LLM prompt preamble
        _ql = query.lower()
        _select_pos = _ql.rfind('select')          # rfind: use LAST 'select' (closest to actual SQL)
        _group_by_pos = _ql.rfind('group by')      # rfind: use LAST 'group by'
        _gender_in_groupby = False
        _gender_in_select = True  # default safe: don't inject
        if _select_pos > -1 and _group_by_pos > -1 and _group_by_pos > _select_pos:
            _group_by_snippet = query[_group_by_pos:]
            _gender_in_groupby = 'e.gender' in _group_by_snippet.lower()
            # Find FROM between SELECT and GROUP BY using ' from ' (spaces avoid matching 'from_date')
            _from_pos = _ql.find(' from ', _select_pos)
            if _from_pos > -1:
                _select_snippet = query[_select_pos:_from_pos]
                _gender_in_select = 'e.gender' in _select_snippet.lower()
        if _gender_in_groupby and not _gender_in_select:
            print("[DEBUG][SQL_FIXER][GENDER_SELECT] e.gender in GROUP BY but missing from SELECT — injecting")
            # Find the first comma AFTER the SELECT keyword (safe: skips any schema text before SELECT)
            _actual_select_pos = _ql.rfind('select')
            first_comma = query.find(',', _actual_select_pos)
            if first_comma > -1:
                query = query[:first_comma + 1] + '\ne.gender,' + query[first_comma + 1:]
            print(f"[DEBUG][SQL_FIXER][GENDER_SELECT_AFTER] {query[_actual_select_pos:_actual_select_pos+250]}")
        
        # Fix for e.dept_name error: LLM incorrectly puts dept_name on employee table
        # dept_name lives on the department table, requires dept_emp + department JOINs
        has_e_dept_name = re.search(r'\be\.dept_name\b', query, re.IGNORECASE)
        if has_e_dept_name:
            print("[DEBUG][SQL_FIXER][MISSING_JOIN] e.dept_name referenced — dept_name is not on employee table, adding dept_emp + department JOINs")
            emp_alias = re.search(r'FROM\s+employee\s+(\w+)', query, re.IGNORECASE)
            e_alias = emp_alias.group(1) if emp_alias else 'e'
            # Add dept_emp join if missing
            if not re.search(r'dept_emp\s+de\b', query, re.IGNORECASE):
                emp_table_pattern = rf'(FROM\s+employee\s+{e_alias})'
                if re.search(emp_table_pattern, query, re.IGNORECASE):
                    query = re.sub(emp_table_pattern, rf'\1\nJOIN dept_emp de ON {e_alias}.emp_no = de.emp_no', query, flags=re.IGNORECASE)
                    print("[DEBUG][SQL_FIXER][MISSING_JOIN] Added dept_emp join")
            # Add department join if missing
            if not re.search(r'JOIN\s+department\b', query, re.IGNORECASE):
                dept_emp_pattern = r'(JOIN\s+dept_emp\s+de\s+ON\s+[^\n]+)'
                if re.search(dept_emp_pattern, query, re.IGNORECASE):
                    query = re.sub(dept_emp_pattern, r'\1\nJOIN department d ON de.dept_no = d.dept_no', query, flags=re.IGNORECASE)
                    print("[DEBUG][SQL_FIXER][MISSING_JOIN] Added department join")
            # Fix to_date filter on dept_emp
            if 'de.to_date' not in query.lower():
                where_match = re.search(r'(\bWHERE\b)', query, re.IGNORECASE)
                if where_match:
                    query = re.sub(r'\bWHERE\b', lambda m: "WHERE de.to_date = '9999-01-01' AND", query, flags=re.IGNORECASE, count=1)
                    print("[DEBUG][SQL_FIXER][MISSING_JOIN] Added de.to_date filter")
            # Replace e.dept_name → d.dept_name everywhere
            query = re.sub(r'\be\.dept_name\b', 'd.dept_name', query, flags=re.IGNORECASE)
            print("[DEBUG][SQL_FIXER][MISSING_JOIN] Fixed e.dept_name → d.dept_name")
            print(f"[DEBUG][SQL_FIXER][MISSING_JOIN_AFTER] {query}")
        
        # Fix: title table has no dept_no column — LLM hallucinates direct title→department join
        # Correct path: title → dept_emp (via emp_no) → department (via dept_no)
        if re.search(r'\bt\.dept_no\b', query, re.IGNORECASE):
            print("[DEBUG][SQL_FIXER][TITLE_DEPT_NO] t.dept_no doesn't exist — fixing title→dept_emp→department join")
            # Replace: JOIN department X ON t.dept_no = X.dept_no
            # With: JOIN dept_emp de ON t.emp_no = de.emp_no \n JOIN department X ON de.dept_no = X.dept_no
            fixed = re.sub(
                r'JOIN\s+department\s+(?:AS\s+)?(\w+)\s+ON\s+t\.dept_no\s*=\s*\1\.dept_no',
                r'JOIN dept_emp de ON t.emp_no = de.emp_no\nJOIN department \1 ON de.dept_no = \1.dept_no',
                query, flags=re.IGNORECASE
            )
            if fixed != query:
                query = fixed
                # Add de.to_date filter if missing
                if 'de.to_date' not in query.lower():
                    query = re.sub(r'\bWHERE\b', "WHERE de.to_date = '9999-01-01' AND", query, flags=re.IGNORECASE, count=1)
                print(f"[DEBUG][SQL_FIXER][TITLE_DEPT_NO_AFTER] {query[:300]}")


        has_department_join = re.search(r'\bjoin\s+department\b|\bfrom\s+department\b', query, re.IGNORECASE | re.DOTALL)
        # Updated pattern to handle multiline formatting and whitespace - simplified
        has_dept_emp_join = re.search(r'dept_emp\s+de\b', query, re.IGNORECASE | re.DOTALL)
        
        # Fix for incorrect alias usage: T2.dept_no/T2.dept_name when T2 is salary table
        has_salary_alias_dept_cols = re.search(r'\bT2\.(dept_no|dept_name)\b', query, re.IGNORECASE)
        has_salary_table_as_t2 = re.search(r'\bsalary\s+AS\s+T2\b', query, re.IGNORECASE)
        
        if has_salary_alias_dept_cols and has_salary_table_as_t2:
            print("[DEBUG][SQL_FIXER][MISSING_JOIN] T2.dept_no/dept_name referenced but T2 is salary table - need dept_emp and department joins")
            # This is a fundamental schema error - the LLM is trying to get dept info from salary table
            # We need to add dept_emp and department joins and fix the column references
            
            # Check if we already have dept_emp join
            if not re.search(r'dept_emp', query, re.IGNORECASE):
                # Add dept_emp join after employee table
                employee_join_pattern = r'(FROM\s+employee\s+AS\s+T1)'
                if re.search(employee_join_pattern, query, re.IGNORECASE):
                    query = re.sub(employee_join_pattern, r'\1\nJOIN dept_emp de ON T1.emp_no = de.emp_no', query, flags=re.IGNORECASE)
                    print("[DEBUG][SQL_FIXER][MISSING_JOIN] Added dept_emp join")
            
            # Check if we need department join  
            if not re.search(r'department', query, re.IGNORECASE):
                # Add department join after dept_emp
                query = re.sub(r'(JOIN dept_emp de ON T1\.emp_no = de\.emp_no)', r'\1\nJOIN department d ON de.dept_no = d.dept_no', query, flags=re.IGNORECASE)
                print("[DEBUG][SQL_FIXER][MISSING_JOIN] Added department join")
            
            # Fix the column references: T2.dept_no -> d.dept_no, T2.dept_name -> d.dept_name
            query = re.sub(r'\bT2\.dept_no\b', 'd.dept_no', query, flags=re.IGNORECASE)
            query = re.sub(r'\bT2\.dept_name\b', 'd.dept_name', query, flags=re.IGNORECASE)
            print("[DEBUG][SQL_FIXER][MISSING_JOIN] Fixed column references from T2 to d")
            print(f"[DEBUG][SQL_FIXER][MISSING_JOIN_AFTER] {query}")
        
        elif has_d_dept_name and not has_department_join and has_dept_emp_join:
            print("[DEBUG][SQL_FIXER][MISSING_JOIN] d.dept_name referenced but no department table joined")
            # Insert department join after the first JOIN clause (simplified approach)
            first_join_match = re.search(r'(JOIN\s+\w+\s+\w+\s+ON\s+[^\n]+)', query, re.IGNORECASE)
            if first_join_match:
                join_clause = first_join_match.group(1)
                new_join = join_clause + '\nJOIN department d ON de.dept_no = d.dept_no'
                query = query.replace(join_clause, new_join)
                print("[DEBUG][SQL_FIXER][MISSING_JOIN] Added department table join after first JOIN")
                print(f"[DEBUG][SQL_FIXER][MISSING_JOIN_AFTER] {query}")
            else:
                # Fallback: insert department join before WHERE clause
                where_match = re.search(r'(\bWHERE\b)', query, re.IGNORECASE)
                if where_match:
                    insert_pos = where_match.start()
                    new_join = '\nJOIN department d ON de.dept_no = d.dept_no\n'
                    query = query[:insert_pos] + new_join + query[insert_pos:]
                    print("[DEBUG][SQL_FIXER][MISSING_JOIN] Added department table join (fallback method)")
                    print(f"[DEBUG][SQL_FIXER][MISSING_JOIN_AFTER] {query}")
    except Exception:
        pass
    
    # Final normalization: remove duplicate LIMITs again, trim extra semicolons & spaces
    try:
        query = re.sub(r"(LIMIT\s+\d+)(?:\s+LIMIT\s+\d+)+", r"\1", query, flags=re.IGNORECASE)
        query = re.sub(r";{2,}$", ";", query.strip())
        query = re.sub(r"\s+;", ";", query)
        query = re.sub(r"[ \t]{2,}", " ", query)
    except Exception:
        pass
    # Final: if we identified a simple extremes pattern but sanitizer introduced alias mismatch, normalize
    if simple_extremes:
        # Ensure aliasing consistency
        if re.search(r'FROM\s+salary\s+s', query, re.IGNORECASE):
            # Convert MIN(salary) to MIN(s.salary) if missing
            query = re.sub(r'MIN\(salary\)', 'MIN(s.salary)', query, flags=re.IGNORECASE)
            query = re.sub(r'MAX\(salary\)', 'MAX(s.salary)', query, flags=re.IGNORECASE)
        # Remove trailing LIMIT if present (not needed for two-row aggregate)
        if re.search(r'LIMIT\s+100;?$', query, re.IGNORECASE):
            query = re.sub(r'LIMIT\s+100;?$', ';', query, flags=re.IGNORECASE)
    # Fix malformed subquery aliases like "FROM salary s s2" -> "FROM salary s2"
    # Common LLM error: generates "FROM table alias1 alias2" instead of proper aliases
    try:
        # Pattern: FROM table_name alias1 alias2 (followed by space/newline, not JOIN)
        # Must have exactly 3 words after FROM and be followed by whitespace or end
        malformed_alias_pattern = r'\bFROM\s+(\w+)\s+(\w+)\s+(\w+)(?=\s|$)'
        matches = re.findall(malformed_alias_pattern, query, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            # Only fix if we have exactly the malformed pattern "FROM table alias1 alias2"
            # and not something like "FROM table alias JOIN"
            for match in matches:
                table_name, alias1, alias2 = match
                # Check if this is actually malformed (alias2 shouldn't be SQL keywords)
                if alias2.upper() not in ['JOIN', 'WHERE', 'GROUP', 'ORDER', 'LIMIT', 'HAVING', 'ON', 'AND', 'OR', 'BY']:
                    # Additional safeguard: Don't fix if the old_pattern appears in a subquery context
                    # Look for patterns like "SELECT...FROM table WHERE" which indicate a subquery
                    old_pattern = f"FROM {table_name} {alias1} {alias2}"
                    
                    # Check if this pattern appears near SELECT and WHERE keywords (subquery indicator)
                    if old_pattern in query:
                        # Find the position of this pattern
                        pos = query.find(old_pattern)
                        # Look before and after for SELECT/WHERE to detect subqueries
                        before_context = query[max(0, pos-100):pos].upper()
                        after_context = query[pos:pos+100].upper()
                        
                        # Skip if this looks like a subquery (has SELECT before and WHERE after)
                        if 'SELECT' in before_context and 'WHERE' in after_context:
                            print(f"[DEBUG][SQL_FIXER][MALFORMED_ALIAS] Skipping subquery pattern: {old_pattern}")
                            continue
                    
                    # Safe to fix
                    new_pattern = f"FROM {table_name} {alias2}"
                    query = query.replace(old_pattern, new_pattern)
                    print(f"[DEBUG][SQL_FIXER][MALFORMED_ALIAS] Fixed malformed table alias: {old_pattern} -> {new_pattern}")
    except Exception as e:
        print(f"[DEBUG][SQL_FIXER][MALFORMED_ALIAS] Error: {e}")
        pass

    # Fix alias reference in SELECT clause (PostgreSQL doesn't allow referencing column aliases in same SELECT)
    # Common pattern: (avg_male_salary - avg_female_salary) AS salary_gap
    # Fix: Replace with the actual expressions
    # TEMPORARILY DISABLED - causing SQL corruption
    try:
        # Skip alias reference fix for now to avoid SQL corruption
        pass
        # alias_ref_pattern = r'\(\s*avg_male_salary\s*-\s*avg_female_salary\s*\)'
        # if re.search(alias_ref_pattern, query, flags=re.IGNORECASE):
        #     # Find the SELECT clause and extract alias definitions
        #     select_part = re.search(r'SELECT\s+(.*?)\s+FROM', query, flags=re.IGNORECASE | re.DOTALL)
        #     if select_part:
        #         select_content = select_part.group(1)
        #         # Look for avg_male_salary and avg_female_salary definitions with proper word boundaries
        #         male_salary_match = re.search(r'(AVG\s*\(\s*CASE\s+WHEN\s+e\.gender\s*=\s*[\'"]M[\'"].*?END\s*\))\s+AS\s+avg_male_salary', select_content, flags=re.IGNORECASE | re.DOTALL)
        #         female_salary_match = re.search(r'(AVG\s*\(\s*CASE\s+WHEN\s+e\.gender\s*=\s*[\'"]F[\'"].*?END\s*\))\s+AS\s+avg_female_salary', select_content, flags=re.IGNORECASE | re.DOTALL)
        #         
        #         if male_salary_match and female_salary_match:
        #             male_expr = male_salary_match.group(1).strip()
        #             female_expr = female_salary_match.group(1).strip()
        #             # Replace the alias references with actual expressions, being more careful about boundaries
        #             replacement = f'({male_expr} - {female_expr})'
        #             query = re.sub(r'\(\s*avg_male_salary\s*-\s*avg_female_salary\s*\)', replacement, query, flags=re.IGNORECASE)
        #             print(f"[DEBUG][SQL_FIXER][ALIAS_REF_FIX] Replaced alias references with expressions")
    except Exception as e:
        print(f"[DEBUG][SQL_FIXER][ALIAS_REF_FIX] Error: {e}")
        pass

    # Fix missing employee alias in JOIN
    try:
        if re.search(r'FROM\s+employee\s+JOIN', query, flags=re.IGNORECASE):
            query = re.sub(r'FROM\s+employee\s+JOIN', 'FROM employee e JOIN', query, flags=re.IGNORECASE)
            print(f"[DEBUG][SQL_FIXER][MISSING_ALIAS] Added missing employee alias")
    except Exception as e:
        print(f"[DEBUG][SQL_FIXER][MISSING_ALIAS] Error: {e}")
        pass

    # Fix invalid ds.to_date references
    try:
        if 'ds.to_date' in query:
            query = query.replace('ds.to_date', 's.to_date')
            print(f"[DEBUG][SQL_FIXER][INVALID_COLUMN] Fixed ds.to_date to s.to_date")
    except Exception as e:
        print(f"[DEBUG][SQL_FIXER][INVALID_COLUMN] Error: {e}")
        pass

    # Fix unnecessary GROUP BY that matches COUNT DISTINCT field
    # Pattern: SELECT COUNT(DISTINCT alias.column) ... GROUP BY alias.column
    # This is always wrong - returns one row per value instead of total count
    try:
        # Find COUNT(DISTINCT alias.column) in SELECT
        count_distinct_pattern = r'COUNT\s*\(\s*DISTINCT\s+(\w+)\.(\w+)\s*\)'
        count_match = re.search(count_distinct_pattern, query, re.IGNORECASE)
        
        if count_match:
            alias = count_match.group(1)
            column = count_match.group(2)
            
            # Check if there's a GROUP BY with the same alias.column
            group_by_pattern = rf'GROUP\s+BY\s+{re.escape(alias)}\.{re.escape(column)}\b'
            if re.search(group_by_pattern, query, re.IGNORECASE):
                # Remove the GROUP BY clause entirely
                query = re.sub(group_by_pattern, '', query, flags=re.IGNORECASE)
                # Clean up any trailing commas or extra whitespace
                query = re.sub(r'\s+', ' ', query).strip()
                print(f"[DEBUG][SQL_FIXER][GROUP_BY_FIX] Removed unnecessary GROUP BY {alias}.{column}")
    except Exception as e:
        print(f"[DEBUG][SQL_FIXER][GROUP_BY_FIX] Error: {e}")
        pass

    query = query.strip()
    # Debug: print SQL after fixing
    try:
        print("[DEBUG][SQL_FIXER][AFTER]", query)
    except Exception:
        pass
    return query

# ---------------- Hybrid micro-generation verification helper -----------------
def verify_micro_sql(candidate_sql: Optional[str], question: str, tag: str) -> Optional[str]:
    """Verify & (optionally) adjust a micro-generated candidate via MCP/LLM.
    Returns validated SQL or None to signal fallback to full LLM generation.
    """
    if not candidate_sql or not isinstance(candidate_sql, str):
        return None
    if os.environ.get("SKIP_MICRO_VERIFICATION", "0") in ("1","true","True"):
        return candidate_sql
    try:
        base_val = validate_sql_query(candidate_sql)
        eh_val = enhanced_validate_sql(candidate_sql)
        if base_val.get('valid') and eh_val.get('valid'):
            # Already valid, skip fixer
            return candidate_sql
        prompt = (
            f"{EMPLOYEE_DB_SCHEMA}\n\n"
            "You are an SQL verifier. A candidate SQL statement was programmatically assembled for the question below. "
            "If it's already valid, you may return it unchanged. If SMALL fixes (aliases, correct column name existing in schema, missing LIMIT) are needed, correct them. "
            "DO NOT invent new tables or columns. Always end with a semicolon; add 'LIMIT 100' if a plain SELECT without limiting and not an aggregate returning <= 1 row.\n"
            f"Tag: {tag}\n"
            f"Question: {question}\n"
            "Candidate SQL:\n" + candidate_sql + "\n\nReturn ONLY the final SQL (no explanation)."
        )
        verified_text = None
        # Try MCP first
        try:
            from .mcp_client import mcp_complete
            verified_text = mcp_complete(prompt, timeout=MICRO_VERIFY_MCP_TIMEOUT)
        except Exception:
            try:
                llm_local = Settings.llm
                resp = llm_local.complete(prompt=prompt, timeout=MICRO_VERIFY_LLM_TIMEOUT)
                verified_text = getattr(resp, 'text', str(resp))
            except Exception as e2:
                print(f"[MICRO-VERIFY][WARN] Verification LLM failed: {e2}; using original candidate if valid.")
                if base_val.get('valid'):
                    return candidate_sql
                return None
        if not verified_text:
            return candidate_sql if base_val.get('valid') else None
        # Extract SQL
        if '```' in verified_text:
            parts = re.split(r"```(?:sql)?", verified_text, flags=re.IGNORECASE)
            verified_text = '\n'.join(p for p in parts if 'select' in p.lower() or 'with' in p.lower())
        if verified_text and verified_text.lower().startswith('sql'):
            verified_text = verified_text[3:].strip()
        # Only fix if not valid
        v1 = validate_sql_query(verified_text)
        v2 = enhanced_validate_sql(verified_text)
        if v1.get('valid') and v2.get('valid'):
            print(f"[MICRO-VERIFY] Accepted verified SQL for tag={tag}")
            return verified_text
        else:
            print(f"[MICRO-VERIFY][WARN] Verified SQL invalid (base={v1}, issues={v2.get('issues')}); applying fixer.")
            # Guard: only pass actual SQL to the fixer, not the full LLM prompt
            _vt_stripped = verified_text.strip().lstrip('\n') if verified_text else ''
            _vt_lower = _vt_stripped.lower()
            if not (_vt_lower.startswith('select') or _vt_lower.startswith('with ') or _vt_lower.startswith('insert') or _vt_lower.startswith('update')):
                # LLM returned the full prompt instead of SQL — extract the last SELECT...; block
                _sql_match = re.search(r'(SELECT\s.+?;)\s*$', _vt_stripped, re.IGNORECASE | re.DOTALL)
                if _sql_match:
                    _vt_stripped = _sql_match.group(1).strip()
                    print(f"[MICRO-VERIFY][WARN] Extracted SQL from prompt echo: {_vt_stripped[:120]}")
                elif base_val.get('valid'):
                    print("[MICRO-VERIFY][WARN] Could not extract SQL; falling back to original candidate")
                    return candidate_sql
                else:
                    return None
            fixed = fix_common_sql_errors(_vt_stripped)
            v1f = validate_sql_query(fixed)
            v2f = enhanced_validate_sql(fixed)
            if v1f.get('valid') and v2f.get('valid'):
                print(f"[MICRO-VERIFY] Fixed SQL accepted for tag={tag}")
                return fixed
            # fallback to original if it was valid
            if base_val.get('valid') and eh_val.get('valid'):
                return candidate_sql
            return None
    except Exception as e:
        print(f"[MICRO-VERIFY][ERROR] Unexpected error: {e}; falling back to full generation")
        return None


def schema_aware_column_mapper(query: str) -> str:
    """Map alias.column references to the correct alias based on the inspected schema.

    If an alias references a column not present in its table, search other tables
    present in the FROM/JOIN clauses for that column and rewrite the alias to
    the correct one. This fixes cases like `de.dept_name` -> `d.dept_name`.
    """
    try:
        # Build alias -> table mapping from FROM and JOIN clauses
        aliases: Dict[str, str] = {}
        for m in re.finditer(r"\bFROM\s+([a-zA-Z_][\w]*)\s+(?:AS\s+)?([a-zA-Z_][\w]*)", query, flags=re.IGNORECASE):
            table, alias = m.group(1).lower(), m.group(2).lower()
            aliases[alias] = table
        for m in re.finditer(r"\bJOIN\s+([a-zA-Z_][\w]*)\s+(?:AS\s+)?([a-zA-Z_][\w]*)", query, flags=re.IGNORECASE):
            table, alias = m.group(1).lower(), m.group(2).lower()
            aliases[alias] = table

        # Cache columns for each alias (table)
        alias_cols: Dict[str, List[str]] = {}
        for alias, table in aliases.items():
            try:
                cols = [c['name'].lower() for c in inspector.get_columns(table_name=table)]
            except Exception:
                cols = []
            alias_cols[alias] = cols

        # Replacement: for each occurrence alias.col, if col not in alias table, try to find other alias that has it
        def _replace_alias_col(m: re.Match) -> str:
            a = m.group(1)
            col = m.group(2)
            a_l = a.lower()
            col_l = col.lower()
            # If alias known and column exists there, keep
            if a_l in alias_cols and col_l in alias_cols[a_l]:
                return f"{a}.{col}"
            # Search other aliases
            for other_alias, cols in alias_cols.items():
                if col_l in cols:
                    return f"{other_alias}.{col}"
            # No alternative found; return original
            return f"{a}.{col}"

        new_query = re.sub(r"\b([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)\b", _replace_alias_col, query)
        return new_query
    except Exception as e:
        print(f"schema_aware_column_mapper failed: {e}")
        return query

# --- Manager query micro-generation helpers ---
def generate_manager_list_sql(question: str) -> Optional[str]:
    """Micro LLM generation for manager listing without static template."""
    try:
        mini_llm = initialize_llm(progressive_reduction=1)
        prompt = (
            "Generate ONLY a PostgreSQL SQL query listing current department managers.\n"
            "Use tables: department d, dept_manager dm, employee e.\n"
            "Joins: d.dept_no = dm.dept_no AND e.emp_no = dm.emp_no.\n"
            "Current managers: dm.to_date='9999-01-01'.\n"
            "Select: d.dept_name AS department_name, e.first_name, e.last_name.\n"
            "Order by d.dept_name. End with LIMIT 100; Include semicolon.\n"
            f"Question: {question}\nReturn ONLY SQL." )
        resp = TimeoutManager.run_with_timeout(
            func=mini_llm.complete,
            timeout=10.0,
            prompt=prompt
        )
        sql_txt = resp.text.strip()
        if '```' in sql_txt:
            parts = re.split(r"```(?:sql)?", sql_txt, flags=re.IGNORECASE)
            sql_txt = ''.join(p for p in parts if 'select' in p.lower()).strip()
        sql_txt = fix_common_sql_errors(sql_txt)
        needed = ['from department', 'join dept_manager', 'join employee', 'dept_name', 'limit']
        if all(n in sql_txt.lower() for n in needed) and sql_txt.lower().startswith('select'):
            if not sql_txt.strip().endswith(';'):
                sql_txt += ';'
            return sql_txt
        return None
    except Exception as e:
        print(f"Manager micro-generation failed: {e}")
        return None

def adjust_manager_query(sql_query: str, original_question: str) -> str:
    ql = original_question.lower()
    if 'manager' not in ql:
        return sql_query
    missing_dm = not re.search(r'join\s+dept_manager', sql_query, re.IGNORECASE)
    missing_e = not re.search(r'join\s+employee', sql_query, re.IGNORECASE)
    if missing_dm or missing_e:
        regen = generate_manager_list_sql(original_question)
        if regen:
            return regen
    # If only dept name selected, regenerate for full manager listing
    sel = sql_query.split('FROM')[0].lower() if 'from' in sql_query.lower() else ''
    if 'dept_name' in sel and 'first_name' not in sel and 'last_name' not in sel:
        regen = generate_manager_list_sql(original_question)
        if regen:
            return regen
    return sql_query

# --- Department listing micro-generation & adjuster ---
def generate_department_list_sql(question: str) -> Optional[str]:
    """Micro LLM generation for department name listing (pure LLM path)."""
    try:
        mini_llm = initialize_llm(progressive_reduction=1)
        prompt = (
            "Generate ONLY a concise PostgreSQL SQL query that lists all department names.\n"
            "Use table: department d. Select: d.dept_name AS department_name.\n"
            "Order alphabetically by department_name. End with LIMIT 100; Include semicolon.\n"
            f"Question: {question}\nReturn ONLY SQL." )
        resp = TimeoutManager.run_with_timeout(
            func=mini_llm.complete,
            timeout=6.0,
            prompt=prompt
        )
        sql_txt = resp.text.strip()
        if '```' in sql_txt:
            parts = re.split(r"```(?:sql)?", sql_txt, flags=re.IGNORECASE)
            sql_txt = ''.join(p for p in parts if 'select' in p.lower()).strip()
        sql_txt = fix_common_sql_errors(sql_txt)
        needed = ['select', 'from department', 'dept_name', 'limit']
        if all(n in sql_txt.lower() for n in needed) and sql_txt.lower().startswith('select'):
            if not sql_txt.strip().endswith(';'):
                sql_txt += ';'
            return sql_txt
        return None
    except Exception as e:
        print(f"Department micro-generation failed: {e}")
        return None

def adjust_department_query(sql_query: str, original_question: str) -> str:
    ql = original_question.lower()
    if not any(k in ql for k in ['department name', 'department names', 'list departments', 'departments list']):
        return sql_query
    if re.search(r'department\.department_name', sql_query, re.IGNORECASE) or re.search(r'JOIN\s+dept\s+ON', sql_query, re.IGNORECASE):
        regen = generate_department_list_sql(original_question)
        if regen:
            return regen
    if ' ON.' in sql_query and 'dept_name' in sql_query.lower():
        regen = generate_department_list_sql(original_question)
        if regen:
            return regen
    return sql_query

# --- Average salary per department micro-generation & adjuster ---
def generate_avg_salary_by_department_sql(question: str) -> Optional[str]:
    """Micro LLM generation for average salary per department (pure LLM)."""
    try:
        mini_llm = initialize_llm(progressive_reduction=1)
        prompt = (
            "Generate ONLY a PostgreSQL SQL query returning average current salary per department.\n"
            "Use tables/aliases: salary s, employee e, dept_emp de, department d.\n"
            "Joins: s.emp_no = e.emp_no; e.emp_no = de.emp_no; de.dept_no = d.dept_no.\n"
            "Current rows: s.to_date='9999-01-01' AND de.to_date='9999-01-01'.\n"
            "Select: d.dept_name AS department_name, AVG(s.salary) AS average_salary.\n"
            "Group by d.dept_name. Order by average_salary DESC. End with LIMIT 100; Include semicolon.\n"
            f"Question: {question}\nReturn ONLY SQL." )
        resp = TimeoutManager.run_with_timeout(func=mini_llm.complete, timeout=8.0, prompt=prompt)
        sql_txt = resp.text.strip()
        if '```' in sql_txt:
            parts = re.split(r"```(?:sql)?", sql_txt, flags=re.IGNORECASE)
            sql_txt = ''.join(p for p in parts if 'select' in p.lower()).strip()
        sql_txt = fix_common_sql_errors(sql_txt)
        required = ['avg', 'from salary', 'join', 'dept_name', 'group by', 'limit']
        if all(r in sql_txt.lower() for r in required) and sql_txt.lower().startswith('select'):
            if not sql_txt.strip().endswith(';'):
                sql_txt += ';'
            return sql_txt
        return None
    except Exception as e:
        print(f"Avg salary micro-generation failed: {e}")
        return None

def adjust_avg_salary_department_query(sql_query: str, original_question: str) -> str:
    ql = original_question.lower()
    if 'average salary' not in ql or 'department' not in ql:
        return sql_query
    # If question asks for an additional grouping dimension (title, job title, gender, etc.)
    # AND the SQL already joins that table, do not regenerate — the LLM handled the
    # multi-dimension correctly and regenerating would drop that dimension.
    _extra_dims = ['title', 'job title', 'gender', 'hire_date', 'position']
    if any(dim in ql for dim in _extra_dims):
        if re.search(r'\b(from|join)\s+title\b', sql_query, re.IGNORECASE):
            print(f"[ADJUSTER][AVG_SALARY] Skipping regen — SQL already has title join for multi-dimension query")
            return sql_query
    # Conditions that signal regeneration
    invalid_bare_department = re.search(r'\bselect\s+department\b', sql_query, re.IGNORECASE)
    missing_dept_join = 'join department' not in sql_query.lower() and ' department d' not in sql_query.lower()
    missing_avg = 'avg(' not in sql_query.lower()
    missing_filters = 'to_date' not in sql_query.lower()
    if invalid_bare_department or missing_dept_join or missing_avg or missing_filters:
        regen = generate_avg_salary_by_department_sql(original_question)
        if regen:
            return regen
    return sql_query

# --- Title query semantic adjuster (post-generation heuristic, no hardcoded full query) ---
def adjust_title_query(sql_query: str, original_question: str, normalized_title: Optional[str]) -> str:
    """Heuristically adjust title/engineer count queries to valid schema usage.
    Rules (non-destructive, only if clearly a title count intent):
      - Ensure alias t for title table
      - Replace misuse of 'department =' with correct t.title comparison
      - Ensure COUNT(DISTINCT t.emp_no) and current record filter t.to_date='9999-01-01'
    """
    ql = original_question.lower()
    if not any(k in ql for k in ["engineer", "title", "developer", "analyst", "staff"]):
        return sql_query
    # Guard: if user is asking for count of DISTINCT title types (not employees with a title),
    # the correct SQL is COUNT(DISTINCT title) with no emp_no or engineer filter — do not corrupt it.
    is_distinct_title_count = (
        any(k in ql for k in ["how many", "count", "number of", "different", "distinct", "types of"]) and
        any(k in ql for k in ["title", "titles", "job title", "job titles", "positions", "roles"]) and
        not any(k in ql for k in ["engineer", "engineers", "analyst", "developer", "staff"])
    )
    if is_distinct_title_count:
        print(f"[ADJUSTER][TITLE] Skipping adjust_title_query — question is asking for count of distinct title types, not employee count")
        return sql_query
    intent_is_count = any(k in ql for k in ["how many", "count"]) and any(k in ql for k in ["engineer", "engineers"])    
    # If LLM failed to reference title table at all for an engineer count intent, synthesize minimal correct pattern.
    # This is a heuristic repair (constructed dynamically) – not a stored fallback template list.
    if 'from title' not in sql_query.lower() and intent_is_count:
        target = normalized_title
        if not target:
            for pat, canon in [
                (r"senior engineers?", "Senior Engineer"),
                (r"staff engineers?", "Staff Engineer"),
                (r"assistant engineers?", "Assistant Engineer"),
                (r"engineers?", "Engineer"),
            ]:
                if re.search(pat, ql):
                    target = canon
                    break
        target = target or "Engineer"
        try:
            generated = generate_title_count_sql(original_question, target)
            if generated:
                return generated
        except Exception as gen_err:
            print(f"Title micro-generation failed: {gen_err}")
            # Fall through to incremental adjustments below without introducing a static template
            pass
    # Continue adjustments for queries that already reference title
    target = normalized_title
    # Attempt to infer target from question if not provided
    if not target:
        for pat, canon in [
            (r"senior engineers?", "Senior Engineer"),
            (r"staff engineers?", "Staff Engineer"),
            (r"assistant engineers?", "Assistant Engineer"),
            (r"engineers?", "Engineer"),
        ]:
            if re.search(pat, ql):
                target = canon
                break
    # Minimal non-template adjustments only (structural hygiene, no full pattern injection)
    # 1. Ensure alias t if title table already present and alias missing (pure structural, not full query template)
    if re.search(r"from\s+title\b", sql_query, re.IGNORECASE) and not re.search(r"from\s+title\s+(?:as\s+)?t\b", sql_query, re.IGNORECASE):
        sql_query = re.sub(r"from\s+title\b", "FROM title t", sql_query, flags=re.IGNORECASE)
    # 2. Replace misuse of department = 'X' (a semantic correction, not a canned query)
    if re.search(r"department\s*=\s*'([^']+)'", sql_query, re.IGNORECASE):
        sql_query = re.sub(r"department\s*=\s*'([^']+)'", lambda m: f"t.title = '{(target or m.group(1)).rstrip('s')}'", sql_query, flags=re.IGNORECASE)
    # 3. If we see COUNT(*) with only title table referenced, prefer regenerating via micro LLM instead of hard substituting a fragment
    if re.search(r"select\s+count\(\*\)", sql_query, re.IGNORECASE) and re.search(r"from\s+title", sql_query, re.IGNORECASE):
        regen = None
        try:
            regen = generate_title_count_sql(original_question, target or 'Engineer')
        except Exception:
            regen = None
        if regen:
            return regen
    # 4. If essential current-row filter missing, attempt micro-generation instead of injecting static predicate
    # Only trigger current-filter regen if this is TRULY an employee-with-title count query
    # (not a generic 'count distinct titles' query which doesn't need to_date)
    is_employee_title_count = intent_is_count or (target is not None)
    needs_current_filter = (
        re.search(r"from\s+title", sql_query, re.IGNORECASE) and
        't.to_date' not in sql_query.lower() and
        is_employee_title_count
    )
    if needs_current_filter:
        regen = None
        try:
            regen = generate_title_count_sql(original_question, target or 'Engineer')
        except Exception:
            regen = None
        if regen:
            return regen
    # 5. Do NOT inject LIMIT / semicolon here; global sanitizers handle generic formatting consistently
    return sql_query

# Lightweight micro-generation (still LLM based) to avoid static hardcoded pattern for title count
def generate_title_count_sql(question: str, canonical_title: str) -> Optional[str]:
    """Use a minimal LLM prompt to produce the title count SQL instead of embedding a static query.
    Returns cleaned SQL or None on failure. Keeps pure LLM path philosophy."""
    # Deterministic schema-driven builder first (no LLM)
    try:
        if 'title' in available_tables:
            try:
                cols = [c['name'].lower() for c in inspector.get_columns(table_name='title')]
            except Exception:
                cols = []

            # Required columns for deterministic build
            if 'emp_no' in cols and 'title' in cols and 'to_date' in cols:
                title_literal = canonical_title.strip()
                # Assemble SQL programmatically via build_sql helper (avoid literal concatenation)
                select_cols = ["COUNT(DISTINCT t.emp_no) AS title_count"]
                base_table = "title t"
                where = [f"LOWER(t.title) = '{title_literal.lower()}'", "t.to_date = '9999-01-01'"]
                sql = build_sql(select_cols=select_cols, base_table=base_table, where=where, limit=100)
                return sql

    except Exception as e:
        print(f"Deterministic title count builder error: {e}")

    # Fallback: small LLM-based micro-generator (existing behavior)
    try:
        mini_llm = initialize_llm(progressive_reduction=2)
        memory_manager.register_llm_instance(mini_llm)
        prompt = (
            "Generate ONLY a concise PostgreSQL SQL query counting distinct employees with a specific title.\n"
            "Constraints:\n"
            "- Use title table only with alias t\n"
            "- Current rows: t.to_date='9999-01-01'\n"
            f"- Exact title match: '{canonical_title}' (use singular)\n"
            "- COUNT(DISTINCT t.emp_no) AS TitleCount\n"
            "- End with LIMIT 100;\n"
            "Question: " + question + "\n"
            "Return ONLY SQL; no prose."
        )
        resp = TimeoutManager.run_with_timeout(
            func=mini_llm.complete,
            timeout=8.0,
            prompt=prompt
        )
        memory_manager.unregister_llm_instance(mini_llm)
        sql_txt = resp.text.strip()
        if "```" in sql_txt:
            parts = re.split(r"```(?:sql)?", sql_txt, flags=re.IGNORECASE)
            sql_txt = ''.join(p for p in parts if 'select' in p.lower()).strip()
        sql_txt = fix_common_sql_errors(sql_txt)
        # Enforce essential fragments; if missing treat as failure so caller may fallback to adjustments
        required = ["count", "from title", "t.to_date", "t.title", "limit"]
        if all(r in sql_txt.lower() for r in required):
            if not sql_txt.lower().startswith("select"):
                return None
            if not sql_txt.strip().endswith(';'):
                sql_txt += ';'
            return sql_txt
        return None
    except Exception as e:
        try:
            memory_manager.unregister_llm_instance(mini_llm)
        except Exception:
            pass
        print(f"Title count micro-generation failed: {e}")
        return None

# --- Employee count by department micro-generation & adjuster ---
def generate_employee_count_by_department_sql(question: str) -> Optional[str]:
    """Deterministic, schema-driven SQL builder for employees-per-department.

    This avoids relying on a small LLM call which has been producing
    malformed WHERE/alias combinations under low-memory/degraded-model
    conditions. The builder inspects available tables/columns and returns
    a safe SQL string or None if required tables/columns are missing.
    """
    try:
        ql = question.lower()
        superlative = any(w in ql for w in ["most", "largest", "highest"])
        least = any(w in ql for w in ["fewest", "least"])  # prefer ASC
        order_asc = least

        # Required tables
        if 'department' not in available_tables or 'dept_emp' not in available_tables:
            return None

        # Discover canonical column names
        try:
            dept_cols = [c['name'].lower() for c in inspector.get_columns(table_name='department')]
        except Exception:
            dept_cols = []
        try:
            de_cols = [c['name'].lower() for c in inspector.get_columns(table_name='dept_emp')]
        except Exception:
            de_cols = []

        # Choose dept name column (dept_name preferred)
        dept_name_col = 'dept_name' if 'dept_name' in dept_cols else (dept_cols[0] if dept_cols else 'dept_name')
        # dept_no column name
        dept_no_col = 'dept_no' if 'dept_no' in dept_cols or 'dept_no' in de_cols else (dept_cols[0] if dept_cols else 'dept_no')
        # emp_no column
        emp_no_col = 'emp_no' if 'emp_no' in de_cols else (de_cols[0] if de_cols else 'emp_no')

        # Build SQL programmatically
        order_dir = 'ASC' if order_asc else 'DESC'
        limit_clause = 'LIMIT 1' if superlative or least else 'LIMIT 100'

        select_cols = [f"d.{dept_name_col} AS department_name", f"COUNT(DISTINCT de.{emp_no_col}) AS employee_count"]
        base_table = f"department d"
        joins = [f"JOIN dept_emp de ON d.{dept_no_col} = de.{dept_no_col}"]

        where_clauses = []
        # Prefer standard current-row sentinel if available
        if 'to_date' in de_cols:
            where_clauses.append("de.to_date = '9999-01-01'")

        parts = []
        parts.append("SELECT " + ", ".join(select_cols))
        parts.append("FROM " + base_table)
        parts.extend(joins)
        if where_clauses:
            parts.append("WHERE " + " AND ".join(where_clauses))
        parts.append("GROUP BY d." + dept_name_col)
        parts.append(f"ORDER BY employee_count {order_dir}")
        parts.append(limit_clause + ';')

        sql = " ".join(parts)
        return sql
    except Exception as e:
        print(f"Employee count builder failed: {e}")
        return None

def adjust_employee_count_by_department_query(sql_query: str, original_question: str) -> str:
    ql = original_question.lower()
    if not ('department' in ql and any(k in ql for k in ['count', 'how many', 'number of', 'most', 'largest', 'highest', 'fewest', 'least']) and any(k in ql for k in ['employee', 'employees'])):
        return sql_query
    # If missing essential join or aggregation/grouping, regenerate
    missing_join = 'join dept_emp' not in sql_query.lower()
    missing_group = 'group by' not in sql_query.lower()
    missing_count = 'count(' not in sql_query.lower()
    if missing_join or missing_group or missing_count:
        regen = generate_employee_count_by_department_sql(original_question)
        if regen:
            return regen
    # Superlative queries should generally have LIMIT 1
    if any(w in ql for w in ['most','largest','highest','fewest','least']) and 'limit 1' not in sql_query.lower():
        regen = generate_employee_count_by_department_sql(original_question)
        if regen:
            return regen
    return sql_query

async def present_results(question: str, sql_query: str) -> str:
    """Execute the query and present results with intelligent formatting and insights."""
    print("[PRESENTER] Starting intelligent result presentation...")
    
    try:
        # Execute the query with diagnostic verification
        print("[PRESENTER] Executing query...")
        results = await execute_query(sql_query)
        
        # Perform result verification for suspicious patterns
        if results and isinstance(results, list) and len(results) <= 5:
            try:
                flat_vals = []
                for r in results:
                    for v in r.values():
                        if isinstance(v, (int, float)):
                            flat_vals.append(v)
                
                if flat_vals and all(v == flat_vals[0] for v in flat_vals):
                    if flat_vals[0] in (1103, 1000, 999):
                        print(f"[PRESENTER] Detected uniform result pattern {flat_vals} - verifying accuracy...")
                    
                    lowered_q = question.lower()
                    is_simple_total = re.search(r"total number of employees|total employees|count of employees|how many employees", lowered_q) and 'department' not in lowered_q and 'each' not in lowered_q
                    simple_sql_shape = re.match(r"^\s*select\s+count\s*\(\s*\*\s*\)\s+as\s+total_employees\s+from\s+employee\s*;?\s*$", sql_query, flags=re.IGNORECASE)
                    
                    if is_simple_total or simple_sql_shape:
                        try:
                            print("[PRESENTER] Cross-verifying employee count...")
                            verification_results = []
                            verify_queries = [
                                build_sql(select_cols=["COUNT(*) AS total_emp_check"], base_table="employee", limit=None),
                                build_sql(select_cols=["COUNT(DISTINCT emp_no) AS total_emp_distinct"], base_table="employee", limit=None)
                            ]
                            for vq in verify_queries:
                                try:
                                    vr = await execute_query(vq)
                                    verification_results.append((vq, vr))
                                except Exception as verr:
                                    verification_results.append((vq, str(verr)))
                            
                            counts = []
                            for _, vr in verification_results:
                                if isinstance(vr, list) and vr and isinstance(vr[0], dict):
                                    for val in vr[0].values():
                                        if isinstance(val, (int, float)):
                                            counts.append(val)
                            
                            if len(counts) >= 2 and counts[0] != counts[1]:
                                print(f"[PRESENTER] Count verification mismatch detected: {counts}")
                            else:
                                print(f"[PRESENTER] Count verification successful: {counts[0] if counts else 'N/A'}")
                        except Exception as ver_err:
                            print(f"[PRESENTER] Verification failed: {ver_err}")
            except Exception:
                pass
        
        # Apply intelligent formatting and context
        print("[PRESENTER] Applying intelligent formatting...")
        formatted_response = format_results(results, question)
        
        # Add presentation insights
        if results:
            result_count = len(results)
            print(f"[PRESENTER] Presenting {result_count} result(s) with contextual insights")
        else:
            print("[PRESENTER] No results found - providing explanatory response")
        
        print("[PRESENTER] Result presentation completed successfully")
        return formatted_response
        
    except Exception as e:
        error_msg = f"[PRESENTER] Error during result presentation: {str(e)}"
        print(error_msg)
        return f"I encountered an error while presenting the results: {str(e)}"

async def agentic_query_process(question: str) -> str:
    """Process a query through an agentic workflow."""
    print(f"[DEBUG][INCOMPLETE_FUNCTION] Called with question: '{question}'")
    # Generate SQL query from LLM
    sql_query = query_craft(question)
    # Validate the query before fixing
    validation = enhanced_validate_sql(sql_query)
    if validation["valid"]:
        print("[DEBUG][AGENTIC] Query is valid, skipping fixer.")
        fixed_query = sql_query
    else:
        print(f"[DEBUG][AGENTIC] Query has issues: {validation['issues']}. Applying fixer.")
        fixed_query = fix_common_sql_errors(sql_query)
    # Continue with the rest of the workflow using fixed_query
    # (The rest of the function remains unchanged)
async def get_doc_text(doc_id: str) -> Optional[str]:
    """Return stored OCR text for a given doc_id if present in chat_store/docs."""
    try:
        docs_dir = Path('chat_store') / 'docs'
        fpath = docs_dir / f"{doc_id}.txt"
        if fpath.exists():
            return fpath.read_text(encoding='utf-8')
    except Exception:
        pass
    return None


async def agentic_query_process(question: str, doc_id: Optional[str] = None) -> str:
    """Process a query through an agentic workflow. Optionally include OCR text from doc_id."""
    
    print(f"[ENTRY_POINT_DEBUG] agentic_query_process called with question: '{question}'")
    print(f"[ENTRY_POINT_DEBUG] doc_id: {doc_id}")
    
    # Initialize database if not already done
    global async_session, available_tables
    if async_session is None or not available_tables:
        print("--- Database Initialization ---")
        db_metadata = await initialize_database(uri=DB_CONNECTION_URL)
        if not db_metadata["tables"]:
            return "Database initialization failed: No tables found"
        print(f"Database initialized with tables: {available_tables}")
    
    # If a doc_id is provided, fetch its OCR text and append to the question for context
    if doc_id:
        try:
            doc_text = await get_doc_text(doc_id)
            if doc_text:
                try:
                    from os import getenv as _oget
                    snippet_limit = int(_oget("OCR_SNIPPET_LIMIT", "2000"))
                except Exception:
                    snippet_limit = 2000
                if snippet_limit <= 0:
                    snippet_limit = 2000
                snippet = doc_text[:snippet_limit]
                question = (
                    question
                    + f"\n\n[Document {doc_id} excerpt start]\n{snippet}\n[Document excerpt end]"
                )
        except Exception as e:
            print(f"Warning: failed to include doc_text for {doc_id}: {e}")

    # First clean up the question to handle common typos
    print(f"[TYPO_DEBUG] Question before typo correction: '{question}'")
    corrected_question = question
    common_typos = {
        "deparment": "department",
        "deparments": "departments", 
        "avarage": "average",
        # Removed "employe": "employee" and "employes": "employees" to prevent over-correction
    }
    for typo, correction in common_typos.items():
        corrected_question = corrected_question.replace(typo, correction)
    if corrected_question != question:
        print(f"Corrected query: '{corrected_question}' (was: '{question}')")
        question = corrected_question

    # Initial memory cleanup before processing
    print(f"\n--- Initial Memory Cleanup for question: {question} ---")
    gc.collect()
    stats_before = memory_manager.get_memory_stats()
    memory_manager.force_collect_garbage()
    stats_after = memory_manager.get_memory_stats()
    print(f"Memory before cleanup: {stats_before['available_mb']:.2f}MB")
    print(f"Memory after cleanup: {stats_after['available_mb']:.2f}MB")
    print(f"Memory reclaimed: {stats_after['available_mb'] - stats_before['available_mb']:.2f}MB")

    # Check if there's enough memory to safely process the request
    if stats_after['available_mb'] < memory_manager.critical_mb:
        print(f"System is critically low on memory: {stats_after['available_mb']:.2f}MB available")
        print("LLM-based approach not viable due to critical memory constraints")
        return "I'm sorry, but your system is critically low on available memory. Please close some applications and try again, or try a simpler query."

    print("\n=== MEMORY AGENT ===")
    
    # Handle re-execute requests by cleaning the question first
    original_question = question
    clean_question = question
    for keyword in ["re-execute", "reexecute", "regenerate", "try again", "force", "bypass memory"]:
        clean_question = clean_question.replace(keyword, "").strip()
    
    # Check if this is a re-execute request
    is_reexecute = any(k in original_question.lower() for k in ["re-execute", "reexecute", "regenerate", "try again", "force", "bypass memory"])
    
    if is_reexecute and clean_question and len(clean_question.strip()) > 0:
        print(f"User requested to re-execute. Using question: '{clean_question}'")
        question = clean_question  # Use the cleaned question for processing
        print("Bypassing memory check for re-execute request.")
    else:
        # Normal memory check
        memory_result = memory_check(question)
        if memory_result.startswith("FOUND_IN_MEMORY:"):
            print("Found previous answer in memory")
            return memory_result.replace("FOUND_IN_MEMORY: ", "")
        elif memory_result.startswith("FOUND_IN_MEMORY_WITH_ERROR:"):
            print("Found previous answer with error. Suggesting to retry.")
            error_message = memory_result.replace("FOUND_IN_MEMORY_WITH_ERROR: ", "")
            return f"{error_message}\n\nIt looks like there was an error with this query. You can try asking again with 're-execute' to regenerate the query."
        elif memory_result.startswith("RE_EXECUTE_ORIGINAL:"):
            print("User requested to re-execute. Retrieving original question.")
            # Only use the fallback question if user didn't specify a question after "re-execute"
            if clean_question and len(clean_question.strip()) > 0:
                print(f"User provided specific question for re-execute: '{clean_question}' - using that instead of fallback")
                question = clean_question
            else:
                print("User didn't specify question for re-execute, using fallback from memory")
                question = memory_result.replace("RE_EXECUTE_ORIGINAL: ", "")
            print(f"Final question for re-execution: {question}")
        elif "Bypassing memory check" in memory_result:
            print("User requested to re-execute. Bypassing memory check.")
            # Continue with query generation
        else:
            print("No previous answer found in memory")

    # Memory check again before heavy processing
    print("\n--- Pre-Generation Memory Check ---")
    stats = memory_manager.get_memory_stats()
    if stats['available_mb'] < memory_manager.threshold_mb:
        print(f"⚠️ Memory low before query generation: {stats['available_mb']:.2f}MB available")
        memory_manager.force_collect_garbage()
        stats = memory_manager.get_memory_stats()
        if stats['available_mb'] < memory_manager.critical_mb:
            return "I'm sorry, but your system is currently low on available memory. Please close some applications and try again, or try a simpler query."

    print("\n=== QUERY CRAFT AGENT ===")
    try:
        gc.collect()
        try:
            sql_query = query_craft(question)
        except ValueError as gen_fail:
            ql = question.lower()
            if ('manager' in ql and ('male' in ql or 'female' in ql or 'gender' in ql)):
                print("[FALLBACK] Building manager gender count query programmatically after LLM failure")
                select_cols = ["e.gender", "COUNT(*) AS manager_count"]
                joins = ["JOIN dept_manager dm ON dm.emp_no = e.emp_no"]
                sql_query = build_sql(select_cols=select_cols, base_table="employee e", joins=joins, group_by=["e.gender"], order_by="manager_count DESC", limit=None)
            else:
                raise
        try:
            global LAST_GENERATED_SQL
            LAST_GENERATED_SQL = sql_query
        except Exception:
            pass
        try:
            print("<<<SQL_START>>>")
            print(sql_query.strip())
            print("<<<SQL_END>>>")
        except Exception:
            print(f"Generated SQL query: {sql_query}")

        # Invariant enforcement (classification + constraint-driven regeneration).
        if ENABLE_INVARIANTS:
            # Quick validation check BEFORE running expensive invariants.
            # If the SQL already passes validation, invariants will only produce false positives
            # and waste 90-160s. Use a flag to gate the entire enforce_invariants block.
            _skip_invariants = False
            try:
                _pre_inv_valid = enhanced_validate_sql(sql_query)
                _pre_inv_base = validate_sql_query(sql_query)
                if _pre_inv_valid.get('valid') and _pre_inv_base.get('valid'):
                    print("[INVARIANTS] SQL passed pre-check — skipping enforce_invariants (already valid)")
                    print("<<<SQL_REVISED_START>>>")
                    print(sql_query.strip())
                    print("<<<SQL_REVISED_END>>>")
                    LAST_GENERATED_SQL = sql_query
                    _skip_invariants = True
            except Exception:
                pass  # If pre-check fails, fall through to normal invariants

            if not _skip_invariants:
                try:
                    try:
                        cat = classify_question(question)
                    except Exception:
                        cat = None
                    _sql_normalized = re.sub(r'\s+', ' ', sql_query).lower()
                    _has_dept_emp = 'dept_emp' in _sql_normalized
                    _has_gender = 'gender' in _sql_normalized or 'e.gender' in _sql_normalized
                    if cat == 'EMP_COUNT_BY_DEPT' and not _has_dept_emp and not _has_gender:
                        rescue = generate_employee_count_by_department_sql(question)
                        if rescue:
                            print("[RESCUE] Replaced incomplete employee count query with micro-generated version before invariants")
                            sql_query = rescue
                    classification_question = re.sub(r'^re-execute\s+', '', question, flags=re.IGNORECASE)
                    inv_result = enforce_invariants(classification_question, sql_query, Settings.llm, run_llm_with_timeout)
                    if inv_result.get('regenerations'):
                        for regen in inv_result['regenerations']:
                            print(f"[INVARIANTS] Regen {regen['attempt']} fixed missing {regen['missing_before']}")
                    if 'remaining_missing' in inv_result and inv_result['remaining_missing']:
                        print(f"[INVARIANTS] Remaining unmet invariants: {inv_result['remaining_missing']}")
                        try:
                            micro_map = {
                                'count_depts': generate_department_list_sql,
                                'emp_count_by_dept': generate_employee_count_by_department_sql,
                                'count_employees_by_department': generate_employee_count_by_department_sql,
                                'EMP_COUNT_BY_DEPT': generate_employee_count_by_department_sql,
                            }
                            for miss in inv_result['remaining_missing']:
                                key = miss.lower()
                                gen_fn = micro_map.get(key)
                                if gen_fn:
                                    try:
                                        print(f"[INVARIANTS] Attempting micro-generator for unmet invariant '{miss}' using {gen_fn.__name__}")
                                        micro_sql = gen_fn(question)
                                        if micro_sql:
                                            print(f"[INVARIANTS] Micro-generator produced SQL for '{miss}': {micro_sql}")
                                            sql_query = micro_sql
                                            break
                                    except Exception as micro_err:
                                        print(f"[INVARIANTS] Micro-generator {gen_fn.__name__} failed: {micro_err}")
                        except Exception as micro_map_err:
                            print(f"[INVARIANTS] Micro-generator mapping failed: {micro_map_err}")
                    if not sql_query or (isinstance(sql_query, str) and sql_query.strip() == ''):
                        sql_query = inv_result.get('final_sql', sql_query)
                    else:
                        try:
                            print("<<<SQL_REVISED_START>>>")
                            print(sql_query.strip())
                            print("<<<SQL_REVISED_END>>>")
                            LAST_GENERATED_SQL = sql_query
                        except Exception:
                            print(f"[INVARIANTS] Final SQL after enforcement/micro-gen: {sql_query}")
                    if not sql_query:
                        sql_query = inv_result.get('final_sql', sql_query)
                except Exception as inv_err:
                    print(f"[INVARIANTS] Enforcement skipped due to error: {inv_err}")

        memory_manager.force_collect_garbage()

        # --- MAIN PATCH: Only fix if invalid, and always use the validated/fixed query ---
        validation = enhanced_validate_sql(sql_query)
        if validation["valid"]:
            fixed_query = sql_query
            print("[DEBUG][AGENTIC] Query is valid, skipping fixer.")
        else:
            print(f"[DEBUG][AGENTIC] Query has issues: {validation['issues']}. Applying fixer.")
            fixed_query = fix_common_sql_errors(sql_query)

        # All subsequent processing uses fixed_query only
        sql_query = fixed_query

        # --- Unconditional semantic fixes (run even when fixer was skipped) ---
        # Fix 1: dept_no = 'DeptName' → dept_no IN (SELECT dept_no FROM department WHERE dept_name = 'DeptName')
        # This pattern is syntactically valid but semantically wrong (dept_no is a code, not a name)
        try:
            _dn_match = re.search(r"\bdept_no\s*=\s*'([A-Za-z][A-Za-z\s&]*)'", sql_query, re.IGNORECASE)
            if _dn_match:
                _dn_val = _dn_match.group(1)
                if not re.match(r'^d\d+$', _dn_val.strip(), re.IGNORECASE):
                    _dn_fix = f"dept_no IN (SELECT dept_no FROM department WHERE dept_name = '{_dn_val}')"
                    sql_query = re.sub(
                        r"\bdept_no\s*=\s*'" + re.escape(_dn_val) + r"'",
                        _dn_fix, sql_query, flags=re.IGNORECASE
                    )
                    print(f"[POST-FIX] dept_no = '{_dn_val}' replaced with subquery lookup")
        except Exception:
            pass

        # Fix 2: department.dept_name = 'X' used in WHERE but department table not in FROM/JOIN
        # → add the missing dept_emp + department joins
        try:
            _has_dept_filter = re.search(r"\bdepartment\.dept_name\s*=\s*'([^']+)'", sql_query, re.IGNORECASE)
            _has_dept_join = re.search(r'\bjoin\s+department\b|\bfrom\s+department\b', sql_query, re.IGNORECASE)
            _has_dept_emp_join = re.search(r'\bjoin\s+dept_emp\b|\bfrom\s+dept_emp\b', sql_query, re.IGNORECASE)
            if _has_dept_filter and not _has_dept_join:
                _dept_filter_val = _has_dept_filter.group(1)
                # Find the first FROM table to determine the join anchor
                _from_match = re.search(r'\bFROM\s+(\w+)', sql_query, re.IGNORECASE)
                _anchor = _from_match.group(1) if _from_match else 'salary'
                # Build the missing joins - anchor through employee if present
                _has_emp = re.search(r'\bjoin\s+employee\b|\bfrom\s+employee\b', sql_query, re.IGNORECASE)
                if _has_emp and not _has_dept_emp_join:
                    _join_to_add = (
                        "JOIN dept_emp de ON e.emp_no = de.emp_no "
                        "JOIN department d ON de.dept_no = d.dept_no"
                    )
                elif not _has_dept_emp_join:
                    _join_to_add = (
                        f"JOIN dept_emp de ON {_anchor}.emp_no = de.emp_no "
                        "JOIN department d ON de.dept_no = d.dept_no"
                    )
                else:
                    _join_to_add = "JOIN department d ON de.dept_no = d.dept_no"
                # Insert join before WHERE
                sql_query = re.sub(r'\bWHERE\b', _join_to_add + ' WHERE', sql_query, count=1, flags=re.IGNORECASE)
                # Replace department.dept_name with d.dept_name (alias form)
                sql_query = re.sub(r'\bdepartment\.dept_name\b', 'd.dept_name', sql_query, flags=re.IGNORECASE)
                print(f"[POST-FIX] Added missing department JOIN for filter dept_name = '{_dept_filter_val}'")
        except Exception:
            pass

        # Fix 3: "Which X has the highest/most/lowest/fewest..." → singular superlative → LIMIT 1
        # These questions expect a single top result, not a ranked list of all rows.
        try:
            _singular_superlative = re.search(
                r'\bwhich\b.{1,40}\b(highest|most|lowest|fewest|best|worst|largest|smallest|greatest)\b',
                question.lower()
            )
            if _singular_superlative:
                # Only change LIMIT 100 (the default) — never touch intentional LIMIT N values
                if re.search(r'\bLIMIT\s+100\b', sql_query, re.IGNORECASE):
                    sql_query = re.sub(r'\bLIMIT\s+100\b', 'LIMIT 1', sql_query, flags=re.IGNORECASE)
                    print("[POST-FIX] Changed LIMIT 100 -> LIMIT 1 for singular superlative query")
        except Exception:
            pass

        # Fix 4: department alias used with columns it doesn't have (emp_no, from_date, to_date)
        # These columns belong to dept_manager. When both 'd' (department) and 'dm' (dept_manager)
        # are present, redirect d.emp_no -> dm.emp_no, d.from_date -> dm.from_date, d.to_date -> dm.to_date
        # and fix wrong JOIN conditions like "ON d.emp_no = e.emp_no" -> "ON dm.emp_no = e.emp_no"
        try:
            _dept_alias = re.search(r'\bFROM\s+department\s+(\w+)\b', sql_query, re.IGNORECASE)
            _dm_alias = re.search(r'\b(?:FROM|JOIN)\s+dept_manager\s+(\w+)\b', sql_query, re.IGNORECASE)
            if _dept_alias and _dm_alias:
                _da = _dept_alias.group(1)   # e.g. 'd'
                _dma = _dm_alias.group(1)    # e.g. 'dm'
                _cols_department_lacks = ['emp_no', 'from_date', 'to_date']
                _fixed = False
                for _col in _cols_department_lacks:
                    _bad = rf'\b{re.escape(_da)}\.{_col}\b'
                    if re.search(_bad, sql_query, re.IGNORECASE):
                        sql_query = re.sub(_bad, f'{_dma}.{_col}', sql_query, flags=re.IGNORECASE)
                        _fixed = True
                if _fixed:
                    print(f"[POST-FIX] Redirected {_da}.{{emp_no|from_date|to_date}} -> {_dma}.* (department lacks these columns)")
        except Exception:
            pass

        # Fix 5: The employee table does not have a 'to_date' column.
        # If employee is the ONLY table (no JOINs) but the query has "to_date = '9999-01-01'", remove it.
        # This happens because the system prompt universally instructs filtering current rows with to_date.
        try:
            if re.search(r'\bFROM\s+employee\b', sql_query, re.IGNORECASE) and not re.search(r'\bJOIN\b', sql_query, re.IGNORECASE):
                if re.search(r'\bto_date\s*=\s*\'9999-01-01\'', sql_query, re.IGNORECASE):
                    # Remove if it's the first/only condition
                    sql_query = re.sub(r'\bWHERE\s+(?:e\.)?to_date\s*=\s*\'9999-01-01\'\s*(?:AND)?\s*', 'WHERE ', sql_query, flags=re.IGNORECASE)
                    # Remove if it's a subsequent condition
                    sql_query = re.sub(r'\bAND\s+(?:e\.)?to_date\s*=\s*\'9999-01-01\'\s*', '', sql_query, flags=re.IGNORECASE)
                    # Cleanup trailing empty WHERE clauses
                    sql_query = re.sub(r'\bWHERE\s+(GROUP\s+BY|ORDER\s+BY|LIMIT|;|$)', r'\1', sql_query, flags=re.IGNORECASE)
                    sql_query = re.sub(r'\bWHERE\s*$', '', sql_query, flags=re.IGNORECASE)
                    print("[POST-FIX] Removed invalid 'to_date' filter (employee table has no to_date column)")
        except Exception:
            pass

        # Fix 6: hire_date only exists in the employee table. If it is prefixed by an alias belonging to another table,
        # redirect it to the employee table's alias (or 'employee' if no alias).
        try:
            if re.search(r'\bhire_date\b', sql_query, re.IGNORECASE):
                _emp_alias_match = re.search(r'\b(?:FROM|JOIN)\s+employee\s+(?:AS\s+)?(\w+)\b', sql_query, re.IGNORECASE)
                _emp_alias = _emp_alias_match.group(1) if _emp_alias_match else 'employee'
                
                def _fix_hire_date_alias(match):
                    _prefix = match.group(1)
                    if _prefix.lower() not in [_emp_alias.lower(), 'employee']:
                        return f"{_emp_alias}.hire_date"
                    return match.group(0)
                    
                _sql_query_new = re.sub(r'\b(\w+)\.hire_date\b', _fix_hire_date_alias, sql_query, flags=re.IGNORECASE)
                if _sql_query_new != sql_query:
                    print(f"[POST-FIX] Redirected wrong alias for hire_date to {_emp_alias}.hire_date")
                    sql_query = _sql_query_new
        except Exception:
            pass

        # Continue with the rest of the workflow as before, but always use sql_query (now validated/fixed)
        # Pre-execution semantic/alias guard for title queries
        if re.search(r"from\s+title\b", sql_query, re.IGNORECASE):
            if re.search(r"t\.(?:emp_no|title|to_date)", sql_query, re.IGNORECASE) and not re.search(r"from\s+title\s+(?:as\s+)?t\b", sql_query, re.IGNORECASE):
                print("Injecting missing alias 't' for title before execution")
                sql_query = re.sub(r"from\s+title\b", "FROM title t", sql_query, flags=re.IGNORECASE)
            if re.search(r"where\s+.*department\s*=", sql_query, re.IGNORECASE) and not re.search(r"join\s+department", sql_query, re.IGNORECASE):
                print("Adjusting misuse of department in title-only query before execution")
                sql_query = adjust_title_query(sql_query, question, None)
        # Sanitize: move misplaced WHERE clauses that appear after GROUP BY/ORDER BY back to before them
        try:
            upper_sql = sql_query.upper()
            where_m = re.search(r'\bWHERE\b', upper_sql)
            if where_m:
                where_idx = where_m.start()
                remainder = upper_sql[where_idx + 5:]
                next_clause_m = re.search(r'\b(GROUP\s+BY|ORDER\s+BY|LIMIT|HAVING)\b', remainder)
                if next_clause_m:
                    next_rel = where_idx + 5 + next_clause_m.start()
                    where_clause = sql_query[where_idx:next_rel]
                    sql_query = sql_query[:where_idx] + sql_query[next_rel:]
                    grp_m = re.search(r'\bGROUP\s+BY\b', sql_query, re.IGNORECASE)
                    ord_m = re.search(r'\bORDER\s+BY\b', sql_query, re.IGNORECASE)
                    lim_m = re.search(r'\bLIMIT\b', sql_query, re.IGNORECASE)
                    if grp_m:
                        insert_at = grp_m.start()
                    elif ord_m:
                        insert_at = ord_m.start()
                    elif lim_m:
                        insert_at = lim_m.start()
                    else:
                        insert_at = len(sql_query)
                    sql_query = sql_query[:insert_at] + where_clause + ' ' + sql_query[insert_at:]
                    print("[AUTO-FIX] Repositioned WHERE clause before GROUP/ORDER/LIMIT.")
        except Exception:
            pass
        try:
            # DISABLED: Schema mapper has a bug where it replaces table aliases with SQL keywords
            # See GitHub issue: regex captures "ON" and "JOIN" as aliases, causing syntax errors
            # The 3-level architecture generates correct SQL, so this post-processing is unnecessary
            skip_schema_mapper = True  # FORCE DISABLED to prevent corruption
            if not skip_schema_mapper:
                normalized_sql = schema_aware_column_mapper(sql_query)
                if normalized_sql.strip() != sql_query.strip():
                    print("[SCHEMA-MAPPER] Rewrote SQL to match inspected schema.")
                    print("Before:", sql_query)
                    print("After:", normalized_sql)
                    sql_query = normalized_sql
            else:
                print("[SCHEMA-MAPPER] Disabled to prevent corruption of valid SQL")
        except Exception as _map_err:
            print(f"Schema mapper error (continuing with original SQL): {_map_err}")

        print(f"Executing query: {sql_query}")
        if re.search(r"count\s*\(", sql_query, re.IGNORECASE):
            print("[DIAGNOSTIC] Detected COUNT query. Full SQL (sanitized):")
            print(sql_query)
        # Fix stray semicolons before LIMIT — LLM sometimes puts ';' after ORDER BY clause
        # Pattern 1: explicit ;\s*LIMIT  (original)
        if re.search(r";\s*LIMIT\s+\d+", sql_query, re.IGNORECASE):
            sql_query = re.sub(r";\s*(LIMIT\s+\d+)", r" \1", sql_query, flags=re.IGNORECASE)
            print("[AUTO-FIX] Removed stray semicolon before LIMIT")
        # Pattern 2: semicolon after ORDER BY / GROUP BY / HAVING clause, before LIMIT on next line
        sql_query = re.sub(r"(ORDER\s+BY\s+[^;]+);(\s*LIMIT)", r"\1\2", sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(r"(GROUP\s+BY\s+[^;]+);(\s*LIMIT)", r"\1\2", sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(r";\s*;+$", ";", sql_query)
        try:
            if re.search(r'\b(count|sum|avg|min|max)\s*\(', sql_query, flags=re.IGNORECASE) and 'group by' not in sql_query.lower():
                sql_query = re.sub(r"\s+LIMIT\s+\d+\s*;?\s*$", ";" if sql_query.strip().endswith(';') else '', sql_query, flags=re.IGNORECASE)
                sql_query = re.sub(r"\s+LIMIT\s+\d+\b", '', sql_query, flags=re.IGNORECASE)
        except Exception:
            pass
        if re.search(r"\bdde\.", sql_query):
            print("[AUTO-FIX] Correcting alias typo 'dde.' to 'de.'")
            sql_query = re.sub(r"\bdde\.", "de.", sql_query)
        if re.search(r"\bdee\.", sql_query):
            print("[AUTO-FIX] Correcting alias typo 'dee.' to 'de.'")
            sql_query = re.sub(r"\bdee\.", "de.", sql_query)
        if re.search(r"avg\s*\(\s*s\.salary", sql_query, re.IGNORECASE) and 'group by' in sql_query.lower():
            # Only regenerate if SQL has actual issues — skip if already valid
            _regen_pre_check = enhanced_validate_sql(sql_query)
            if not _regen_pre_check.get('valid'):
                print("[REGEN] Regenerating AVG salary per department query for correct structure via LLM.")
                from .invariants import build_invariants, evaluate_invariants, regenerate_with_constraints, CATEGORY_AVG_SALARY
                cat = classify_question(question)
                if cat == CATEGORY_AVG_SALARY:
                    invs = build_invariants(cat, question)
                    missing = evaluate_invariants(sql_query, invs)
                    if missing:
                        regen_sql = regenerate_with_constraints(question, sql_query, missing, Settings.llm, run_llm_with_timeout, attempt=0)
                        if regen_sql:
                            sql_query = regen_sql
            else:
                print("[REGEN] AVG salary REGEN skipped — SQL already valid")
        if re.match(r"\s*select\s+count\s*\(\s*\*\s*\)\s+from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+limit", sql_query, flags=re.IGNORECASE) and 'group by' not in sql_query.lower():
            print("[REGEN] Regenerating simple total count query for correct structure via LLM.")
            from .invariants import build_invariants, evaluate_invariants, regenerate_with_constraints, CATEGORY_SIMPLE_COUNT_EMP
            cat = classify_question(question)
            if cat == CATEGORY_SIMPLE_COUNT_EMP:
                invs = build_invariants(cat, question)
                missing = evaluate_invariants(sql_query, invs)
                if missing:
                    regen_sql = regenerate_with_constraints(question, sql_query, missing, Settings.llm, run_llm_with_timeout, attempt=0)
                    if regen_sql:
                        sql_query = regen_sql
        if re.search(r"group\s+by\s+.*dept_name", sql_query, re.IGNORECASE) and re.search(r"count\s*\(\s*\*\s*\)", sql_query, re.IGNORECASE) and 'employee_count' not in sql_query.lower():
            try:
                select_part, rest = sql_query.split('FROM', 1)
                select_part_fixed = re.sub(r"count\s*\(\s*\*\s*\)\s*(?!as)", "COUNT(*) AS employee_count ", select_part, count=1, flags=re.IGNORECASE)
                sql_query = select_part_fixed + 'FROM' + rest
            except Exception:
                pass
        if ('dept_name' in sql_query.lower() and 'count(' in sql_query.lower() and 'dept_emp' not in sql_query.lower() and re.search(r'from\s+department', sql_query, re.IGNORECASE)):
            try:
                from .invariants import build_invariants, evaluate_invariants, regenerate_with_constraints, CATEGORY_EMP_COUNT_BY_DEPT
                cat = classify_question(question)
                if cat == CATEGORY_EMP_COUNT_BY_DEPT:
                    invs = build_invariants(cat, question)
                    missing = evaluate_invariants(sql_query, invs)
                    if missing:
                        print('[REGEN] Initiating LLM regeneration to add missing dept_emp join and related invariants')
                        regen_sql = regenerate_with_constraints(question, sql_query, missing, Settings.llm, run_llm_with_timeout, attempt=0)
                        if regen_sql:
                            sql_query = regen_sql
            except Exception as regen_err:
                print(f'[REGEN] Skipped regeneration due to error: {regen_err}')
        if re.search(r"having\s+count\s*\(\s*\*\s*\)\s*=\s*\d+", sql_query, re.IGNORECASE) and 'join dept_emp' not in sql_query.lower():
            repair = generate_employee_count_by_department_sql(question)
            if repair:
                print("[REPAIR] Replaced malformed HAVING constant count query with micro-generated employee count query")
                sql_query = repair
        hallucinated_tables = []
        # Strip out EXTRACT(... FROM ...) and similar functions before scanning for tables
        _sql_for_tables = re.sub(r'\bEXTRACT\s*\([^)]+\)', '', sql_query, flags=re.IGNORECASE)
        for tbl in re.findall(r'\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)\b', _sql_for_tables, re.IGNORECASE):
            if tbl and tbl.lower() not in [x.lower() for x in available_tables]:
                hallucinated_tables.append(tbl)
        if hallucinated_tables:
            print(f"[REGEN] Hallucinated tables detected: {hallucinated_tables}. Regenerating SQL without them.")
            feedback = f"Do NOT use these tables: {', '.join(hallucinated_tables)}. Use only these: {', '.join(available_tables)}. For current rows, use WHERE ...to_date = '9999-01-01'."
            regen_sql = query_craft(question + " " + feedback)
            sql_query = regen_sql
        print("\n=== RESULT PRESENTER AGENT ===")
        print("Executing query and preparing intelligent presentation...")
        
        # Execute query using the present_results function
        final_result = await present_results(question, sql_query)
        
        # Add diagnostic verification for result quality
        if final_result and isinstance(final_result, str):
            print(f"[PRESENTER] Generated response of {len(final_result)} characters")
            if len(final_result) > 100:
                print("[PRESENTER] Response includes contextual insights and formatting")
            else:
                print("[PRESENTER] Concise response provided")
        
        print("[PRESENTER] Result presentation completed successfully")
    except Exception as exec_error:
        error_msg = str(exec_error)
        if 'syntax error' in error_msg.lower():
            corrected = sql_query
            corrected = re.sub(r';\s*LIMIT', ' LIMIT', corrected, flags=re.IGNORECASE)
            if re.search(r'HAVING\s+COUNT\s*\(\s*\*\s*\)\s*=\s*\d+', corrected, flags=re.IGNORECASE) and not re.search(r'SELECT\s+.*COUNT\s*\(', corrected, flags=re.IGNORECASE):
                try:
                    if re.search(r'from\s+department', corrected, re.IGNORECASE):
                        corrected = re.sub(r'^SELECT\s+dept_name', 'SELECT dept_name, COUNT(*) AS employee_count', corrected, flags=re.IGNORECASE)
                        corrected = re.sub(r'HAVING\s+COUNT\s*\(\s*\*\s*\)\s*=\s*\d+\s*', '', corrected, flags=re.IGNORECASE)
                        if 'GROUP BY' not in corrected.upper():
                            corrected = re.sub(r'FROM', 'FROM', corrected)
                        if not re.search(r'GROUP\s+BY', corrected, re.IGNORECASE):
                            corrected = re.sub(r'LIMIT', 'GROUP BY dept_name LIMIT', corrected, flags=re.IGNORECASE)
                        if 'employee_count' not in corrected.lower():
                            corrected = re.sub(r'COUNT\s*\(\s*\*\s*\)', 'COUNT(*) AS employee_count', corrected, count=1, flags=re.IGNORECASE)
                except Exception:
                    pass
            if corrected != sql_query:
                print(f"[AUTO-FIX] Applying syntax corrections: {corrected}")
                try:
                    results = await execute_query(corrected)
                    memory_manager.force_collect_garbage()
                    final_result = format_results(results, question)
                    update_history(question, final_result)
                    return final_result
                except Exception as post_fix_err:
                    print(f"[AUTO-FIX] Corrected query still failed: {post_fix_err}")
        if not 'tried_regen_after_error' in locals():
            tried_regen_after_error = False
        if not tried_regen_after_error and 'does not exist' in error_msg.lower() and 'column' in error_msg.lower():
            tried_regen_after_error = True
            missing_cols = []
            for m in re.findall(r'column "([a-zA-Z0-9_\.]+)" does not exist', error_msg, flags=re.IGNORECASE):
                missing_cols.append(m)
            feedback = f"Previously generated SQL referenced non-existent columns: {', '.join(missing_cols)}. Remove or replace them using ONLY columns from the provided schemas."
            print(f"[REGEN_AFTER_ERROR] Feedback: {feedback}")
        if "relation" in error_msg.lower() and "does not exist" in error_msg.lower():
            match = re.search(r'relation "([^"]+)" does not exist', error_msg.lower())
            if match:
                bad_table = match.group(1)
                if bad_table.endswith('s') and bad_table[:-1] in available_tables:
                    fixed_query = sql_query.replace(bad_table, bad_table[:-1])
                    print(f"Fixing pluralization issue: {bad_table} -> {bad_table[:-1]}")
                    print(f"Retrying with fixed query: {fixed_query}")
                    try:
                        results = await execute_query(fixed_query)
                        final_result = format_results(results, question)
                    except Exception as retry_error:
                        print(f"Retry failed with error: {str(retry_error)}")
                        final_result = f"I encountered an error while querying the database: {str(retry_error)}\n\nPlease try rephrasing your question or ask about a different aspect of the data."
                else:
                    available_tables_str = ", ".join(available_tables)
                    final_result = f"I encountered an error with the database query. The table '{bad_table}' doesn't exist in the database. Available tables are: {available_tables_str}."
            else:
                final_result = f"I encountered an error while querying the database: {error_msg}"
        else:
            if "syntax error" in error_msg.lower():
                final_result = f"I encountered a syntax error in the SQL query. Please try rephrasing your question: {error_msg}"
            else:
                final_result = f"I encountered an error while querying the database: {error_msg}"

    # Final system cleanup and reporting (separate from Result Presenter Agent)
    print("\n--- Post-Query System Cleanup ---")

    if hasattr(memory_manager, 'perform_full_cleanup'):
        memory_manager.perform_full_cleanup()
    else:
        memory_manager.force_collect_garbage()
    stats = memory_manager.get_memory_stats()
    print(f"Memory after processing: {stats['available_mb']:.2f}MB available")
    update_history(question, final_result)
    return final_result

async def main():
    global llm
    # Initialize memory management
    print("\n--- Memory Management Initialization ---")
    # Force garbage collection and memory monitoring
    memory_manager.force_collect_garbage()
    
    # Check memory before starting
    print("\n--- System Resource Check ---")
    stats = memory_manager.get_memory_stats()
    print(f"Memory status: {stats['available_mb']:.2f}MB available of {stats['total_mb']:.2f}MB total")
    
    if stats['available_mb'] < 1000:
        print("⚠️ WARNING: System memory is low at startup. Application may experience issues.")
        print("Please close other applications to free up memory before proceeding.")
        print("At least 1GB of free memory is recommended for reliable operation.")
        # Continue anyway, but with a warning
    
    # Initialize database
    print("\n--- Database Initialization ---")
    db_metadata = await initialize_database(uri=DB_CONNECTION_URL)
    if not db_metadata["tables"]:
        print("Database initialization failed: No tables found")
        return
    
    print(f"Database initialized with tables: {available_tables}")
    
    # Print welcome message
    print("\nWelcome to the SQL Assistant! Type 'quit' to exit.")
    print("You can ask questions about the employee database, and I'll help you find the answers.")
    print("Example questions:")
    print("- What is the average salary for employees?")
    print("- How many employees work in each department?")
    print("- Who are the current department managers?")
    print("\nWhat would you like to know?")
    
    # Main loop
    consecutive_errors = 0
    max_consecutive_errors = 3
    
    while True:
        try:
            # Get user input
            user_question = input("\nYour question: ").strip()
            if user_question.lower() == 'quit':
                print("\nGoodbye! Have a great day!")
                break
            # Handle empty input
            if not user_question:
                print("Please enter a question or type 'quit' to exit.")
                continue
            # Reset consecutive errors on successful input
            consecutive_errors = 0
            # Run garbage collection before each query
            gc.collect()
        except Exception as e:
            print(f"Error in main loop: {e}")
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                print("Multiple consecutive errors detected. Performing memory cleanup...")
                gc.collect()
                consecutive_errors = 0
            # Skip this iteration on input/read error
            continue

        # Check if there's enough memory to safely process the request
        if not check_memory_safety():
            stats_now = memory_manager.get_memory_stats()
            # If memory is critically low, refuse to proceed
            if stats_now['available_mb'] < memory_manager.critical_mb:
                print("System is critically low on memory. Please close other applications and try again.")
                print("Cannot proceed due to critically low memory. Please close some applications and try again.")
                continue
            # Otherwise, operate in a degraded, aggressive memory mode instead of outright refusing
            print(f"⚠️ Low memory warning: Only {stats_now['available_mb']:.2f}MB available, need at least {memory_manager.threshold_mb:.2f}MB")
            print("Proceeding in degraded mode with aggressive resource reduction. Responses may be less detailed or slower.")
            # Try to create a reduced LLM and continue; if that fails, then abort
            try:
                gc.collect()
                reduced = initialize_llm(progressive_reduction=2)
                if reduced is not None:
                    llm = reduced
                    Settings.llm = reduced
                    print("Degraded LLM initialized and set as active model.")
                else:
                    print("Failed to initialize degraded LLM instance. Aborting request.")
                    continue
            except Exception as e:
                print(f"Failed to initialize degraded LLM due to error: {e}")
                print("Cannot proceed due to low memory. Please close other applications and try again.")
                continue

        # Process the query through the agentic workflow
        print(f"\n--- Processing Query: {user_question} ---")

        try:
            # Run the agentic workflow
            result = await agentic_query_process(user_question)

            # Display the result
            print(f"\nAnswer: {result}")

            # Run garbage collection after successful query
            gc.collect()

        except Exception as e:
            # Handle potential memory or timeout errors
            if "model requires more system memory" in str(e) or "memory" in str(e).lower() or "timed out" in str(e).lower() or "timeout" in str(e).lower():
                print(f"\n⚠️ Model issue detected: {str(e)}")

                # Try to free up memory
                gc.collect()

                # (No template fallback path.)
                print(f"Trying with smaller model ({FALLBACK_MODEL_NAME})...")

                # Create a new LLM with the smaller model and more aggressive memory settings
                try:
                    fallback_llm = Ollama(
                        model=FALLBACK_MODEL_NAME,
                        base_url=OLLAMA_BASE_URL,
                        temperature=0.1,
                        request_timeout=30.0,  # Shorter timeout for fallback model
                        additional_kwargs={
                            "num_ctx": 1024,      # Ultra-minimal context window
                            "num_batch": 1,      # Minimal batch size
                            "num_gpu": 0,        # Force CPU mode
                            "f16_kv": True,      # Use half-precision
                            "mirostat": 0,       # Disable mirostat sampling
                            "num_thread": 1,     # Minimal threads
                            "seed": 42,          # Consistent seed
                            "top_k": 20,
                            "top_p": 0.9,
                            "num_predict": 256
                        }
                    )

                    # Temporarily replace the LLM
                    original_llm = llm
                    llm = fallback_llm
                    Settings.llm = fallback_llm

                    # Try again with the smaller model
                    try:
                        result = await agentic_query_process(user_question)
                        print(f"\nAnswer (using fallback model): {result}")
                    except Exception as inner_fallback_error:
                        print(f"Fallback model also failed: {str(inner_fallback_error)}")
                        print("All query generation methods failed. Please try rephrasing your question.")
                except Exception as fallback_error:
                    print(f"Error with fallback model: {str(fallback_error)}")
                    print("All query generation methods failed. Please try rephrasing your question.")

        # If we've had too many consecutive errors, trigger a manual garbage collection
        if consecutive_errors >= max_consecutive_errors:
            print("Multiple consecutive errors detected. Performing memory cleanup...")
            gc.collect()
            consecutive_errors = 0
        

if __name__ == "__main__":
    asyncio.run(main())