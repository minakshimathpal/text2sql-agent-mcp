import os
try:
    # Ensure local .env values (OLLAMA_*) are loaded when this module is imported
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
import uuid
import re
import base64
from typing import Dict, Any, Optional

def _mock_result(received: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = {"doc_id": str(uuid.uuid4()), "text": "(mock OCR output)", "metadata": {"pages": 1, "confidence": 0.0}}
    if received:
        out["received_args"] = received
    return out


def clean_ocr_text(raw: Optional[str]) -> str:
    """Lightweight post-processing for OCR output to improve readability.

    - turns literal "\\n" sequences into real newlines
    - unescapes common unicode/HTML escapes when safe
    - collapses repeated spaces and blank lines
    - trims and merges lines that were broken mid-sentence
    """
    if not raw:
        return ""
    s = raw
    try:
        # Convert literal backslash-n to actual newline
        s = s.replace('\\n', '\n')
    except Exception:
        pass

    # Try to interpret common escape sequences (conservative)
    try:
        s_decoded = s.encode('utf-8', errors='surrogatepass').decode('unicode_escape')
        # Accept decoded result if it reduces backslash artifacts
        if '\\n' not in s_decoded:
            s = s_decoded
    except Exception:
        pass

    # Normalize line endings and whitespace
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    s = re.sub(r'[ \t]{2,}', ' ', s)
    # Collapse 3+ newlines into max 2
    s = re.sub(r'\n{3,}', '\n\n', s)

    # Trim whitespace on each line
    lines = [ln.strip() for ln in s.splitlines()]

    # Merge lines that likely were broken mid-sentence: if a line doesn't end with
    # sentence punctuation and the next line begins with a lowercase letter or digit,
    # join them with a space.
    merged = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if i < len(lines) - 1:
            nxt = lines[i + 1]
            if line and nxt and not re.search(r'[\.\!\?\:;]$', line) and re.match(r'^[a-z0-9]', nxt):
                line = line + ' ' + nxt
                i += 2
                # Continue merging subsequent similar lines
                while i < len(lines) and lines[i] and not re.search(r'[\.\!\?\:;]$', line) and re.match(r'^[a-z0-9]', lines[i]):
                    line = line + ' ' + lines[i]
                    i += 1
                merged.append(line)
                continue
        merged.append(line)
        i += 1

    s = '\n'.join([ln for ln in merged if ln])

    # Remove long runs of underscores or pipes
    s = re.sub(r'[_\|]{2,}', '', s)

    # Strip control chars except common whitespace
    s = ''.join(ch for ch in s if ord(ch) >= 9 and ord(ch) != 11 and ord(ch) != 12)

    try:
        import html

        s = html.unescape(s)
    except Exception:
        pass

    # Decode literal unicode escapes like \u0027 -> ' and \xA4 -> currency symbol
    try:
        def _u_decode(m):
            code = m.group(1)
            try:
                return chr(int(code, 16))
            except Exception:
                return m.group(0)

        s = re.sub(r'\\u([0-9A-Fa-f]{4})', _u_decode, s)
        s = re.sub(r'\\x([0-9A-Fa-f]{2})', lambda m: chr(int(m.group(1), 16)), s)
    except Exception:
        pass

    # Remove stray leading 'n' characters that appear before uppercase words (common OCR/newline artifact)
    try:
        s = re.sub(r'(?<=\s|^)[nN](?=[A-Z])', '', s)
    except Exception:
        pass

    # Ensure space after commas when missing
    try:
        s = re.sub(r',(?=[^\s])', ', ', s)
    except Exception:
        pass

    # Remove spaces before punctuation
    try:
        s = re.sub(r'\s+([,\.\:\;\%])', r'\1', s)
    except Exception:
        pass

    # Attempt to fix common mojibake (double-encoded UTF-8) conservatively.
    # Only apply if there are visible 'Ã' sequences suggesting encoding issues.
    try:
        if 'Ã' in s or 'Â' in s:
            try:
                candidate = s.encode('latin-1').decode('utf-8')
                # Accept candidate if it reduces suspicious sequences
                if candidate.count('Ã') < s.count('Ã'):
                    s = candidate
            except Exception:
                pass
    except Exception:
        pass

    return s.strip()


def process_image(image_url: Optional[str] = None, image_bytes: Optional[bytes] = None, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Process an image using local OCR first (pytesseract), with a light transformers fallback.

    Accepts either `image_url` or `image_bytes` (bytes or base64 string). Returns dict: {doc_id, text, metadata}
    """

    # Lightweight imports that may be missing in minimal environments
    try:
        from PIL import Image
        from io import BytesIO
        import requests
    except Exception:
        # Missing imaging/network deps -> return mock with echo of args
        return _mock_result(received={"image_url": image_url, "has_bytes": bool(image_bytes)})

    # Normalize image_bytes if base64 string was passed
    if isinstance(image_bytes, str):
        try:
            image_bytes = base64.b64decode(image_bytes)
        except Exception:
            image_bytes = None

    # Acquire PIL image
    img = None
    try:
        if image_bytes:
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
        elif image_url:
            resp = requests.get(image_url, timeout=15)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
        else:
            return _mock_result(received={"image_url": image_url, "has_bytes": False})
    except Exception:
        return _mock_result(received={"image_url": image_url, "has_bytes": bool(image_bytes)})

    # PRIORITY: Try local Tesseract (pytesseract) first — lightweight and deterministic on Windows

    try:
        import pytesseract
        # Robustly find tesseract executable: prefer PATH, then common Windows install locations
        try:
            import shutil
            found = shutil.which('tesseract')
        except Exception:
            found = None
        if not found:
            # common install locations on Windows
            for candidate in (r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
                try:
                    if os.path.exists(candidate):
                        found = candidate
                        break
                except Exception:
                    continue
        try:
            if found:
                pytesseract.pytesseract.tesseract_cmd = found
                print(f"[document_scanner] using tesseract executable: {found}")
            else:
                # leave default; pytesseract will raise a helpful error which we catch below
                print("[document_scanner] tesseract executable not found in PATH or common locations")
        except Exception:
            # ignore assignment errors and let pytesseract raise on use
            pass

        try:
            if img is not None:
                print("[document_scanner] attempting PRIMARY pytesseract OCR")
                ttxt = pytesseract.image_to_string(img)
                if ttxt and ttxt.strip():
                    print(f"[document_scanner] pytesseract returned text (len={len(ttxt.strip())})")
                    return {"doc_id": str(uuid.uuid4()), "text": clean_ocr_text(ttxt), "metadata": {"pages": 1, "confidence": None}}
                else:
                    print("[document_scanner] pytesseract returned no text; falling through to model-based fallbacks")
        except Exception as e_pyt_first:
            print(f"[document_scanner] primary pytesseract attempt failed: {e_pyt_first}")
    except Exception:
        # pytesseract not installed in environment — continue to available backends
        pass

    # Last resort: return a safe mock that echoes inputs
    # (Tesseract failed or is not installed — Q&A will still work via Granite Vision if image is saved)
    return _mock_result(received={"image_url": image_url, "has_bytes": bool(image_bytes)})
