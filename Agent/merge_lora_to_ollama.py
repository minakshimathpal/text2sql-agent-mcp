"""Merge a PEFT LoRA adapter (Gemma 1B) into base weights and prepare an Ollama model directory.

Usage (after installing extra deps in requirements.txt):
    python Agent/merge_lora_to_ollama.py \
        --base google/gemma-1.1-1b-it \
        --adapter-dir Agent/Gemma3_1B_weights \
        --out-dir merged_gemma1b_text2sql \
        --ollama-model-name gemma1b-text2sql

Then create the Ollama model:
    (In PowerShell)
    ollama create gemma1b-text2sql -f merged_gemma1b_text2sql/Modelfile

Finally set FINETUNED_MODEL_NAME=gemma1b-text2sql in your .env (or environment) and run the agent.

Notes:
- Ensure you used the SAME base checkpoint for finetuning as you pass via --base.
- Script keeps everything on CPU (device_map="cpu") to stay RAM friendly; you can add --load-in-8bit if you install bitsandbytes and have GPU.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi
try:  # Compatibility across hub versions
    from huggingface_hub.errors import RepositoryNotFoundError  # type: ignore
except Exception:  # Older versions
    class RepositoryNotFoundError(Exception):
        pass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def _detect_base_from_adapter(adapter_dir: str) -> str | None:
    p = Path(adapter_dir)
    resolved = p.resolve()
    print(f"[auto-base] Inspecting adapter_dir: {adapter_dir} -> {resolved}")
    if not resolved.exists():
        print(f"[auto-base] Directory does not exist: {resolved}")
        # Attempt smart fallbacks (common when user runs inside Agent/ already)
        candidates = []
        name = p.name
        cwd = Path.cwd()
        candidates.append(cwd / name)
        candidates.append(cwd.parent / name)
        candidates.append(cwd.parent / 'Agent' / name)
        for cand in candidates:
            if cand.exists():
                print(f"[auto-base] Found alternative adapter dir: {cand}")
                resolved = cand
                break
        if not resolved.exists():
            print(f"[auto-base] Fallback search failed. Tried: {[str(c) for c in candidates]}")
            return None
    cfg_path = resolved / "adapter_config.json"
    if not cfg_path.exists():
        try:
            contents = [c.name for c in resolved.iterdir()]
        except Exception:
            contents = []
        print(f"[auto-base] adapter_config.json not found in {resolved}. Contents: {contents}")
        return None
    try:
        raw = cfg_path.read_text(encoding="utf-8")
        print(f"[auto-base] Found adapter_config.json ({len(raw)} bytes)")
        data = json.loads(raw)
        base = data.get("base_model_name_or_path")
        print(f"[auto-base] base_model_name_or_path: {base}")
        return base
    except Exception as e:
        print(f"[auto-base] Failed reading adapter_config.json: {e}")
        return None


def merge_adapter(base: str, adapter_dir: str, out_dir: str, ollama_model_name: str, template_temperature: float = 0.05, hf_token: str | None = None, local_files_only: bool = False, auto_base: bool = False, trust_remote_code: bool = True):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if auto_base or base.lower() == "auto":
        detected = _detect_base_from_adapter(adapter_dir)
        if detected:
            print(f"[auto-base] Detected base model from adapter_config: {detected}")
            base = detected
        else:
            raise RuntimeError("Could not auto-detect base model; supply --base explicitly.")

    auth_kwargs = {}
    if hf_token:
        # Only pass 'token' (use_auth_token deprecated & causes conflict if both set)
        auth_kwargs = {"token": hf_token}
        os.environ.setdefault("HF_TOKEN", hf_token)
        os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", hf_token)
        print(f"Using HF token (masked): {hf_token[:8]}... (length {len(hf_token)})")

    # Quick repo existence check (skipped if local_files_only)
    if not local_files_only:
        try:
            HfApi().model_info(base, token=hf_token)
        except RepositoryNotFoundError:
            print(f"\n[ERROR] Base model repo '{base}' not found on Hugging Face.\n")
            print("Troubleshooting:")
            print("  - Verify the exact model id: e.g. 'google/gemma-2-2b-it' or 'google/gemma-1.1-2b-it'.")
            print("  - If it's a gated/private model ensure your token has accepted the terms.")
            print("  - Pass --hf-token YOUR_TOKEN or set HF_TOKEN env var.")
            return 2
        except Exception as e:
            print(f"[WARN] Could not verify model existence pre-download: {e}")

    print(f"Loading base model: {base} (local_files_only={local_files_only})")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=local_files_only, **auth_kwargs)
    except Exception as e:
        msg = str(e)
        if "403" in msg or "Forbidden" in msg:
            print("\n[AUTH ERROR] 403 while fetching tokenizer/config.")
            print("Checklist:")
            print("  1. Ensure token has: 'Read access to public gated models'.")
            print("  2. Visit model page and accept license (if not already).")
            print("  3. If using a fine-grained token, enable 'Gated repos' permission.")
            print("  4. Retry with: --hf-token $env:HF_TOKEN (PowerShell: $env:HF_TOKEN='xxxx')")
            print("  5. If model already cached on another machine, copy that ~/.cache/huggingface/hub directory.")
        raise

    # Keep on CPU to reduce VRAM needs; if you want faster and have GPU, remove device_map and add torch_dtype.
    model_kwargs = dict(
        device_map="cpu",
        local_files_only=local_files_only,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        **auth_kwargs,
    )
    if trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    # Lower RAM peak during load
    model_kwargs["low_cpu_mem_usage"] = True

    try:
        print("[stage] Downloading / loading base model weights...")
        base_model = AutoModelForCausalLM.from_pretrained(base, **model_kwargs)
        print("[stage] Base model loaded successfully.")
    except ValueError as ve:
        msg = str(ve)
        if "does not recognize this architecture" in msg or "model type" in msg:
            print("[model-load] Unrecognized model type. Attempting retry after upgrading transformers may be required.")
            print("Suggestion: pip install --upgrade transformers huggingface_hub --no-cache-dir")
            if not trust_remote_code:
                print("Retrying with trust_remote_code=True ...")
                model_kwargs["trust_remote_code"] = True
                base_model = AutoModelForCausalLM.from_pretrained(base, **model_kwargs)
            else:
                # Last resort: advise source install
                raise
        else:
            raise
    except Exception as e:
        msg = str(e)
        if "403" in msg or "Forbidden" in msg:
            print("\n[AUTH ERROR] 403 while fetching model weights (config or pytorch model). See guidance above.")
        raise

    print(f"Loading LoRA adapter from: {adapter_dir}")
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    print("[stage] LoRA adapter loaded.")

    print("Merging LoRA weights into base weights (this may take a minute)...")
    merged_model = peft_model.merge_and_unload()
    print("[stage] Merge complete in-memory.")

    print(f"Saving merged model to: {out_dir}")
    merged_model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("[stage] Saved model and tokenizer.")

    # Create a simple Modelfile for Ollama (avoid nested triple-quote confusion)
    modelfile_lines = [
        # Use current directory as model context so Modelfile works regardless of parent path
        "FROM .",
        f"PARAMETER temperature {template_temperature}",
        "# Basic template; adjust if you have a custom chat format.",
        'TEMPLATE """{{ .Prompt }}"""',
        ""
    ]
    modelfile = "\n".join(modelfile_lines)
    (out_path / "Modelfile").write_text(modelfile, encoding="utf-8")

    meta = {
        "base": base,
        "adapter_dir": adapter_dir,
        "merged_dir": out_dir,
        "ollama_model_name": ollama_model_name,
        "temperature": template_temperature,
    }
    (out_path / "merge_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Merge complete. Next steps:")
    print(f"  1. ollama create {ollama_model_name} -f {out_dir}/Modelfile")
    print(f"  2. Set FINETUNED_MODEL_NAME={ollama_model_name}")
    print("  3. Run your agent code again.")


def main():
    if len(sys.argv) == 1:
        print("\n[merge_lora_to_ollama] Missing arguments. Example usage:\n")
        print("python Agent/merge_lora_to_ollama.py \\")
        print("  --base google/gemma-1.1-1b-it \\")
        print("  --adapter-dir Agent/Gemma3_1B_weights \\")
        print("  --out-dir merged_gemma1b_text2sql \\")
        print("  --ollama-model-name gemma1b-text2sql\n")
        return 1

    parser = argparse.ArgumentParser(description="Merge a Gemma LoRA adapter for Ollama usage.")
    parser.add_argument("--base", required=True, help="Base HF model id used during finetuning (e.g., google/gemma-3-1b-it or 'auto' to read adapter_config)")
    parser.add_argument("--auto-base", action="store_true", help="Ignore --base value and read base from adapter_config.json")
    parser.add_argument("--adapter-dir", required=True, help="Path to the LoRA adapter directory (with adapter_model.safetensors)")
    parser.add_argument("--out-dir", required=True, help="Directory to write merged weights and Modelfile")
    parser.add_argument("--ollama-model-name", required=True, help="Target Ollama model name to create")
    parser.add_argument("--temperature", type=float, default=0.05, help="Default temperature for Modelfile")
    parser.add_argument("--hf-token", help="Hugging Face access token (optional; else read HF_TOKEN / HUGGINGFACEHUB_API_TOKEN env)")
    parser.add_argument("--local-files-only", action="store_true", help="Do not attempt to download; require weights to be cached locally")
    parser.add_argument("--no-trust-remote-code", action="store_true", help="Disable trust_remote_code (not recommended for Gemma 3)")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        print("[INFO] No HF token provided. If the model is gated/private, pass --hf-token or set HF_TOKEN env.")

    merge_adapter(
        base=args.base,
        adapter_dir=args.adapter_dir,
        out_dir=args.out_dir,
        ollama_model_name=args.ollama_model_name,
        template_temperature=args.temperature,
        hf_token=hf_token,
        local_files_only=args.local_files_only,
        auto_base=args.auto_base,
        trust_remote_code=not args.no_trust_remote_code,
    )


if __name__ == "__main__":
    main()
