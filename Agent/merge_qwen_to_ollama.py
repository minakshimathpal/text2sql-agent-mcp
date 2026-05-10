
"""Merge a PEFT LoRA adapter (Qwen 2.5 3b) into base weights and prepare an Ollama model directory.

This is a standalone, non-destructive helper. It follows the same pattern as
`Agent/merge_lora_to_ollama.py` but defaults to a Qwen base and includes flags
for optional low-memory / 8-bit loading.

Usage (PowerShell):
   Agent/merge_qwen_to_ollama.py \
    --base Qwenpython/Qwen2.5-3B-Instruct \
    --adapter-dir Agent/Qwen2.5_3b_weights \
    --out-dir merged_qwen2_5_3b_text2sql \
    --ollama-model-name qwen2.5-text2sql

Notes:
- Keeps merge on CPU to minimize peak memory.
- If you have a GPU and want faster tokenizers/model ops, see OLLAMA GPU notes file.
"""
import argparse
import json
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi
except Exception:
    HfApi = None

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformers import BitsAndBytesConfig

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

def merge_adapter(base: str, adapter_dir: str, out_dir: str, ollama_model_name: str, template_temperature: float = 0.05, hf_token: str | None = None, local_files_only: bool = False, trust_remote_code: bool = True, auto_base: bool = False):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # If requested, detect base from adapter_config.json
    if auto_base or (isinstance(base, str) and base.lower() == "auto"):
        detected = _detect_base_from_adapter(adapter_dir)
        if detected:
            print(f"[auto-base] Detected base model from adapter_config: {detected}")
            base = detected
        else:
            raise RuntimeError("Could not auto-detect base model; supply --base explicitly.")

    # Load the tokenizer first, it's small and fast
    try:
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=local_files_only)
    except Exception as e:
        print(f"Failed loading tokenizer for base '{base}': {e}")
        raise

    # Configure 4-bit quantization with the necessary offload flag
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True, # THIS IS CRUCIAL
    )

    print(f"Loading base model: {base} (using 4-bit quantization with auto device mapping) 🧠💡")

    try:
        # Use the BitsAndBytesConfig and let device_map="auto" handle the offloading
        base_model = AutoModelForCausalLM.from_pretrained(
            base,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
    except Exception as e:
        print(f"Failed loading base model '{base}': {e}")
        raise

    print(f"Loading LoRA adapter from: {adapter_dir}")
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    print("LoRA adapter loaded.")

    print("Merging LoRA weights into base weights (in-memory)...")
    merged_model = peft_model.merge_and_unload()
    print("Merge complete.")

    print(f"Saving merged model to: {out_dir}")
    merged_model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    modelfile_lines = [
        "FROM .",
        f"PARAMETER temperature {template_temperature}",
        'TEMPLATE """{{ .Prompt }}"""',
        "",
    ]
    (out_path / "Modelfile").write_text("\n".join(modelfile_lines), encoding="utf-8")

    meta = {
        "base": base,
        "adapter_dir": adapter_dir,
        "merged_dir": out_dir,
        "ollama_model_name": ollama_model_name,
        "temperature": template_temperature,
    }
    (out_path / "merge_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Merge complete. Next steps:")
    print(f" 1. ollama create {ollama_model_name} -f {out_dir}/Modelfile")
    print(f" 2. Set FINETUNED_MODEL_NAME={ollama_model_name} in your environment")
    print(" 3. Run your agent code again.")

def main():
    parser = argparse.ArgumentParser(description="Merge a Qwen LoRA adapter for Ollama usage.")
    parser.add_argument("--base", required=True, help="Base HF model id used during finetuning (default: qwen2.5:3b-instruct)")
    parser.add_argument("--auto-base", action="store_true", help="Ignore --base value and read base from adapter_config.json")
    parser.add_argument("--adapter-dir", required=True, help="Path to the LoRA adapter directory (with adapter_model.safetensors)")
    parser.add_argument("--out-dir", required=True, help="Directory to write merged weights and Modelfile")
    parser.add_argument("--ollama-model-name", required=True, help="Target Ollama model name to create")
    parser.add_argument("--temperature", type=float, default=0.05, help="Default temperature for Modelfile")
    parser.add_argument("--hf-token", help="Hugging Face access token (optional)")
    parser.add_argument("--local-files-only", action="store_true", help="Do not attempt to download; require weights to be cached locally")
    parser.add_argument("--no-trust-remote-code", action="store_true", help="Disable trust_remote_code (not recommended for some models)")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    merge_adapter(
        base=args.base,
        adapter_dir=args.adapter_dir,
        out_dir=args.out_dir,
        ollama_model_name=args.ollama_model_name,
        template_temperature=args.temperature,
        hf_token=hf_token,
        local_files_only=args.local_files_only,
        trust_remote_code=not args.no_trust_remote_code,
    )

if __name__ == "__main__":
    main()