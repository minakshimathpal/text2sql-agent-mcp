Merge Qwen LoRA adapter into an Ollama-compatible model

This repository contains a helper script `Agent/merge_qwen_to_ollama.py` to merge
a PEFT/LoRA adapter into a base Qwen model and produce a directory suitable for
`ollama create`.

Quick example (PowerShell):

```powershell
# Merge adapter (example)
python Agent/merge_qwen_to_ollama.py \
  --base qwen2.5:3b-instruct \
  --adapter-dir Agent/Qwen2.3_3b_weights \
  --out-dir merged_qwen2_5_3b_text2sql \
  --ollama-model-name qwen2.5-text2sql

# Create Ollama model (after the merge completes)
ollama create qwen2.5-text2sql -f merged_qwen2_5_3b_text2sql/Modelfile

# Set your environment so your agent picks up the finetuned model
$env:FINETUNED_MODEL_NAME = 'qwen2.5-text2sql'

# Run the agent as you normally would
python run.py
```

Notes and tips
- The script keeps merging on CPU to avoid GPU/VRAM OOM. If you have a GPU and want
  to use it for faster tokenization or if your environment supports 8-bit loading,
  consider installing `bitsandbytes` and passing related flags in a custom local
  edit of the script (the default aims to be safe on modest machines).
- If the base model is gated on HF, provide `--hf-token` or set `HF_TOKEN` env var.
- The resulting directory contains a simple `Modelfile` and `merge_metadata.json`.

If you want, I can also provide an alternate script that attempts an 8-bit merge/load
or a Dockerfile to produce an environment for merging on a larger machine.
