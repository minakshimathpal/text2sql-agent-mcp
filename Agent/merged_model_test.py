import json
from safetensors.numpy import load_file
from pathlib import Path

base = Path(r"e:\Capstone_Project\text2sql-agent-mcp\merged_qwen2_5_3b_text2sql")
safetensor = base / "model.safetensors"
cfg_path = base / "config.json"

cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
hidden = cfg.get("hidden_size")
print("expected attn_q shape (hidden, hidden) =", hidden, hidden)

print("Loading safetensors (this may use RAM)...")
tensors = load_file(str(safetensor))

# find candidate keys
candidates = [k for k in tensors.keys() if "attn_q.weight" in k or "blk.0.attn_q.weight" in k]
if not candidates:
    # also show some top-level keys to help debug
    print("No attn_q keys found. Showing first 40 keys in the safetensor:")
    for k in list(tensors.keys())[:40]:
        print("  ", k)
else:
    print("Found attn_q keys (showing up to 30):")
    for k in candidates[:30]:
        print(k, "->", tensors[k].shape, "dtype:", tensors[k].dtype)