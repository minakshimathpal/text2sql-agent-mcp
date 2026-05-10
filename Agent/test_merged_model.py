from safetensors.numpy import load_file
from pathlib import Path

base = Path(r"e:\Capstone_Project\text2sql-agent-mcp\merged_qwen2_5_3b_text2sql")
safetensor = base / "model.safetensors"

print("Loading safetensors (may use RAM)...")
d = load_file(str(safetensor))

weights = [k for k in d.keys() if k.endswith(".weight")]
print("weight keys count:", len(weights))

if weights:
    for k in weights[:200]:
        print(k, d[k].shape, d[k].dtype)
else:
    print("\nNo '.weight' keys found — showing first 120 keys:")
    for k in list(d.keys())[:120]:
        print(k)