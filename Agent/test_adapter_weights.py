
# Set environment variables BEFORE importing torch
import os
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
# Optional: force a specific GPU (uncomment if needed)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# In-code disables for torch.compile and inductor/triton
torch._dynamo.config.suppress_errors = True
#torch._dynamo.config.enable = False
torch.set_float32_matmul_precision('high')

#base = "google/gemma-3-1b-it"
#adapter = "Agent/Gemma3_1B_weights"
base = "Qwen/Qwen2.5-3B-Instruct"
adapter = "Agent/Qwen2.5_3b_weights"
model = AutoModelForCausalLM.from_pretrained(
    base,
    dtype="auto",
    device_map=None,
    load_in_8bit=True,
    low_cpu_mem_usage=True,
    local_files_only=False  # Set to False to force download if needed
    #trust_remote_code=True
)
model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
model = model.to("cuda")  # Move model to GPU
tokenizer = AutoTokenizer.from_pretrained(base)

prompt = "Schema: employee(emp_no, birth_date, first_name, last_name, gender, hire_date),\n" \
         "department(dept_no, dept_name),\n" \
         "dept_emp(emp_no, dept_no, from_date, to_date),\n" \
         "salary(emp_no, salary, from_date, to_date),\n" \
         "Write PostgreSQL query for highest and lowest salary across all departments. Do not include any explanation."
inputs = tokenizer(prompt, return_tensors="pt")
# Move all input tensors to CUDA
inputs = {k: v.to("cuda") for k, v in inputs.items()}

output = model.generate(**inputs, max_new_tokens=1028)
print(tokenizer.decode(output[0], skip_special_tokens=True))