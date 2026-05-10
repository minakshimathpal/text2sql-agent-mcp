Ollama GPU compatibility notes

Short summary
- Ollama can run models on GPU if the Ollama daemon/binary you installed includes
  CUDA/GPU support and the host has proper NVIDIA drivers installed.
- The merge step (creating a merged model directory) is independent of Ollama; once
  you have `merged_dir` and `Modelfile`, you run `ollama create` which registers the
  model for the Ollama runtime.

Recommendations to use GPU with Ollama
1. Ensure NVIDIA drivers + CUDA are installed and compatible with your GPU.
2. Install an Ollama build that supports GPU. Check `ollama --help` or the Ollama
   release notes for GPU options. Some Ollama builds automatically use CUDA if
   present.
3. After `ollama create <model> -f <Modelfile>`, use `ollama run` or your client
   and look for flags such as `--gpu` or `--device` (CLI flags vary by Ollama
   version). If unclear, run `ollama --help`.
4. If you run Ollama inside Docker, make sure to use the NVIDIA runtime (e.g.
   `--gpus all`) and a GPU-enabled Ollama image.

Practical tips
- Test on a small prompt first to confirm the model runs on GPU.
- If Ollama appears to fall back to CPU, check daemon logs and environment
  variables; some installs require an explicit `--use-gpu` or similar flag.
- If you plan to host the Ollama runtime on a separate GPU server, you can keep
  the merge step on your local machine and copy the `merged_dir` to the server
  before running `ollama create` there.

If you'd like, I can:
- Add a Dockerfile example for an NVIDIA-enabled Ollama runtime.
- Add an alternate merge script that uses 8-bit (bitsandbytes) if you have a GPU.
