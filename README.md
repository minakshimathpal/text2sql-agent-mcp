# Text-to-SQL and Document VQA Multi-Agent System

![Architecture](https://img.shields.io/badge/Architecture-Multi--Agent-blue)
![Database](https://img.shields.io/badge/Database-Agnostic-success)
![LLM](https://img.shields.io/badge/LLMs-Qwen%20%7C%20Ollama-orange)
![Protocol](https://img.shields.io/badge/Protocol-MCP-purple)

An advanced, hardware-optimized AI pipeline built on a multi-agent orchestration architecture. This system seamlessly integrates natural language Text-to-SQL generation with a Vision-Language Model (VLM) for Optical Character Recognition (OCR) and Visual Question Answering (VQA).

By leveraging the Model Context Protocol (MCP), local LLMs via Ollama, and intelligent memory management, this project achieves production-grade data extraction and database querying running entirely on local hardware within tight memory constraints.

---

## Key Features

- **Multi-Agent Orchestration:** A dynamic routing system featuring specialized agents — MemoryAgent, QueryCraftAgent, and ResultPresenterAgent — each with a clearly defined role in the pipeline.
- **Vision-Language Question Answering:** Integrates Qwen 3.5 2B Vision for accurate data extraction from complex documents such as invoices and unstructured tables, served via Ollama.
- **Three-Level Text-to-SQL Engine:** Translates natural language questions into syntactically valid SQL through a progressive reasoning architecture, with automatic self-correction at each escalation level.
- **Semantic Gatekeeping:** A multi-layer validation pipeline filters hallucinated SQL, undefined aliases, missing joins, and incorrect aggregation patterns before any query reaches the database.
- **Dynamic VRAM Orchestration:** Custom memory management algorithms automatically load and unload models from GPU VRAM to prevent out-of-memory crashes, enabling concurrent AI workloads on consumer hardware.
- **Unified MCP Integration:** All external data sources — relational databases and vision models alike — are accessed through a single, uniform Model Context Protocol interface.
- **Database Agnostic Design:** Architected to interface with PostgreSQL, MySQL, SQLite, MS SQL Server, and Oracle via SQLAlchemy.
- **Conversational Interface:** A full Web UI supporting natural language interaction with persistent session context.

---

## System Architecture and Agentic Workflow

The project employs a decoupled, asynchronous microservices architecture. The Web UI communicates with a backend worker, which orchestrates all agent requests through the Model Context Protocol.

### The Three-Agent Pipeline

**1. MemoryAgent**

The MemoryAgent acts as the entry point for all user queries. It performs three functions before any SQL generation occurs: checking whether the current query can be satisfied from historical conversation context, rephrasing ambiguous or under-specified queries based on prior interactions, and routing validated new queries to the appropriate downstream agent. This layer prevents redundant computation and improves response latency for repeated or related questions.

**2. QueryCraftAgent — The Three-Level Text-to-SQL Engine**

The QueryCraftAgent is the core reasoning engine of the system. It implements a progressive escalation architecture that balances inference speed with accuracy:

- **Level 1 — Pure LLM Intelligence:** Zero-shot semantic translation of natural language to SQL using Qwen 2.5 Coder. The generated query passes through a semantic validation pipeline that checks alias correctness, required table joins, aggregation integrity, and department or entity filters before being accepted.

- **Level 2 — LLM with SQL Fixer:** When Level 1 fails validation, the agent enters an autonomous self-correction loop. The SQL Fixer applies a deterministic set of targeted transformations — normalizing aliases, injecting missing to_date filters, repairing phantom column references, and enforcing PostgreSQL-compliant GROUP BY clauses — before re-validating the output.

- **Level 3 — LLM with MCP Schema-Grounded SQL Generation:** The deepest reasoning level, reserved for queries that fail both prior levels. The agent calls one of fourteen registered `fast_sql` MCP tools, each of which queries the live database schema via SQLAlchemy's inspector to retrieve actual column names and table structures, then programmatically constructs a valid SQL template grounded in the real schema. This template is passed as a verified hint to the LLM, which refines it before the Fixer loop runs. Because the hint is built from the actual schema rather than the LLM's training data, this level eliminates the class of failures caused by hallucinated column names or missing joins.

**3. ResultPresenterAgent**

The ResultPresenterAgent executes the validated SQL query against the target database, translates raw result sets into natural conversational responses, handles execution errors and edge cases, and persists the interaction to the conversation history for future MemoryAgent lookups.

### Multi-Agent Orchestration Flow

```mermaid
graph TD
    User([User]) --> Web["Web Application\nPort 8000"]

    Web -->|Image Upload| Worker["Backend Worker\nPort 8700"]
    Web -->|Text Query| Worker

    subgraph MCP["MCP Tool Server — Port 8701"]
        DocScan["document_scanner.process\nTesseract OCR"]
        FastSQL["fast_sql tools\nSchema-Grounded SQL Generation"]
        VisionQA["granite_vision.qa\nQwen 3.5 2B Vision"]
    end

    Worker -->|"Upload: MCP tool call"| DocScan
    DocScan -->|"doc_id + OCR text + image path"| Store[("Document Store\nchat_store/docs/")]

    Worker -->|"OCR Q&A mode\ndoc_id present"| VisionQA
    Store -->|image path| VisionQA
    VisionQA --> OllamaV(["Ollama\nQwen 3.5 2B Vision"])
    OllamaV --> Presenter

    Worker -->|Text Query| Memory["MemoryAgent"]
    Memory -->|Cached answer| Presenter["ResultPresenterAgent"]
    Memory -->|New query| QCAgent["QueryCraftAgent"]

    QCAgent -->|"Level 1 and 2\nDirect Ollama call"| OllamaSQL(["Ollama\nQwen 2.5 Coder"])
    QCAgent -->|"Level 3\nMCP tool call"| FastSQL
    FastSQL -->|"Schema-validated SQL hint"| OllamaSQL
    OllamaSQL -->|Raw SQL| Fixer["SQL Fixer +\nSemantic Validator"]
    Fixer -->|"Validated SQL\nvia SQLAlchemy"| DB[("PostgreSQL / MySQL")]

    DB --> Presenter
    Presenter -->|Persist to history| Memory
    Presenter --> Answer(["Natural Language Answer"])
```

---

## Infrastructure and Deployment Architecture

### Containerized Microservices

The system is implemented as a Dockerized microservices architecture, with each component running as an isolated, independently deployable service. Docker Compose orchestrates three services: the main application and agent worker, the MCP tool server, and the relational databases (PostgreSQL and MySQL). Database containers are provisioned with health checks using `pg_isready` and `mysqladmin ping`, and the application container is configured with `condition: service_healthy` dependencies to eliminate the startup race condition inherent in naive `depends_on` configurations.

### Native-Bridge Hybrid Offloading

For peak inference performance, the system uses a hybrid offloading strategy. The application logic and MCP servers run inside Docker containers for portability and isolation, while Ollama and all GPU inference are offloaded to the native Windows host and accessed by the containers over the WSL2 host bridge network (`OLLAMA_BASE_URL=http://<wsl2-host-ip>:11434`).

This architecture eliminates the virtualization overhead introduced by Docker Desktop's Linux VM layer. By giving the inference engine direct access to NVIDIA CUDA cores and the full system memory pool — rather than a virtualized slice — the 1.5B parameter model achieves response times measured in seconds rather than minutes. The separation also means that GPU memory pressure from Ollama does not compete with the container's allocated memory ceiling.

### Database Health and Startup Reliability

A production-grade startup sequence is enforced through Docker health checks and application-level retry logic. The agent application implements a ten-attempt retry loop with five-second intervals on initialization, including a `SELECT 1` connection verification before the schema is read. This ensures consistent behavior across restarts and eliminates the class of intermittent failures caused by the application reading an empty or partially initialized schema during database startup.

### MCP Tool Server Architecture

The Model Context Protocol server runs as a dedicated microservice on port 8701. Three categories of tools are registered at startup:

**document_scanner.process** handles Tesseract OCR and is called at image upload time, not at query time. It returns a doc_id, the extracted text, and triggers persistence of both the OCR output and the image path to the document store.

**granite_vision.qa** is the direct visual question-answering tool backed by Qwen 3.5 2B Vision via Ollama. It is called when a user submits a question with a doc_id and the original image file is available on disk. The internal tool identifier `granite_vision.qa` was retained from the original design specification; the underlying model was later replaced with Qwen 3.5 2B Vision for improved OCR accuracy and reduced VRAM usage, with no changes required to the agent orchestration code.

**fast_sql tools** are fourteen schema-grounded SQL generation tools used exclusively at Level 3 of the QueryCraftAgent. Each tool queries the live database schema via SQLAlchemy's inspector to retrieve actual column names, then uses a deterministic `_build_sql()` function to construct a valid query, optionally verified by a short LLM call. The result is returned as a schema-validated SQL hint to the QueryCraftAgent, which passes it to the LLM for final refinement. This design means Level 3 failures are caused by query complexity or intent ambiguity, never by hallucinated schema details.

---

## Unified Multimodal Integration via MCP

A defining architectural feature of this system is that OCR and multimodal reasoning are integrated through the same Model Context Protocol used for database access. The Qwen 3.5 2B Vision model is registered as a named MCP tool: `granite_vision.qa`. This internal tool identifier was retained from the original design specification; the underlying engine was subsequently replaced with Qwen 3.5 2B Vision, which demonstrated superior OCR accuracy and a smaller VRAM footprint. Because the tool interface is decoupled from its implementation, this substitution required no changes to the agent orchestration code — only the MCP tool registration was updated.

This design delivers three concrete engineering properties:

**Uniform Protocol:** The orchestrator issues identical-format tool calls regardless of whether the data source is a SQL table or a visual document. There are no special-case code paths for different modalities.

**Modular Replaceability:** The vision model is encapsulated behind its MCP interface. Upgrading to a different vision model — such as a GPT-4V endpoint — requires updating only the tool registration, not any agent logic.

**Cross-Modal Reasoning Capability:** Because both data sources share the same protocol, the agent can compose multi-step reasoning across modalities. For example, the system can extract a value from a document via the vision tool and immediately verify it against a database record via the SQL tool, within a single agentic workflow.

```text
Multimodal Request Flow

Worker receives query with image path
        |
        +-- Image detected --> ocr_agent_qa()
                                    |
                                    +-- Unload Qwen 2.5 Coder (free VRAM)
                                    |
                                    +-- MCP tool call: granite_vision.qa
                                                |
                                            MCP Server (port 8701)
                                                |
                                            Qwen 3.5 2B Vision (Ollama)
                                                |
                                            Structured answer returned
        |
        +-- No image --> Standard Text-to-SQL pipeline (unchanged)
```

---

## SQL Validation Pipeline

The semantic validation layer is one of the most technically substantive components of the system. Rather than trusting LLM output directly, every generated SQL query passes through a deterministic validation pipeline before execution. Key checks include:

- **Alias Integrity:** All table aliases used in SELECT, WHERE, JOIN, and GROUP BY clauses must be defined in FROM or JOIN declarations. The validator maintains a set of defined and used aliases and rejects queries with undefined references.
- **Salary Table Recognition:** The `salary` table name was explicitly excluded from the SQL keyword blocklist, as it is both a legitimate table name and a column name in the employee schema. Incorrect inclusion in the keyword set caused the validator to silently discard valid salary table joins.
- **Department Filter Enforcement:** Queries that reference a specific department by name are validated to confirm that a corresponding WHERE filter exists in the generated SQL. This prevents queries that return all departments when the user asked for one.
- **PostgreSQL GROUP BY Compliance:** Queries containing aggregate functions are checked to ensure all non-aggregated SELECT columns appear in the GROUP BY clause, enforcing strict PostgreSQL semantics.
- **Manager Schema Awareness:** Queries involving manager data are validated to confirm use of the `dept_manager` table rather than `dept_emp`. The validation produces a warning rather than a hard rejection, allowing the downstream semantic adjuster to regenerate the correct SQL before the query reaches the database.

---

## Research Methodology: Fine-Tuning and Model Selection

As part of the project's scientific evaluation, a custom model fine-tuning pipeline was developed and benchmarked against the selected production model.

**LoRA Fine-Tuning:** The base Qwen 2.5 3B model was fine-tuned using Low-Rank Adaptation on a specialized Text-to-SQL dataset.

**Weight Merging:** The LoRA adapters were merged back into the base model weights via `merge_qwen_to_ollama.py` and quantized for local deployment via Ollama.

**Comparative Evaluation:** The fine-tuned general model was benchmarked against the zero-shot `qwen2.5-coder:1.5b`, which is pre-trained on a large corpus of code and SQL.

**Engineering Decision:** The benchmark results showed that the Coder model's zero-shot SQL accuracy exceeded that of the general LoRA fine-tune, with faster inference and lower error rates on complex multi-table joins. The production architecture was updated accordingly. This outcome demonstrates a data-driven engineering decision: fine-tuning is not always the optimal path when a domain-specialized pretrained model exists for the target task.

See `Fine_Tuning_Qwen_2_5_3B.ipynb` and the `merged_qwen2_5_3b_text2sql` directory for the complete fine-tuning methodology and benchmark results.

---

## Technology Stack

| Component | Technology |
|---|---|
| Core Framework | Python 3.11+, FastAPI, Uvicorn |
| AI Orchestration | LlamaIndex, Model Context Protocol (MCP) |
| Local Inference | Ollama (C++ optimized, memory-mapped) |
| Text-to-SQL Model | qwen2.5-coder:1.5b |
| Vision/OCR Model | qwen3.5:2b (registered as granite_vision.qa) |
| Database ORM | SQLAlchemy |
| Containerization | Docker, Docker Compose |
| Databases | PostgreSQL, MySQL |

---

## Installation and Setup

### Prerequisites

- Ollama installed and running on the host machine
- Python 3.11 or higher
- Docker and Docker Compose
- A supported database (PostgreSQL or MySQL)

### 1. Pull Required Models

```bash
ollama pull qwen2.5-coder:1.5b
ollama pull qwen3.5:2b
```

### 2. Configure Environment

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the `Agent/` directory:

```env
DB_CONNECTION_URL=postgresql://user:password@localhost:5432/dbname
OLLAMA_MODEL_NAME=qwen2.5-coder:1.5b
GRANITE_MODEL_PATH=qwen3.5:2b
```

### 3. Run Without Docker

Start each microservice in a separate terminal:

```bash
# Terminal 1 — Backend Worker
python -u -m uvicorn "Agent.worker:app" --port 8700

# Terminal 2 — MCP Tool Server
python -m uvicorn "Agent.mcp_server:app" --port 8701

# Terminal 3 — Web UI
python -m uvicorn "web.app:app" --port 8000
```

Navigate to `http://localhost:8000` to interact with the system.

### 4. Run With Docker

Update `OLLAMA_BASE_URL` in `docker-compose.yml` to point to your host machine's Ollama instance, then run:

```bash
docker-compose up --build agent-app
```

The database containers include health checks and will be fully initialized before the application starts.

---

## Deployment on HuggingFace Spaces

This section documents the **complete deployment journey** — from the first proof-of-concept tunnel approach to the final production-ready hybrid architecture. Each phase is explained with commands and rationale so this can serve as a reference for deploying any multi-modal AI application on HuggingFace Spaces.

---

### Understanding HuggingFace Spaces Constraints

Before choosing a deployment strategy, it is essential to understand what the free tier of HuggingFace Spaces provides and restricts:

| Resource | Free Tier |
|----------|-----------|
| CPU | 2 vCPUs |
| RAM | 16 GB |
| GPU | None |
| Storage | Ephemeral (wiped on restart) |
| Public port | 7860 only |
| Timeout | Space sleeps after inactivity |

The key challenge: **running a local LLM (e.g. `qwen2.5-coder:1.5b`) via Ollama consumes ~1.5–3 GB RAM, leaving limited headroom for vision models and the application stack.** Running a vision-language model such as `qwen3.5:2b` (~3 GB) alongside the text model causes out-of-memory (OOM) failures on the free CPU tier.

---

### Phase 1 — Cloudflare Tunnel: Bridging Local Resources to the Cloud

#### Theory

The simplest approach to deploy a resource-intensive app on HF Spaces is to **not run the heavy models in the cloud at all**. Instead, you run the LLM and VLM locally on your development machine and expose the Ollama API endpoint to the internet via a secure tunnel. The HF Space container merely hosts the web frontend and proxies inference requests back to your local machine.

This approach has zero cloud compute cost and allows full GPU acceleration locally. However, it requires your development machine to remain **online and running at all times**, making it unsuitable for a persistent public demo.

#### Setup

**Step 1: Install Cloudflared on your local machine**
```bash
# Windows (via winget)
winget install Cloudflare.cloudflared

# Or download directly
# https://github.com/cloudflare/cloudflared/releases
```

**Step 2: Start Ollama locally**
```bash
ollama serve
# Ollama now listens on http://localhost:11434
```

**Step 3: Create a public tunnel to your local Ollama port**
```bash
cloudflared tunnel --url http://localhost:11434
```

Cloudflared will print a public HTTPS URL like:
```
https://your-random-id.trycloudflare.com
```

**Step 4: Set the tunnel URL as a secret in your HF Space**

In the HF Space settings under **Variables and Secrets**, add:
```
OLLAMA_BASE_URL = https://your-random-id.trycloudflare.com
```

The application code reads `OLLAMA_BASE_URL` and routes all inference calls through the tunnel to your local GPU.

#### Limitation

- The tunnel URL changes every time `cloudflared` restarts, requiring a secret update.
- Requires a continuously running local machine.
- Not viable for a shared capstone demo or public deployment.

---

### Phase 2 — Self-Contained Docker: Running Ollama Inside the Container

#### Theory

The next evolution is to **embed Ollama directly into the Docker image** and pull models at container startup. This makes the deployment fully self-contained — no external machine is needed. The HF Space container becomes a complete inference server.

The challenge is that HF Spaces rebuilds the Docker image on every push but does **not cache downloaded model weights** between restarts (storage is ephemeral). This means every **Factory Reboot** triggers a fresh ~1–3 GB model download, causing a cold-start delay of 2–5 minutes.

To prevent HF's health monitor from timing out the Space during this download, model pulling is performed **asynchronously in the background**, allowing the web UI to become responsive immediately while the model fetches in parallel.

#### Dockerfile Changes

Add Ollama installation to the `Dockerfile`:

```dockerfile
# Install Ollama binary (pinned to a stable release)
RUN curl -L https://github.com/ollama/ollama/releases/download/v0.5.11/ollama-linux-amd64.tgz \
    -o ollama.tgz && \
    tar -C /usr -xzf ollama.tgz && \
    rm ollama.tgz
```

#### Startup Script (`start_hf.sh`)

```bash
#!/bin/bash
export HOME=/root

echo "[HF-INIT] Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to become healthy (up to 20 attempts)
for i in $(seq 1 20); do
    if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "[HF-INIT] Ollama is ALIVE!"
        break
    fi
    echo "[HF-INIT] Waiting... ($i/20)"
    sleep 3
done

# Pull the text model asynchronously (non-blocking)
echo "[HF-INIT] Pulling qwen2.5-coder:1.5b in background..."
ollama pull qwen2.5-coder:1.5b &

# Start application services immediately (do not wait for model pull)
echo "[HF-INIT] Starting Backend Worker..."
python -m uvicorn Agent.worker:app --host 0.0.0.0 --port 8700 &

echo "[HF-INIT] Starting MCP Tool Server..."
python -m uvicorn Agent.mcp_server:app --host 0.0.0.0 --port 8701 &

echo "[HF-INIT] Starting Web UI on port 7860..."
exec python -m uvicorn web.app:app --host 0.0.0.0 --port 7860
```

The `&` (ampersand) operator sends the model pull to the background. The `exec` on the last line replaces the shell process with the web server, ensuring HF's process monitor tracks the correct PID.

#### Limitation

- **Vision model OOM**: Adding `qwen3.5:2b` alongside `qwen2.5-coder:1.5b` exceeds the 16 GB RAM limit when both are loaded simultaneously.
- Model weights are re-downloaded on every Factory Reboot.
- Ollama version pinning is critical — newer models require newer Ollama binaries and pulling the wrong combination causes `412` errors.

---

### Phase 3 — Hybrid Serverless API (Final Production Architecture)

#### Theory

The final and most robust architecture eliminates the in-container LLM entirely for text generation and uses **external serverless inference APIs** instead. The container only runs lightweight services (web UI, API proxy, Tesseract OCR), while all neural inference is delegated to dedicated providers:

| Task | Provider | Model | Cost |
|------|----------|-------|------|
| Text-to-SQL | HuggingFace Inference API | `Qwen/Qwen2.5-Coder-7B-Instruct` | Free tier |
| Vision Q&A | Groq API | `meta-llama/llama-4-scout-17b-16e-instruct` | Free (14,400 req/day) |
| OCR | Tesseract (in container) | — | Free |
| Database | Neon PostgreSQL | — | Free tier |

This eliminates OOM risk entirely, removes the model download cold-start, and makes the container startup under 30 seconds.

#### Why Groq for Vision Instead of HuggingFace Serverless?

HuggingFace's serverless inference for vision-language models has two limitations on the free tier:
1. Most VLM models are not routed through any enabled provider for free-tier accounts.
2. Google Gemini's free tier blocks requests originating from cloud datacenter IP ranges (such as HF's infrastructure), setting the effective quota to zero even for valid free-tier API keys.

**Groq** is purpose-built for API-first server deployments and explicitly allows free-tier requests from cloud IPs. It provides 14,400 free requests per day with no IP restrictions.

#### Required HuggingFace Space Secrets

Set these in **Space Settings → Variables and Secrets**:

```
USE_HF_CLOUD       = 1
HF_API_TOKEN       = # HuggingFace access token
GROQ_API_KEY       = # From console.groq.com (free)
DB_CONNECTION_URL  = # Neon PostgreSQL connection string
```

#### Obtaining API Keys

**HuggingFace Token:**
1. Go to https://huggingface.co/settings/tokens
2. Create a token with `read` scope (or `write` if you need to push models)

**Groq API Key:**
1. Go to https://console.groq.com
2. Sign in with Google / GitHub
3. Navigate to **API Keys** → **Create API Key**
4. Copy the key

**Neon PostgreSQL:**
1. Go to https://neon.tech
2. Create a free project
3. Copy the connection string from the dashboard

#### Slim Dockerfile (No Ollama Required)

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV USE_HF_CLOUD=1

RUN apt-get update && apt-get install -y \
    curl \
    tesseract-ocr \
    libtesseract-dev \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["bash", "start_hf.sh"]
```

#### Key `requirements.txt` Additions

```txt
huggingface_hub      # HF Inference API client
groq>=0.9.0          # Groq Vision API client
google-genai>=1.0.0  # Optional: Google Gemini (requires billing for cloud IPs)
pytesseract>=0.3.10  # Local OCR fallback
pillow>=10.0.0
```

#### Deployment Steps

1. **Create a HuggingFace Space** at https://huggingface.co/spaces
   - SDK: **Docker**
   - Visibility: Public or Private

2. **Push your repository** to the Space's git remote:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main
   ```

3. **Add secrets** in Space Settings (see table above). Do **not** hardcode secrets in the Dockerfile or any committed file.

4. **Factory Reboot** the Space after adding secrets to trigger a clean rebuild with the new environment variables.

5. **Monitor the build logs** — a successful startup shows:
   ```
   [GRANITE_VQA] Target Model: meta-llama/llama-4-scout-17b-16e-instruct (via Groq Vision)
   [INFO] CLOUD MODE: Initializing Official HF InferenceClient (Qwen/Qwen2.5-Coder-7B-Instruct)
   INFO: Uvicorn running on http://0.0.0.0:7860
   ```

#### Important Notes on Line Endings

HuggingFace containers run Linux. If you develop on Windows, your shell scripts will have `\r\n` (CRLF) line endings, which cause `bad interpreter` errors on Linux. Add this to your `Dockerfile` to strip them automatically:

```dockerfile
RUN sed -i 's/\r$//' start_hf.sh && chmod +x start_hf.sh
```

Alternatively, configure Git to normalize line endings:
```bash
git config core.autocrlf input
```


---

## Project Summary

This project demonstrates that a self-correcting, multi-modal AI pipeline can be built and deployed entirely on consumer hardware through careful architectural decisions: progressive SQL generation with semantic validation, unified MCP-based tool access across data modalities, dynamic VRAM orchestration to prevent memory exhaustion, and a hybrid containerization strategy that preserves native GPU performance while maintaining deployment portability.

The system was developed iteratively, with each architectural decision — from model selection to validation logic to infrastructure design — informed by observed failure modes and measured outcomes rather than theoretical assumptions.
