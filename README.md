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

For peak inference performance, the system uses a hybrid offloading strategy. The application logic and MCP servers run inside Docker containers for portability and isolation, while Ollama and all GPU inference are offloaded to the native Windows host and accessed by the containers over the WSL2 host bridge network (`OLLAMA_BASE_URL=http://172.27.224.1:11434`).

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
| Core Framework | Python 3.10+, FastAPI, Uvicorn |
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
- Python 3.10 or higher
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

1. Create a new Space and select Docker as the SDK.
2. Upload this repository.
3. Configure the Space to an instance with at least 16GB RAM, or a T4 GPU tier.
4. Add `DB_CONNECTION_URL` and Ollama model configuration to the Space Secrets.
5. In the Dockerfile, uncomment the Ollama installation line to run inference inside the container rather than offloading to a host.
6. HuggingFace will build the Docker image and serve the Web UI publicly.

---

## Project Summary

This project demonstrates that a self-correcting, multi-modal AI pipeline can be built and deployed entirely on consumer hardware through careful architectural decisions: progressive SQL generation with semantic validation, unified MCP-based tool access across data modalities, dynamic VRAM orchestration to prevent memory exhaustion, and a hybrid containerization strategy that preserves native GPU performance while maintaining deployment portability.

The system was developed iteratively, with each architectural decision — from model selection to validation logic to infrastructure design — informed by observed failure modes and measured outcomes rather than theoretical assumptions.
