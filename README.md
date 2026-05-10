# Text-to-SQL & Document VQA Multi-Agent System

![Architecture](https://img.shields.io/badge/Architecture-Multi--Agent-blue)
![Database](https://img.shields.io/badge/Database-Agnostic-success)
![LLM](https://img.shields.io/badge/LLMs-Qwen%20%7C%20Ollama-orange)
![Protocol](https://img.shields.io/badge/Protocol-MCP-purple)

An advanced, hardware-optimized AI pipeline built on a multi-agent orchestration architecture. This system seamlessly integrates natural language Text-to-SQL generation with a powerful Vision-Language Model (VLM) for Optical Character Recognition (OCR) and Visual Question Answering (VQA).

By leveraging the **Model Context Protocol (MCP)**, local LLMs via Ollama, and intelligent memory management, this project achieves state-of-the-art data extraction and database querying running entirely locally within tight memory constraints.

---

## Key Features

- **Multi-Agent Orchestration:** A dynamic routing system featuring specialized agents (`MemoryAgent`, `QueryCraftAgent`, `ResultPresenterAgent`).
- **Vision-Language Q&A (VQA):** Integrates **Qwen 3.5 2B Vision** for highly accurate data extraction from complex documents (e.g., Invoices, unstructured tables) directly via Ollama.
- **Advanced Text-to-SQL:** Utilizes **Qwen 2.5 Coder 1.5B** to translate complex natural language questions into syntactically valid SQL queries, executing them securely against local databases.
- **Dynamic Memory Management:** Custom algorithms automatically swap models in and out of GPU VRAM to prevent Out-Of-Memory (OOM) crashes, enabling heavy AI workloads on consumer hardware.
- **Database Agnostic:** Designed to interface seamlessly with PostgreSQL, MySQL, SQLite, MS SQL Server, and Oracle.
- **Conversational Interface:** Full Web UI for natural language interactions, maintaining context across sessions.

---

## System Architecture & Agentic Workflow

The project utilizes a decoupled, asynchronous microservices architecture. The Web UI communicates with a backend worker, which in turn orchestrates requests via the Model Context Protocol (MCP).

### The Three-Agent Triad

Our orchestration is built upon three highly specialized agents working in sequence:

1. **MemoryAgent:** 
   - Acts as the first layer of defense.
   - Checks if the user's query can be answered using historical context.
   - Rephrases ambiguous queries based on past interactions.
   - Routes new questions to the appropriate downstream agent.
2. **QueryCraftAgent (The 3-Level Text-to-SQL Engine):**
   - The core logic engine for SQL generation, utilizing a sophisticated 3-Level execution architecture:
     - **Level 1 (Pure LLM Intelligence):** Zero-shot semantic translation of natural language to raw SQL using Qwen 2.5 Coder.
     - **Level 2 (LLM + SQL Fixer Enhancement):** An autonomous self-reflection loop where the agent catches its own syntax errors and rewrites the query.
     - **Level 3 (LLM + MCP Hint + SQL Fixer):** The deepest reasoning level, where the agent actively uses MCP tools to introspect live database schemas (tables, columns, foreign keys) to ground its SQL generation in absolute reality before executing the Fixer loop.
3. **ResultPresenterAgent:**
   - Executes the validated SQL via the MCP Tool Server.
   - Translates raw, technical SQL results into natural, conversational responses.
   - Handles edge cases and updates the persistent conversation history.

### Multi-Agent Orchestration Flowchart

```mermaid
graph TD
    User([User Query / Upload]) --> API[FastAPI Gateway]
    API --> Orchestrator{Agent Orchestrator}
    
    Orchestrator -->|Image Upload| VQA[VQA Engine]
    VQA --> Result([Final Answer])
    
    Orchestrator -->|Text Query| Memory[MemoryAgent]
    Memory -->|Context Found| Presenter[ResultPresenterAgent]
    Memory -->|New Query| QueryCraft[QueryCraftAgent]
    
    QueryCraft -->|Schema Context| OllamaSQL[(Qwen 2.5 Coder)]
    OllamaSQL -->|Raw SQL| MCP[Model Context Protocol]
    MCP --> DB[(Enterprise Database)]
    
    DB -->|Result Set| Presenter
    Presenter --> Result
```

### Detailed MCP Orchestration Flow (Vision/OCR Pipeline)

To handle memory constraints while executing heavy Visual QA tasks, the system dynamically unloads models and delegates image processing to the MCP Tool Server.

```text
         Qwen 3.5 2B (OCR Q&A mode)
        │
    Worker /query
        │  image .img_path found?
        ├─ YES → ocr_agent_qa()
        │         │
        │         ├─ unload QwenCoder Text-To-SQL (free GPU VRAM)
        │         │
        │         └─ mcp_call_tool("granite_vision.qa", {image, question})
        │                   │
        │               MCP Server (port 8701)
        │                   │
        │               granite_vision.qa tool (HTTP Proxy)
        │                   │
        │               Qwen 3.5 2B (Ollama C++ Backend)
        │                   │
        │               ← Answer Dict Extracted
        │
        └─ NO (old doc) → Tesseract Text → Gemma3 Fallback (unchanged)
```

---

## Technology Stack

- **Core Framework:** Python 3.10+, FastAPI, Uvicorn
- **AI Orchestration:** LlamaIndex, Model Context Protocol (MCP)
- **Local Inference Engine:** Ollama (C++ Optimized, Memory Mapped)
- **Models Used:** 
  - `qwen2.5-coder:1.5b` (For expert-level SQL syntax generation)
  - `qwen3.5:2b` (For robust Visual Q&A and OCR formatting)
- **Database ORM:** SQLAlchemy

---

## Research & Fine-Tuning: Comparative Analysis

As part of the project's rigorous scientific methodology, a custom foundational model fine-tuning pipeline was developed. 

1. **LoRA Fine-Tuning:** The base **Qwen 2.5 3B** model was fine-tuned using Low-Rank Adaptation (LoRA) on a specialized Text-to-SQL dataset.
2. **Weight Merging:** The LoRA adapters were successfully merged back into the base model weights (`merge_qwen_to_ollama.py`) and quantized for local deployment.
3. **A/B Testing & Evaluation:** During the validation phase, the fine-tuned general model was benchmarked against the highly specialized `qwen2.5-coder:1.5b`. 
4. **Engineering Decision:** The benchmark revealed that the zero-shot capabilities of the Coder model (which is heavily pre-trained on billions of lines of code and SQL) yielded higher syntactical accuracy and faster inference speeds than the generalized LoRA fine-tune. Therefore, the production architecture was optimized to leverage the Coder model, demonstrating a practical, data-driven engineering decision prioritizing system reliability.

*(See `Fine_Tuning_Qwen_2_5_3B.ipynb` and the `merged_qwen2_5_3b_text2sql` directories for the fine-tuning methodology.)*

---

## Installation & Setup

### Prerequisites
- [Ollama](https://ollama.com/) installed and running locally
- Python 3.11+
- Database connection (PostgreSQL, MySQL, SQLite)

### 1. Model Initialization
Pull the required highly-optimized models into Ollama:
```bash
ollama pull qwen2.5-coder:1.5b
ollama pull qwen3.5:2b
```

### 2. Environment Configuration
Install dependencies:
```bash
pip install -r requirements.txt
```
Create a `.env` file in the `Agent/` directory containing your database credentials:
```env
DB_CONNECTION_URL=postgresql://user:password@localhost:5432/dbname
OLLAMA_MODEL_NAME=qwen2.5-coder:1.5b
GRANITE_MODEL_PATH=qwen3.5:2b
```

### 3. Running the Pipeline
The architecture requires three microservices running in parallel for decoupled processing.

**Terminal 1 (Backend Worker):**
```bash
python -u -m uvicorn "Agent.worker:app" --port 8700
```
**Terminal 2 (MCP Tool Server):**
```bash
python -m uvicorn "Agent.mcp_server:app" --port 8701
```
**Terminal 3 (Web UI):**
```bash
python -m uvicorn "web.app:app" --port 8000
```
Navigate to `http://localhost:8000` to interact with the multi-agent system!

---

## Docker Containerization (Optional)

To run the entire system in a robust, containerized environment:

1. Ensure Docker and Docker Compose are installed.
2. Build and start the services:
```bash
docker-compose up --build
```
*(Note: Ensure your `docker-compose.yml` maps the Ollama port `11434` properly so the containers can reach your local LLMs, or install Ollama directly inside the container network.)*

## Deployment Pipeline (HuggingFace Spaces)

To deploy this application to **HuggingFace Spaces**:
1. Create a new Space and select **Docker** as the SDK.
2. Upload this repository.
3. Configure the Space hardware to an instance with at least **16GB RAM** (or a T4 GPU).
4. Add `DB_CONNECTION_URL` to the HuggingFace Secrets.
5. HuggingFace will automatically build the Docker container and host the Web UI publically. 

---
*Built with ❤️ for advanced Agentic Data Engineering.*
