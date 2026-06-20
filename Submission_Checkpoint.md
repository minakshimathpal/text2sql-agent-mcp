# 🏁 Text2SQL Capstone: Project Checkpoint (Pre-Migration)

**Status:** Completed Local Demo / Ready for HF Cloud Migration
**Current Architecture:**
1.  **Memory Agent:** Handles chat history and VRAM cleanup.
2.  **Query Craft Agent:** 3-Level Engine (L1: Pure LLM, L2: SQL Fixer, L3: MCP Grounding).
3.  **Result Presenter Agent:** Executes SQL and formats human-readable answers.
4.  **Vision Agent:** Qwen 3.5 2B (via MCP) for visual OCR/Q&A.

**Key Technical Achievements:**
- Native-Bridge Ollama orchestration (Zero-latency GPU).
- Semantic Gatekeeper (Prevents hallucinations on joins).
- Decoupled MCP Tool Server for Vision and Database Introspection.

**Pending (Post-May 15th):**
- Migrate DB to Neon.tech (Cloud Postgres).
- Switch VLM/LLM to HuggingFace Inference API.
- Deploy Dockerized App to HuggingFace Spaces.
