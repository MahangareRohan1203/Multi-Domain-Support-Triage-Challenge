# Interview Preparation: Multi-Domain Support Triage Agent

This guide provides a structured way to explain the project and prepares you for potential technical cross-questions.

---

## 1. Project Overview (The Elevator Pitch)
"I developed a **High-Reliability Support Triage Pipeline** designed to handle customer inquiries across three distinct ecosystems: **HackerRank, Claude, and Visa**. The system uses a **Retrieval-Augmented Generation (RAG)** architecture to ensure every response is grounded in a 20,000+ line official support corpus. It prioritizes **Safety and Precision** over completion, automatically escalating high-risk cases like security breaches, fraud, or account access issues to human experts."

---

## 2. Key Technical Pillars

### A. RAG Architecture (Retrieval-Augmented Generation)
*   **Vector Database:** Used **FAISS** for high-performance similarity search.
*   **Embeddings:** Utilized the `all-MiniLM-L6-v2` model to transform text into 384-dimensional vectors.
*   **Context Grounding:** The LLM is instructed to answer *only* using retrieved chunks, eliminating outside knowledge hallucinations.

### B. High-Performance Execution
*   **Async Processing:** Built with `asyncio` to handle multiple tickets concurrently using a `Semaphore` to manage hardware load.
*   **Local LLM:** Powered by **Ollama (Llama 3.2)**, keeping data private and avoiding external API costs/latency.

### C. Production-Grade Guardrails
*   **Pydantic Validation:** Used strict schema enforcement to ensure the LLM always returns structured JSON.
*   **The "Critic" Loop:** An optional supervisor mode (`--evaluate`) that performs a secondary audit on responses to detect unsupported claims.
*   **Metadata Isolation:** Implemented logic to prevent "cross-talk" (e.g., ensuring a Visa question never retrieves HackerRank documentation).

---

## 3. Sample Cross-Questions & Answers

### Q1: Why did you choose a local LLM (Ollama) instead of GPT-4?
**Answer:** "Two main reasons: **Privacy and Cost.** In a support context, tickets often contain sensitive user issues. Running a local model ensures no data leaves the infrastructure. Additionally, it allows for unlimited testing and high-volume processing without incurring per-token API costs."

### Q2: How do you handle "hallucinations" (the LLM making things up)?
**Answer:** "I use a three-layered approach:
1.  **Strict Prompting:** The system prompt explicitly forbids using internal knowledge.
2.  **RAG Constraints:** We only provide the LLM with relevant documentation snippets.
3.  **The Critic Loop:** If enabled, a second LLM pass compares the answer against the retrieved context. If the answer contains facts not found in the context, the system automatically escalates to a human."

### Q3: What happens if a ticket doesn't specify which company it's for?
**Answer:** "The `Retriever` is designed to handle 'cross-domain' queries. If the company is unknown, the system performs a global search across the entire corpus. Based on the retrieved documentation, the Agent then identifies the most likely product area and company context to formulate a response."

### Q4: How do you handle high-risk topics like fraud or security?
**Answer:** "The system has a 'Deterministic Escalation' policy. I've instructed the model to identify keywords and categories related to Security, Fraud, Legal, or Account Access. In these cases, even if documentation is available, the `status` is set to `escalated` because these issues require human empathy and verified authority."

### Q5: Your system uses `asyncio`. How do you prevent it from crashing the local machine during heavy load?
**Answer:** "I implemented a `Semaphore` with a configurable concurrency limit (defaulting to 2). This acts as a 'throttle,' ensuring that while we are processing tickets asynchronously, we never overwhelm the local LLM or the system's RAM/GPU."

### Q6: If you had more time, what would you improve?
**Answer:** "I would implement **Hybrid Search** (combining Vector search with BM25 keyword search) to better handle specific technical terms or error codes. I'd also add a **Feedback Loop** where human agents could flag incorrect triage results to fine-tune the local model or improve the documentation indexing."

---

## 4. Technical Stack Summary
*   **Language:** Python 3.12
*   **Data Handling:** Pandas, Pydantic
*   **AI/Vector:** FAISS, SentenceTransformers, Ollama
*   **Infrastructure:** Typer (CLI), Tenacity (Retries), Asyncio
