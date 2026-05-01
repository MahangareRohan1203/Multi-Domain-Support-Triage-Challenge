# Multi-Domain Support Triage Agent
## High-Reliability RAG Pipeline for Support Automation

### Executive Summary
The Multi-Domain Triage Agent is a production-grade automated support system designed to safely handle, classify, and respond to customer inquiries across three distinct ecosystems: **HackerRank**, **Claude**, and **Visa**. 

The system utilizes a **Retrieval-Augmented Generation (RAG)** architecture with a secondary **Safety Evaluation Loop** to ensure all responses are grounded in official documentation and free from hallucinations.

---

### 🏗 System Architecture

The system is built on four core pillars:

#### 1. Knowledge Base (Vector Store)
- **Ingestion**: Scalable scraping engine with specialized handlers for different DOM structures and sitemap parsing.
- **Embedding**: Utilizes `all-MiniLM-L6-v2` for dense vector representation (384 dimensions).
- **Indexing**: FAISS (Facebook AI Similarity Search) FlatL2 index for low-latency retrieval.

#### 2. Retrieval Engine (`Retriever`)
- **Isolation**: Strict company-level metadata filtering during search to prevent cross-domain contamination (e.g., applying Visa policies to Claude users).
- **Precision**: Hybrid retrieval logic that balances context window constraints with document relevance.

#### 3. Core Agent (`Agent`)
- **Async Processing**: Built with `asyncio` and `Semaphore` concurrency control to maximize throughput while respecting API Rate Limits.
- **Schema Enforcement**: Pydantic models enforce 100% JSON compliance from the LLM, ensuring downstream data stability.
- **Resiliency**: Implements exponential backoff via `tenacity` to handle transient API failures and `ResourceExhausted` errors.

#### 4. Safety Evaluation (The Critic)
- **Zero-Trust Audit**: Optional secondary LLM pass (`--evaluate`) that reviews proposed responses against the source context to detect and block unsupported claims.

---

### 🛠 Operational Guide

#### Prerequisites
- Python 3.12+
- FAISS, SentenceTransformers, Google Generative AI, Pydantic, Typer, Tenacity.

#### Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
#### Running the Triage Pipeline
```bash
# Standard high-reliability run
python3 code/main.py --concurrency 2

# High-safety run with supervisor audit
python3 code/main.py --evaluate --concurrency 2
```

#### Quality Assurance (Unit Tests)
```bash
# Verify system logic without API usage
PYTHONPATH=. pytest tests/test_agent.py
```

---

### 📈 Official Compliance

| Field | Allowed Values |
| :--- | :--- |
| **`status`** | `replied`, `escalated` |
| **`request_type`** | `product_issue`, `feature_request`, `bug`, `invalid` |

---

### 🏗 System Architecture

The project follows the official repository structure:
- **`code/`**: Contains the core agent logic and supporting scripts.
- **`data/`**: Knowledge base and vector index.
- **`support_tickets/`**: Input and sample ticket data.
- **`tests/`**: Unit test suite.

- **PII Protection**: System is designed to escalate any tickets involving account access or identity verification to human experts.
- **Grounding**: The "Supervisor" evaluation mode ensures the agent never deviates from the provided support documentation.
- **Escalation Path**: Explicit logic to move sensitive cases (Threats, Fraud, Legal) to human queues immediately.

---
**Senior Engineer Note**: *This system prioritizes "Correctness over Completion." In cases of ambiguity or retrieval failure, the system is hard-coded to escalate to human support rather than risk a hallucinated response.*
