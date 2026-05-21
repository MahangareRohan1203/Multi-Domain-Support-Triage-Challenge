# Low-Level Design (LLD): Code Deep Dive

This document provides a granular, logic-by-logic breakdown of the `code/main.py` implementation.

---

## 1. Knowledge Retrieval (`Retriever` Class)

### Embedding Logic (`SentenceTransformer`)
*   **Line:** `self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")`
*   **Deep Dive:** We use a "Dense Vector" approach. This model maps sentences to a 384-dimensional space. Unlike simple keyword matching, this allows the system to understand that "I can't log in" and "Access denied" are semantically identical.

### Similarity Search (`FAISS`)
*   **Line:** `self.index.search(query_embedding, k * 10)`
*   **Deep Dive:** We perform a "Top-K" search. We actually search for `k * 10` results initially. Why? To allow for **Metadata Filtering**. Since the FAISS index is "flat", it doesn't know about companies. We retrieve a larger pool of results and then manually filter for the correct `company` metadata in Python to ensure domain isolation.

---

## 2. The AI Brain (`Agent` Class)

### Async Orchestration (`ollama.AsyncClient`)
*   **Line:** `await self.client.generate(...)`
*   **Deep Dive:** Inference is an I/O bound task for the CPU/GPU. By using `await`, we release the Python Global Interpreter Lock (GIL), allowing the script to prepare other tickets or filter search results while waiting for the LLM to generate tokens.

### Resilience (`tenacity` decorators)
*   **Line:** `@retry(stop=stop_after_attempt(3), wait=wait_exponential(...))`
*   **Deep Dive:** Local models can occasionally "hang" or fail if the system RAM is spiked. The `wait_exponential` pattern (2s, 4s, 8s) prevents "thundering herd" problems, giving the local Ollama server time to recover before the next attempt.

### Structured Output & Parsing
*   **Line:** `format="json"` in the LLM call.
*   **Deep Dive:** We leverage Ollama's ability to constrain the model to valid JSON. 
*   **The Fallback:** `_fallback_parse` is a critical safety net. If the model produces invalid JSON or hallucinates a status, we use Regex to "scrape" the intention from the text, defaulting to `escalated` if there's any ambiguity.

---

## 3. High-Concurrency Pipeline (`process_tickets_async`)

### The Throttle (`asyncio.Semaphore`)
*   **Line:** `semaphore = asyncio.Semaphore(concurrency_limit)`
*   **Deep Dive:** This is the most important "production" feature. Without it, `asyncio` would attempt to send all 50+ tickets to the LLM at once, which would crash the local model or cause severe latency. The Semaphore ensures only `N` tickets are being processed by the LLM at any single moment.

### Result Gathering
*   **Line:** `asyncio.as_completed(tasks)`
*   **Deep Dive:** Instead of waiting for *all* tickets to finish (`asyncio.gather`), we use `as_completed`. This allows the script to log results to the terminal and `log.txt` in real-time as they finish, providing immediate feedback to the user.

---

## 4. Triage Logic Flow (`triage` method)

1.  **Strict Grounding**: The prompt uses "ONLY on the provided context."
2.  **Safety Assessment**: The logic prioritizes `status="escalated"` for specific categories (Security, Fraud, Score changes) even if documentation exists.
3.  **The Critic Pass**: If `evaluate=True`, the `evaluate` method runs a "Zero-Trust" audit. It asks a second instance of the model: "Does the context actually support the Agent's answer?" If the answer is `False`, the system overrides the agent and escalates.

---

## 5. CLI Entry Point (`typer`)

*   **Initialization**: The script checks for the existence of `index.faiss` and `chunks.json` immediately. Failing fast is better than failing mid-process.
*   **Batch Saving**: We use `pandas` for the final CSV export to ensure the output perfectly matches the schema required by the challenge evaluators.
