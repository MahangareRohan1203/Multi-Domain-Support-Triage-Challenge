# DeepDives: How the Triage Agent Works 🤖

Welcome! If you are new to the AI domain, this document will walk you through how we built this Support Triage Agent from the ground up.

## 1. The Core Architecture: RAG
This agent uses **RAG (Retrieval-Augmented Generation)**. Instead of just asking an AI to "guess" the answer, we:
1.  **Retrieve**: Find the exact support documents related to the user's problem.
2.  **Augment**: Add those documents to the prompt.
3.  **Generate**: Ask the AI to write a response based *only* on those documents.

## 2. Technical Stack
-   **Gemini 1.5 Flash**: Our "Brain." It's fast, efficient, and great at following strict JSON instructions.
-   **FAISS (Facebook AI Similarity Search)**: Our "Library Index." It allows us to search through thousands of documentation chunks in milliseconds.
-   **Sentence Transformers**: Our "Translator." It converts human text into math (embeddings) so the computer can understand semantic meaning.

## 3. The Triage Pipeline
When a ticket comes in:
1.  **Vector Search**: We convert the ticket subject into a vector and find the top 2 matching articles in our `data/` folder.
2.  **Prompt Engineering**: We build a "Lean Prompt" that includes the ticket, the retrieved context, and strict rules (e.g., "Escalate if it's a security risk").
3.  **Structured Output**: The AI responds in a strict JSON format, which our code parses into a spreadsheet.

## 4. Cost & Performance Optimization
To make this "CEO-ready," we implemented:
-   **Context Trimming**: We only use the top 2 knowledge chunks to save tokens.
-   **Async Processing**: We process multiple tickets at once to save time.
-   **Temperature 0**: We keep the AI deterministic so it doesn't "hallucinate" different answers for the same problem.

## 5. Security & Safety
-   **No Hardcoded Keys**: All secrets are in `.env`.
-   **Safety Guardrails**: The prompt explicitly forbids the AI from changing scores or handling sensitive billing without human oversight.

---
*Created for the HackerRank Orchestrate Challenge 2026.*
