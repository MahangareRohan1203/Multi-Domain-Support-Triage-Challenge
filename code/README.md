# Triage Agent

This is the core agent for the HackerRank Orchestrate challenge.

## Setup

1.  **Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # or
    .\venv\Scripts\activate  # Windows
    ```

2.  **Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If requirements.txt is missing, install these manually:*
    `pip install google-generativeai faiss-cpu sentence-transformers pandas typer python-dotenv pydantic tenacity`

3.  **Environment Variables**:
    Copy `.env.example` from the root to `.env` and add your `GOOGLE_API_KEY`.

## Usage

Run the agent from the root directory:

```bash
python code/main.py run
```

### Options:

- `--evaluate`: Enable secondary evaluation step for safer responses.
- `--concurrency <int>`: Set max concurrent API calls (default: 2).
- `--input-csv <path>`: Path to input CSV (default: support_tickets/support_tickets.csv).
- `--output-csv <path>`: Path to output CSV (default: support_tickets/output.csv).

## Architecture

- **Retriever**: Uses FAISS and SentenceTransformers for local knowledge retrieval from `data/`.
- **Agent**: Uses Gemini 1.5 Flash for classification and response generation.
- **Evaluation**: Optional second pass to verify response safety and grounding.
- **Async**: Processes tickets concurrently to improve performance.
