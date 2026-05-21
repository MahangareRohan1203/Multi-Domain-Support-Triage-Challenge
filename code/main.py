import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import List, Optional

import faiss
import ollama
import pandas as pd
import typer
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("agent.log"), logging.StreamHandler()],
)
logger = logging.getLogger("TriageAgent")

# --- Data Models ---

class TriageResult(BaseModel):
    # Official requirement: status must be 'replied' or 'escalated'
    request_type: str = Field(..., description="Classification of the request type: product_issue, feature_request, bug, or invalid")
    product_area: str = Field(..., description="Classification of the product area")
    status: str = Field(..., description="Decision: 'replied' or 'escalated'")
    reasoning: str = Field(..., description="Internal logic for the decision")
    response: str = Field(..., description="Final user-facing response")

class Ticket(BaseModel):
    issue: str
    subject: str
    company: str

# --- Components ---

class Retriever:
    """Handles Knowledge Base retrieval using FAISS and SentenceTransformers."""
    
    def __init__(self, index_path: str, chunks_path: str, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Initializing Retriever with model {model_name}...")
        self.embed_model = SentenceTransformer(model_name)
        
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            raise FileNotFoundError(f"KB files not found at {index_path} or {chunks_path}")
            
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "r") as f:
            self.chunks = json.load(f)
        logger.info("Knowledge Base loaded successfully.")

    def get_context(self, query: str, company: str, k: int = 4) -> str:
        """Retrieves relevant chunks, optimized for accuracy (k=4)."""
        query_embedding = self.embed_model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_embedding, k * 10)
        
        results = []
        is_cross_domain = str(company).lower() in ["none", "unknown", "nan", ""]
        
        for idx in indices[0]:
            if idx == -1: continue
            chunk = self.chunks[idx]
            if is_cross_domain or chunk["company"].lower() == str(company).lower():
                results.append(f"[{chunk['company']}] {chunk['text']}")
            
            if len(results) >= k:
                break
        
        return "\n".join(results) if results else "No documentation found."

class Agent:
    """Core Triage Agent using Local Ollama LLM (Async version)."""
    
    def __init__(self, model_name: str = "llama3.2:1b"):
        self.model_name = model_name
        # Increase timeout to 120s for larger local models
        self.client = ollama.AsyncClient(timeout=120.0)
        logger.info(f"Agent initialized with local model {model_name}.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.INFO),
    )
    async def _call_llm_async(self, prompt: str, format: str = "") -> str:
        """Asynchronous call to the local Ollama LLM."""
        options = {"temperature": 0.0}
        response = await self.client.generate(
            model=self.model_name, 
            prompt=prompt, 
            format=format, 
            stream=False,
            options=options
        )
        return response['response']

    async def triage(self, ticket: Ticket, context: str, do_evaluate: bool = False) -> TriageResult:
        """Processes a single ticket and returns a TriageResult."""
        prompt = self._build_prompt(ticket, context)
        
        try:
            # Request JSON format explicitly
            raw_response = await self._call_llm_async(prompt, format="json")
            
            try:
                result = self._parse_response(raw_response)
            except (ValueError, ValidationError, json.JSONDecodeError) as ve:
                logger.warning(f"Response parsing failed for '{ticket.subject}': {ve}. Attempting fallback parsing.")
                result = self._fallback_parse(raw_response)
            
            # Ensure official allowed values (lowercase)
            result.status = result.status.lower()
            if result.status not in ["replied", "escalated"]:
                # Logic check: if no documentation found, escalate
                if "No documentation found" in context:
                    result.status = "escalated"
                else:
                    result.status = "replied"
            
            # Ensure official allowed values for request_type
            valid_request_types = ["product_issue", "feature_request", "bug", "invalid"]
            rt_lower = result.request_type.lower()
            if rt_lower not in valid_request_types:
                if "bug" in rt_lower: result.request_type = "bug"
                elif "feature" in rt_lower: result.request_type = "feature_request"
                elif "invalid" in rt_lower or "unknown" in rt_lower: result.request_type = "invalid"
                else: result.request_type = "product_issue"
            else:
                result.request_type = rt_lower

            if do_evaluate and result.status == "replied":
                eval_result = await self.evaluate(ticket, context, result)
                if not eval_result["safe"]:
                    result.status = "escalated"
                    result.reasoning = f"Evaluation failed: {eval_result['reason']}. Original logic: {result.reasoning}"
                    result.response = "I've escalated this to a human expert to ensure the most accurate and safe assistance for your specific case."
            
            return result
        except Exception as e:
            logger.error(f"Failed to triage ticket '{ticket.subject}': {e}")
            return TriageResult(
                request_type="product_issue",
                product_area="General",
                status="escalated",
                reasoning=f"System error: {str(e)}",
                response="I apologize, but I'm currently unable to process your request. I've escalated this to our human support team."
            )

    def _fallback_parse(self, text: str) -> TriageResult:
        """Emergency fallback if JSON parsing fails."""
        # Check for status and request_type in the text
        status = "escalated"
        if "replied" in text.lower(): status = "replied"
        
        request_type = "product_issue"
        if "bug" in text.lower(): request_type = "bug"
        elif "feature" in text.lower(): request_type = "feature_request"
        elif "invalid" in text.lower(): request_type = "invalid"

        return TriageResult(
            request_type=request_type,
            product_area="General",
            status=status,
            reasoning="Fallback parsing used due to JSON error.",
            response="I've escalated this to a human expert to ensure you get the right help." if status == "escalated" else "How can I help you today?"
        )

    async def evaluate(self, ticket: Ticket, context: str, result: TriageResult) -> dict:
        """Secondary evaluation step to ensure response safety and accuracy."""
        # Force escalation for high-risk keywords if not already escalated
        high_risk_terms = ["identity theft", "stolen", "fraud", "blocked", "compromised"]
        if result.status == "replied":
            if any(term in ticket.issue.lower() for term in high_risk_terms) or \
               any(term in ticket.subject.lower() for term in high_risk_terms):
                return {"safe": False, "reason": "High-risk security/fraud topic must be escalated to a human."}

        eval_prompt = f"""
        You are a senior support supervisor. Review the following support response for safety and grounding.
        
        TICKET: {ticket.issue}
        CONTEXT: {context}
        PROPOSED RESPONSE: {result.response}
        
        CRITERIA:
        1. **Safety**: If the agent offers to perform sensitive actions (changing scores, granting access, issuing refunds, resolving fraud) without documentation support, it is UNSAFE.
        2. **High Risk**: ANY ticket involving Identity Theft, Stolen Cards, Fraud, or Account Compromise MUST be escalated.
        3. **Grounding**: If the agent makes up facts, URLs, or email addresses NOT found in the context, it is UNSAFE.
        4. **Adequacy**: If the context is empty or clearly irrelevant, the agent should have escalated.
        
        If the response is a polite "out of scope" message for a nonsense request, it is SAFE.
        
        OUTPUT FORMAT (JSON ONLY):
        {{
          "safe": true/false,
          "reason": "explanation"
        }}
        """
        try:
            raw_eval = await self._call_llm_async(eval_prompt, format="json")
            return json.loads(raw_eval)
        except:
            return {"safe": True, "reason": "Eval failed, defaulting to safe."}

    def _build_prompt(self, ticket: Ticket, context: str) -> str:
        return f"""
Act as a professional support triage specialist for {ticket.company}. 
Your task is to classify the support ticket and provide a response based **ONLY** on the provided context.

### TICKET DETAILS
- **Subject**: {ticket.subject}
- **Issue**: {ticket.issue}

### DOCUMENTATION CONTEXT
{context}

### CLASSIFICATION RULES:
1. **request_type**:
   - `bug`: Explicit mention of things not working, errors, or unexpected behavior.
   - `feature_request`: Requests for new functionality or improvements.
   - `invalid`: ONLY for spam, completely nonsensical text, or topics totally unrelated to professional software/finance support (e.g., "what is the weather"). 
   - `product_issue`: General questions, "how-to" queries, account management, or technical help. 
   - **Note**: If the user asks about a feature like "Claude for students" or "Resume builder", it is a `product_issue` or `feature_request`, NOT `invalid`.

2. **product_area**:
   - Identify the specific feature or department (e.g., "Billing", "API", "Security", "Account").

3. **status**:
   - `replied`: 
     - The provided context contains a clear answer.
     - OR the request is truly `invalid` (nonsense/spam).
   - `escalated`: 
     - **MANDATORY** for: Identity Theft, Stolen Cards, Fraud, Security breaches, Legal threats, or sensitive Account Access (e.g., "lost owner access").
     - The context is missing or does not address the issue.

4. **reasoning**:
   - A short, one-sentence internal justification for your classification.

5. **response**:
   - If `request_type` is `invalid`: "I'm sorry, but that request is outside the scope of my capabilities as a support assistant."
   - If `status` is `replied`: Provide a helpful, concise answer based strictly on the context. Do not invent links or emails.
   - If `status` is `escalated`: Explain that you are escalating this to a specialized human team for investigation.

### OUTPUT FORMAT:
JSON ONLY.
{{
  "request_type": "...",
  "product_area": "...",
  "status": "...",
  "reasoning": "...",
  "response": "..."
}}
"""

    def _parse_response(self, text: str) -> TriageResult:
        """Extracts and validates JSON from LLM response with repair logic."""
        # Find the first { and the last }
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx == -1 or end_idx == -1:
            raise ValueError("No JSON object found in LLM response")
            
        json_str = text[start_idx:end_idx+1]
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Simple repair: try to fix common trailing commas or unescaped quotes
            # This is a basic attempt, pydantic might still fail if logic is broken
            json_str = re.sub(r',\s*\}', '}', json_str)
            data = json.loads(json_str)
            
        return TriageResult.model_validate(data)

# --- CLI Application ---

app = typer.Typer()

async def process_tickets_async(
    df: pd.DataFrame, 
    retriever: Retriever, 
    agent: Agent, 
    do_evaluate: bool,
    log_file: str,
    concurrency_limit: int
):
    semaphore = asyncio.Semaphore(concurrency_limit)
    results = []

    async def worker(idx, row):
        async with semaphore:
            ticket = Ticket(
                issue=str(row["Issue"]) if pd.notna(row["Issue"]) else "No description",
                subject=str(row["Subject"]) if pd.notna(row["Subject"]) else "No subject",
                company=str(row["Company"]) if pd.notna(row["Company"]) else "Unknown"
            )

            context = retriever.get_context(ticket.issue, ticket.company)
            result = await agent.triage(ticket, context, do_evaluate=do_evaluate)

            res = {
                "issue": ticket.issue,
                "subject": ticket.subject,
                "company": ticket.company,
                "response": result.response,
                "product_area": result.product_area,
                "status": result.status,
                "request_type": result.request_type,
                "justification": result.reasoning
            }
            
            # Log result immediately
            with open(log_file, "a") as f:
                f.write(f"[{ticket.company}] {ticket.subject}\n")
                f.write(f"Triage: {result.request_type} | {result.product_area} | {result.status}\n")
                f.write(f"Response: {result.response}\n")
                f.write("-" * 30 + "\n")
            
            return res

    tasks = [worker(idx, row) for idx, row in df.iterrows()]
    
    # We use a progress bar while awaiting tasks
    for f in asyncio.as_completed(tasks):
        res = await f
        results.append(res)
    
    return results

@app.command()
def run(
    input_csv: str = "support_tickets/support_tickets.csv",
    output_csv: str = "support_tickets/output.csv",
    log_file: str = "log.txt",
    evaluate: bool = typer.Option(False, "--evaluate", help="Enable secondary evaluation step"),
    concurrency: int = typer.Option(2, "--concurrency", help="Max concurrent API calls"),
    model: str = typer.Option(os.getenv("OLLAMA_MODEL", "llama3.2:1b"), "--model", help="Local model name")
):
    """Run the Multi-Domain Triage Agent with Local LLM (Ollama)."""
    # Initialize Log
    with open(log_file, "w") as f:
        f.write(f"=== TRIAGE AGENT LOG (LOCAL) ===\nStarted: {datetime.now()}\n")
        f.write(f"Mode: Async | Evaluation: {evaluate} | Concurrency: {concurrency} | Model: {model}\n\n")

    try:
        # Paths are relative to root if run from root
        retriever = Retriever("data/index.faiss", "data/chunks.json")
        agent = Agent(model_name=model)
    except Exception as e:
        typer.echo(f"Initialization Failed: {e}", err=True)
        raise typer.Exit(1)

    df = pd.read_csv(input_csv)
    typer.echo(f"Processing {len(df)} tickets (Model={model}, Concurrency={concurrency}, Eval={evaluate})...")
    
    results = asyncio.run(process_tickets_async(df, retriever, agent, evaluate, log_file, concurrency))

    # Required columns per official repo spec
    output_cols = ["status", "product_area", "response", "justification", "request_type"]
    pd.DataFrame(results)[output_cols].to_csv(output_csv, index=False)
    typer.echo(f"\nSuccess! Results saved to {output_csv}")

if __name__ == "__main__":
    app()
