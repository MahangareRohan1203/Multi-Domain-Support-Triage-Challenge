import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import List, Optional

import faiss
import google.generativeai as genai
import pandas as pd
import typer
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from google.api_core import exceptions

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

    def get_context(self, query: str, company: str, k: int = 3) -> str:
        """Retrieves relevant chunks, optionally filtered by company."""
        query_embedding = self.embed_model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_embedding, k * 10)
        
        results = []
        is_cross_domain = str(company).lower() in ["none", "unknown", "nan", ""]
        
        for idx in indices[0]:
            if idx == -1: continue
            chunk = self.chunks[idx]
            
            # If company is provided and not generic, we filter. 
            # Otherwise, we take the best semantic matches regardless of company.
            if is_cross_domain or chunk["company"].lower() == str(company).lower():
                results.append(f"Company: {chunk['company']}\nSource: {chunk['url']}\nContent: {chunk['text']}")
            
            if len(results) >= k:
                break
        
        return "\n---\n".join(results) if results else "No specific support documentation found."

class Agent:
    """Core Triage Agent using Gemini LLM (Async version)."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name,
            generation_config={"temperature": 0.0}
        )
        logger.info(f"Agent initialized with model {model_name} (temperature=0.0).")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type((
            exceptions.ResourceExhausted, 
            exceptions.ServiceUnavailable, 
            exceptions.InternalServerError
        )),
        before_sleep=before_sleep_log(logger, logging.INFO),
    )
    async def _call_llm_async(self, prompt: str) -> str:
        """Asynchronous call to the LLM."""
        response = await self.model.generate_content_async(prompt)
        return response.text

    async def triage(self, ticket: Ticket, context: str, do_evaluate: bool = False) -> TriageResult:
        """Processes a single ticket and returns a TriageResult."""
        prompt = self._build_prompt(ticket, context)
        
        try:
            raw_response = await self._call_llm_async(prompt)
            
            try:
                result = self._parse_response(raw_response)
            except (ValueError, ValidationError, json.JSONDecodeError) as ve:
                logger.warning(f"Response parsing failed for '{ticket.subject}': {ve}. Attempting fallback parsing.")
                # Basic regex fallback or default
                result = self._fallback_parse(raw_response)
            
            # Ensure official allowed values (lowercase)
            result.status = result.status.lower()
            if result.status not in ["replied", "escalated"]:
                result.status = "escalated"
            
            # Ensure official allowed values for request_type
            valid_request_types = ["product_issue", "feature_request", "bug", "invalid"]
            if result.request_type.lower() not in valid_request_types:
                rt = result.request_type.lower()
                if "bug" in rt: result.request_type = "bug"
                elif "feature" in rt: result.request_type = "feature_request"
                elif "invalid" in rt or "unknown" in rt: result.request_type = "invalid"
                else: result.request_type = "product_issue" # Default
            else:
                result.request_type = result.request_type.lower()

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
        eval_prompt = f"""
        You are a senior support supervisor reviewing a junior agent's work.
        Determine if the response is safe, grounded in context, and does not hallucinate policies.
        
        TICKET: {ticket.issue}
        CONTEXT: {context}
        AGENT RESPONSE: {result.response}
        
        CRITERIA:
        1. Is the response supported by the context?
        2. Does it avoid making unsupported promises or claims?
        3. Is it professionally appropriate?
        
        OUTPUT FORMAT:
        Output valid JSON only:
        {{
          "safe": true/false,
          "reason": "Brief explanation"
        }}
        """
        try:
            raw_eval = await self._call_llm_async(eval_prompt)
            json_match = re.search(r"\{.*\}", raw_eval, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return {"safe": True, "reason": "Eval failed to produce JSON, defaulting to safe."}
        except:
            return {"safe": True, "reason": "Eval API error, defaulting to safe."}

    def _build_prompt(self, ticket: Ticket, context: str) -> str:
        return f"""
You are an expert support triage agent for {ticket.company}. 
Analyze the ticket and provide a comprehensive response based on the provided support documentation.

TICKET DETAILS:
Subject: {ticket.subject}
Issue: {ticket.issue}

SUPPORT CONTEXT:
{context}

TASKS:
1. Classify 'request_type' (Allowed values: product_issue, feature_request, bug, invalid).
2. Classify 'product_area' (e.g., Account, Billing, Security, Technical, Integration, Assessment).
3. Determine 'status' (Allowed values: replied, escalated).
   ESCALATE IF:
   - High-risk security/fraud (stolen cards, hacked accounts).
   - Threats of violence or self-harm.
   - Legal or regulatory complaints.
   - Requests to change scores or bypass anti-cheating (for HackerRank).
   - Complex billing issues needing manual verification.
   - The issue is entirely unsupported by the context.
4. Generate 'response':
   - If Status is 'replied': Provide a professional, concise answer based ONLY on the context. 
   - If Status is 'escalated': Provide a polite explanation of why the case is being moved to a human expert.

OUTPUT FORMAT:
Output valid JSON only:
{{
  "request_type": "product_issue | feature_request | bug | invalid",
  "product_area": "...",
  "status": "replied | escalated",
  "reasoning": "...",
  "response": "..."
}}
"""

    def _parse_response(self, text: str) -> TriageResult:
        """Extracts and validates JSON from LLM response."""
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in LLM response")
        
        data = json.loads(json_match.group(0))
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
    concurrency: int = typer.Option(2, "--concurrency", help="Max concurrent API calls")
):
    """Run the Multi-Domain Triage Agent with Async support and Evaluation."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        typer.echo("Error: GOOGLE_API_KEY not found in environment.", err=True)
        raise typer.Exit(1)

    # Initialize Log
    with open(log_file, "w") as f:
        f.write(f"=== TRIAGE AGENT LOG ===\nStarted: {datetime.now()}\n")
        f.write(f"Mode: Async | Evaluation: {evaluate} | Concurrency: {concurrency}\n\n")

    try:
        # Paths are relative to root if run from root
        retriever = Retriever("data/index.faiss", "data/chunks.json")
        agent = Agent(api_key)
    except Exception as e:
        typer.echo(f"Initialization Failed: {e}", err=True)
        raise typer.Exit(1)

    df = pd.read_csv(input_csv)
    typer.echo(f"Processing {len(df)} tickets (Concurrency={concurrency}, Eval={evaluate})...")
    
    results = asyncio.run(process_tickets_async(df, retriever, agent, evaluate, log_file, concurrency))

    # Required columns per official repo spec
    output_cols = ["status", "product_area", "response", "justification", "request_type"]
    pd.DataFrame(results)[output_cols].to_csv(output_csv, index=False)
    typer.echo(f"\nSuccess! Results saved to {output_csv}")

if __name__ == "__main__":
    app()
