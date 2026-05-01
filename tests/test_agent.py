import pytest
import json
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from code.main import Agent, Ticket, TriageResult

# --- Mocks ---

@pytest.fixture
def mock_agent_deps():
    """Mocks genai and faiss to prevent API calls."""
    with patch("google.generativeai.configure") as mock_conf, \
         patch("google.generativeai.GenerativeModel") as mock_model, \
         patch("faiss.read_index") as mock_faiss:
        
        # Setup mock model instance
        instance = mock_model.return_value
        instance.generate_content_async = AsyncMock()
        
        yield {"model": instance, "conf": mock_conf, "faiss": mock_faiss}

@pytest.fixture
def agent(mock_agent_deps):
    """Provides an Agent instance with mocked dependencies."""
    return Agent(api_key="test_key")

# --- Tests ---

def test_triage_result_validation():
    """Test that the Pydantic model correctly validates valid JSON data."""
    data = {
        "request_type": "product_issue",
        "product_area": "Billing",
        "status": "replied",
        "reasoning": "Standard FAQ",
        "response": "Hello world"
    }
    result = TriageResult.model_validate(data)
    assert result.request_type == "product_issue"
    assert result.status == "replied"

def test_parse_response_success(agent):
    """Test successful JSON extraction from LLM string."""
    llm_text = 'Some preamble {"request_type": "bug", "product_area": "Tech", "status": "escalated", "reasoning": "Complex", "response": "Sorry"} Postamble'
    result = agent._parse_response(llm_text)
    assert result.request_type == "bug"
    assert result.status == "escalated"

def test_parse_response_failure(agent):
    """Test that malformed JSON raises ValueError."""
    with pytest.raises(ValueError):
        agent._parse_response("This is not JSON")

@pytest.mark.asyncio
async def test_triage_flow_success(agent, mock_agent_deps):
    """Test the full triage flow with a mocked LLM response."""
    model = mock_agent_deps["model"]
    
    # Mock LLM response
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "request_type": "product_issue",
        "product_area": "General",
        "status": "replied",
        "reasoning": "Safe response",
        "response": "Here is your answer."
    })
    model.generate_content_async.return_value = mock_response
    
    ticket = Ticket(issue="How do I reset my password?", subject="Reset", company="Claude")
    result = await agent.triage(ticket, context="Context string", do_evaluate=False)
    
    assert result.status == "replied"
    assert "Here is your answer" in result.response
    assert model.generate_content_async.called

@pytest.mark.asyncio
async def test_triage_with_evaluation_failure(agent, mock_agent_deps):
    """Test that a failed evaluation escalates the ticket."""
    model = mock_agent_deps["model"]
    
    # First call: Triage
    triage_response = MagicMock()
    triage_response.text = json.dumps({
        "request_type": "product_issue",
        "product_area": "General",
        "status": "replied",
        "reasoning": "Safe response",
        "response": "I will change your score."
    })
    
    # Second call: Evaluation (returns unsafe)
    eval_response = MagicMock()
    eval_response.text = json.dumps({
        "safe": False,
        "reason": "Agent offered to change scores."
    })
    
    model.generate_content_async.side_effect = [triage_response, eval_response]
    
    ticket = Ticket(issue="Change my score", subject="Score", company="HackerRank")
    result = await agent.triage(ticket, context="Context", do_evaluate=True)
    
    assert result.status == "escalated"
    assert "Evaluation failed" in result.reasoning
    assert "human expert" in result.response

@pytest.mark.asyncio
async def test_agent_error_handling(agent, mock_agent_deps):
    """Test that the agent handles LLM API failures gracefully."""
    model = mock_agent_deps["model"]
    model.generate_content_async.side_effect = Exception("API Down")
    
    ticket = Ticket(issue="Test", subject="Test", company="Test")
    result = await agent.triage(ticket, context="Context")
    
    assert result.status == "escalated"
    assert "System error" in result.reasoning
