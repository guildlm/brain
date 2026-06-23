import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class IntentRouter:
    """
    Classifies user prompts and routes them to the correct Guild and Specialist.
    In a full production environment, this is backed by an LLM (e.g. Qwen3-8B).
    For this initial phase, it uses an LLM stub to simulate the classification.
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.supported_domains = ["code", "legal", "finance", "medical", "creative", "general"]

    def classify_intent(self, prompt: str) -> Dict[str, str]:
        """
        Analyzes the prompt to determine domain, language, task, and subtask.
        """
        prompt_lower = prompt.lower()
        
        # Stub logic simulating an LLM classification
        domain = "general"
        language = "unknown"
        task = "general_qa"
        
        # Extremely basic heuristics representing the LLM output for the Code Guild prototype
        if "go" in prompt_lower or "golang" in prompt_lower:
            domain = "code"
            language = "go"
            if "race condition" in prompt_lower or "bug" in prompt_lower or "fix" in prompt_lower:
                task = "bug_fix"
            elif "api" in prompt_lower or "build" in prompt_lower:
                task = "generation"
            elif "test" in prompt_lower:
                task = "testing"
            else:
                task = "review"
                
        classification = {
            "domain": domain,
            "language": language,
            "task": task,
        }
        
        logger.info(f"Classified Intent: {json.dumps(classification)}")
        return classification

    def select_specialist(self, classification: Dict[str, str]) -> str:
        """
        Maps a classification to a specific SLM specialist.
        """
        domain = classification.get("domain")
        language = classification.get("language")
        task = classification.get("task")
        
        if domain == "code":
            if language == "go":
                if task in ["bug_fix", "generation"]:
                    return "guild-code/go_generator"
                elif task == "review":
                    return "guild-code/go_reviewer"
                elif task == "testing":
                    return "guild-code/go_tester"
            
        return "brain/generalist"
