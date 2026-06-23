import logging
import os
from openai import OpenAI
from src.core.router import IntentRouter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BrainOrchestrator:
    """
    The central coordinator of the GuildLM system.
    Receives user requests, delegates to the IntentRouter, and orchestrates execution across Guilds.
    """
    
    def __init__(self, ollama_url: str = "http://localhost:11434/v1"):
        self.router = IntentRouter()
        # Initialize OpenAI client to point to local Ollama instance
        self.client = OpenAI(
            base_url=ollama_url,
            api_key="ollama" # required but ignored
        )
        # Assuming Qwen2.5-7B is our base model in Ollama. 
        # When LoRA is integrated natively in Ollama/vLLM, the model name would include the adapter.
        self.base_model = "qwen2.5:7b-instruct"
        
    def execute_request(self, user_prompt: str) -> str:
        """
        Executes an end-to-end user request via real inference.
        """
        logger.info(f"Received Request: '{user_prompt}'")
        
        # 1. Classify Intent
        classification = self.router.classify_intent(user_prompt)
        
        # 2. Select Specialist
        specialist_id = self.router.select_specialist(classification)
        logger.info(f"Routing to Specialist: {specialist_id}")
        
        # 3. Load & Run Specialist via local API
        response = self._run_specialist(specialist_id, user_prompt)
        
        return response
        
    def _run_specialist(self, specialist_id: str, prompt: str) -> str:
        """
        Calls the local LLM via OpenAI API spec with the specialized system prompt.
        """
        logger.info(f"Calling Local Inference Engine for: {specialist_id}...")
        
        system_prompts = {
            "guild-code/go_generator": "You are an expert Go programmer (GuildLM Code Guild). Write idiomatic, secure Go code without markdown explanations.",
            "guild-code/go_reviewer": "You are an expert Go reviewer (GuildLM Code Guild). Review the code for race conditions, leaks, and security flaws.",
            "brain/generalist": "You are the GuildLM Brain Orchestrator. You are a generalist assistant."
        }
        
        sys_prompt = system_prompts.get(specialist_id, system_prompts["brain/generalist"])
        
        try:
            response = self.client.chat.completions.create(
                model=self.base_model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2048,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return f"Error: Could not connect to inference backend. Is Ollama running on port 11434? Details: {str(e)}"

# Example Usage
if __name__ == "__main__":
    brain = BrainOrchestrator()
    res = brain.execute_request("Can you build a high performance HTTP API in Go?")
    print(f"\\nResponse:\\n{res}\\n")
