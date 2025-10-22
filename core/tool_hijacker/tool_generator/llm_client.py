"""
Simple LLM client for Ollama.
"""

import requests
from typing import Optional


class LLMClient:
    """Handles communication with Ollama API."""
    
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"
    
    def generate(self, prompt: str, max_tokens: int = 150) -> Optional[str]:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Max tokens to generate
            
        Returns:
            Generated text or None if failed
        """
        try:
            response = requests.post(
                self.generate_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                print(f"LLM error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"LLM request failed: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
