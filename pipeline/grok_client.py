"""Real teacher API client supporting multiple providers.

Supports:
- Grok/xAI API
- OpenAI API
- Anthropic Claude API
- Local models via Ollama
"""
import os
import json
import logging
import requests
from typing import Optional, Dict, Any
import time

# Import fallback for when API is not available
from . import generate_samples

logging.basicConfig(level=logging.INFO)

class TeacherAPIClient:
    """Unified client for multiple teacher model APIs."""
    
    def __init__(self, api_key: str = None, provider: str = None, model: str = None):
        self.api_key = api_key or os.environ.get('TEACHER_API_KEY')
        self.provider = provider or os.environ.get('TEACHER_MODEL', '').split('/')[0].lower()
        self.model = model or os.environ.get('TEACHER_MODEL', 'grok-beta')
        
        # Rate limiting
        self.last_call_time = 0
        self.min_interval = 1.0  # seconds between calls
        
        # Validate configuration
        if not self.api_key and self.provider not in ['local', 'ollama']:
            logging.warning(f"No API key for {self.provider}, falling back to synthetic responses")
            self.provider = 'synthetic'
    
    def _rate_limit(self):
        """Apply rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        if time_since_last < self.min_interval:
            time.sleep(self.min_interval - time_since_last)
        self.last_call_time = time.time()
    
    def _call_grok_api(self, prompt: str, code: str) -> str:
        """Call Grok/xAI API."""
        try:
            self._rate_limit()
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert programming assistant. Provide clear, helpful code examples and explanations."
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nCode to analyze:\n```\n{code}\n```"
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logging.error(f"Grok API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logging.error(f"Grok API call failed: {e}")
            return None
    
    def _call_openai_api(self, prompt: str, code: str) -> str:
        """Call OpenAI API."""
        try:
            self._rate_limit()
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model or "gpt-4",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert programming assistant. Provide clear, helpful code examples and explanations."
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nCode to analyze:\n```\n{code}\n```"
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logging.error(f"OpenAI API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logging.error(f"OpenAI API call failed: {e}")
            return None
    
    def _call_anthropic_api(self, prompt: str, code: str) -> str:
        """Call Anthropic Claude API."""
        try:
            self._rate_limit()
            
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": self.model or "claude-3-sonnet-20240229",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nCode to analyze:\n```\n{code}\n```"
                    }
                ]
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['content'][0]['text']
            else:
                logging.error(f"Anthropic API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logging.error(f"Anthropic API call failed: {e}")
            return None
    
    def _call_ollama_api(self, prompt: str, code: str) -> str:
        """Call local Ollama API."""
        try:
            self._rate_limit()
            
            data = {
                "model": self.model or "codellama",
                "prompt": f"{prompt}\n\nCode to analyze:\n```\n{code}\n```",
                "stream": False
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                logging.error(f"Ollama API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logging.error(f"Ollama API call failed: {e}")
            return None
    
    def generate(self, prompt: str, code: str) -> str:
        """Generate response using configured teacher API."""
        # Try real API first
        if self.provider == 'grok' or 'grok' in self.provider:
            response = self._call_grok_api(prompt, code)
            if response:
                return response
        
        elif self.provider == 'openai' or 'gpt' in self.provider:
            response = self._call_openai_api(prompt, code)
            if response:
                return response
        
        elif self.provider == 'anthropic' or 'claude' in self.provider:
            response = self._call_anthropic_api(prompt, code)
            if response:
                return response
        
        elif self.provider in ['local', 'ollama']:
            response = self._call_ollama_api(prompt, code)
            if response:
                return response
        
        # Fallback to synthetic responses
        logging.warning(f"API call failed for {self.provider}, using synthetic response")
        return generate_samples.send_to_teacher(prompt, code)

class GrokClient(TeacherAPIClient):
    """Backward compatible GrokClient."""
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key, provider='grok')

def get_client() -> TeacherAPIClient:
    """Get configured teacher API client."""
    return TeacherAPIClient()

def test_api_connection(client: TeacherAPIClient) -> bool:
    """Test API connection with a simple request."""
    try:
        response = client.generate(
            "What is 2 + 2?",
            "Simple math test"
        )
        return response is not None and len(response.strip()) > 0
    except Exception as e:
        logging.error(f"API connection test failed: {e}")
        return False

if __name__ == '__main__':
    # Test the client
    client = get_client()
    print(f"Testing {client.provider} API...")
    
    if test_api_connection(client):
        print("✅ API connection successful")
        
        # Test with code
        test_code = "def hello(): print('Hello, World!')"
        response = client.generate("Explain this function", test_code)
        print(f"Response: {response[:200]}...")
    else:
        print("❌ API connection failed, using synthetic responses")
