"""
Local LLM Integration Layer
Handles communication with Ollama and manages AI responses
"""

import subprocess
import json
import time
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass
from datetime import datetime


@dataclass
class AIResponse:
    """Structure for AI responses"""
    content: str
    model: str
    timestamp: datetime
    tokens_used: int
    duration_ms: float
    success: bool
    error: Optional[str] = None


class LocalLLM:
    """Local LLM interface using Ollama"""

    def __init__(self, model_name: str = 'llama3.1:8b', temperature: float = 0.7):
        """
        Initialize Local LLM

        Args:
            model_name: Name of the Ollama model to use
            temperature: Sampling temperature (0.0 to 1.0)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = 10  # Keep last 10 exchanges

    def is_available(self) -> bool:
        """Check if Ollama service is running"""
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> AIResponse:
        """
        Generate response from local LLM

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt for context
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response

        Returns:
            AIResponse object with generated text and metadata
        """
        start_time = time.time()

        try:
            # Build the request
            request_data = {
                'model': self.model_name,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': self.temperature,
                    'num_predict': max_tokens
                }
            }

            if system_prompt:
                request_data['system'] = system_prompt

            # Call Ollama API via subprocess
            process = subprocess.Popen(
                ['ollama', 'run', self.model_name],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Send prompt and get response
            stdout, stderr = process.communicate(input=prompt, timeout=120)

            duration_ms = (time.time() - start_time) * 1000

            if process.returncode == 0:
                return AIResponse(
                    content=stdout.strip(),
                    model=self.model_name,
                    timestamp=datetime.now(),
                    tokens_used=0,  # Ollama doesn't provide this easily
                    duration_ms=duration_ms,
                    success=True
                )
            else:
                return AIResponse(
                    content='',
                    model=self.model_name,
                    timestamp=datetime.now(),
                    tokens_used=0,
                    duration_ms=duration_ms,
                    success=False,
                    error=stderr.strip()
                )

        except subprocess.TimeoutExpired:
            return AIResponse(
                content='',
                model=self.model_name,
                timestamp=datetime.now(),
                tokens_used=0,
                duration_ms=(time.time() - start_time) * 1000,
                success=False,
                error='Request timed out after 120 seconds'
            )
        except Exception as e:
            return AIResponse(
                content='',
                model=self.model_name,
                timestamp=datetime.now(),
                tokens_used=0,
                duration_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e)
            )

    def generate_with_api(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000
    ) -> AIResponse:
        """
        Generate response using Ollama's HTTP API (better method)

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate

        Returns:
            AIResponse object
        """
        start_time = time.time()

        try:
            import requests

            # Ollama API endpoint
            url = 'http://localhost:11434/api/generate'

            payload = {
                'model': self.model_name,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': self.temperature,
                    'num_predict': max_tokens
                }
            }

            if system_prompt:
                payload['system'] = system_prompt

            response = requests.post(url, json=payload, timeout=120)

            duration_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                result = response.json()

                return AIResponse(
                    content=result.get('response', '').strip(),
                    model=self.model_name,
                    timestamp=datetime.now(),
                    tokens_used=result.get('eval_count', 0),
                    duration_ms=duration_ms,
                    success=True
                )
            else:
                return AIResponse(
                    content='',
                    model=self.model_name,
                    timestamp=datetime.now(),
                    tokens_used=0,
                    duration_ms=duration_ms,
                    success=False,
                    error=f'API error: {response.status_code} - {response.text}'
                )

        except ImportError:
            # Fallback to subprocess method if requests not available
            return self.generate(prompt, system_prompt, max_tokens)

        except Exception as e:
            return AIResponse(
                content='',
                model=self.model_name,
                timestamp=datetime.now(),
                tokens_used=0,
                duration_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e)
            )

    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        include_history: bool = True
    ) -> AIResponse:
        """
        Chat with the LLM maintaining conversation history

        Args:
            user_message: User's message
            system_prompt: Optional system context
            include_history: Whether to include conversation history

        Returns:
            AIResponse object
        """
        # Build prompt with history
        if include_history and self.conversation_history:
            context = "\n\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in self.conversation_history[-self.max_history:]
            ])
            full_prompt = f"{context}\n\nUser: {user_message}\n\nAssistant:"
        else:
            full_prompt = user_message

        # Generate response
        response = self.generate_with_api(full_prompt, system_prompt)

        # Update history if successful
        if response.success:
            self.conversation_history.append({
                'role': 'user',
                'content': user_message
            })
            self.conversation_history.append({
                'role': 'assistant',
                'content': response.content
            })

            # Trim history if too long
            if len(self.conversation_history) > self.max_history * 2:
                self.conversation_history = self.conversation_history[-self.max_history * 2:]

        return response

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversation_history.copy()

    def set_temperature(self, temperature: float):
        """Update temperature setting"""
        self.temperature = max(0.0, min(1.0, temperature))

    def set_model(self, model_name: str):
        """Switch to a different model"""
        self.model_name = model_name
        self.clear_history()  # Clear history when changing models


def get_ai_response(
    prompt: str,
    model_name: str = 'llama3.1:8b',
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000
) -> str:
    """
    Convenience function to get a quick AI response

    Args:
        prompt: User prompt
        model_name: Model to use
        system_prompt: Optional system context
        temperature: Sampling temperature
        max_tokens: Max tokens to generate

    Returns:
        Generated text or error message
    """
    llm = LocalLLM(model_name, temperature)

    if not llm.is_available():
        return "Error: Ollama service is not running. Please start Ollama first."

    response = llm.generate_with_api(prompt, system_prompt, max_tokens)

    if response.success:
        return response.content
    else:
        return f"Error: {response.error}"


def test_ollama_connection() -> Tuple[bool, str]:
    """
    Test connection to Ollama service

    Returns:
        Tuple of (is_working: bool, message: str)
    """
    try:
        import requests

        url = 'http://localhost:11434/api/tags'
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            models = response.json().get('models', [])
            if models:
                model_names = [m['name'] for m in models]
                return True, f"Ollama is running. Available models: {', '.join(model_names)}"
            else:
                return True, "Ollama is running but no models are installed."
        else:
            return False, f"Ollama API returned status code: {response.status_code}"

    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to Ollama. Please make sure Ollama is running."
    except ImportError:
        # Try subprocess method
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return True, "Ollama is installed and working."
            else:
                return False, "Ollama command failed."
        except Exception as e:
            return False, f"Error: {str(e)}"
    except Exception as e:
        return False, f"Error testing Ollama: {str(e)}"
