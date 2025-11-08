"""
Model Manager for Local LLM
Handles downloading, installing, and managing local LLM models using Ollama
"""

import os
import json
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import streamlit as st


class ModelManager:
    """Manages local LLM models using Ollama"""

    # Recommended models configuration
    MODELS = {
        'llama3.1:8b': {
            'name': 'Llama 3.1 8B',
            'size': '4.7 GB',
            'ram_required': '8 GB',
            'description': 'Best quality/performance balance - Recommended',
            'quality': 5,
            'speed': 3,
            'recommended': True
        },
        'mistral:7b': {
            'name': 'Mistral 7B',
            'size': '4.1 GB',
            'ram_required': '8 GB',
            'description': 'Fast and efficient for code generation',
            'quality': 4,
            'speed': 4,
            'recommended': False
        },
        'phi3:mini': {
            'name': 'Phi-3 Mini',
            'size': '2.3 GB',
            'ram_required': '4 GB',
            'description': 'Lightweight model for quick queries',
            'quality': 3,
            'speed': 5,
            'recommended': False
        },
        'qwen2.5:7b': {
            'name': 'Qwen 2.5 7B',
            'size': '4.4 GB',
            'ram_required': '8 GB',
            'description': 'Excellent for data analysis tasks',
            'quality': 4,
            'speed': 3,
            'recommended': False
        }
    }

    def __init__(self):
        self.config_dir = Path.home() / '.eda_tool'
        self.config_file = self.config_dir / 'ai_config.json'
        self.config_dir.mkdir(exist_ok=True)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Load AI configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
                return {}
        return {}

    def _save_config(self):
        """Save AI configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

    def is_ollama_installed(self) -> bool:
        """Check if Ollama is installed on the system"""
        # First try to check if Ollama service is accessible via API
        try:
            import requests
            response = requests.get('http://localhost:11434/api/tags', timeout=3)
            if response.status_code == 200:
                return True
        except:
            pass

        # Fallback to checking command line
        try:
            result = subprocess.run(
                ['ollama', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def get_ollama_version(self) -> Optional[str]:
        """Get installed Ollama version"""
        try:
            result = subprocess.run(
                ['ollama', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except Exception:
            return None

    def list_installed_models(self) -> List[str]:
        """List all installed Ollama models"""
        # Try using API first
        try:
            import requests
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                return models
        except Exception as e:
            print(f"Error listing models via API: {e}")

        # Fallback to command line
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Skip header
                    models = []
                    for line in lines[1:]:
                        parts = line.split()
                        if parts:
                            models.append(parts[0])
                    return models
            return []
        except Exception as e:
            print(f"Error listing models: {e}")
            return []

    def is_model_installed(self, model_name: str) -> bool:
        """Check if a specific model is installed"""
        installed = self.list_installed_models()
        return model_name in installed

    def download_model(self, model_name: str, progress_callback=None) -> Tuple[bool, str]:
        """
        Download and install a model using Ollama

        Args:
            model_name: Name of the model to download (e.g., 'llama3.1:8b')
            progress_callback: Optional callback function for progress updates

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            if not self.is_ollama_installed():
                return False, "Ollama is not installed. Please install Ollama first."

            # Start download process
            process = subprocess.Popen(
                ['ollama', 'pull', model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Read output line by line
            for line in process.stdout:
                line = line.strip()
                if line and progress_callback:
                    progress_callback(line)

            # Wait for process to complete
            return_code = process.wait()

            if return_code == 0:
                # Update config
                self.config['default_model'] = model_name
                self.config['installed_models'] = self.list_installed_models()
                self._save_config()

                return True, f"Successfully downloaded {model_name}"
            else:
                stderr = process.stderr.read()
                return False, f"Failed to download model: {stderr}"

        except Exception as e:
            return False, f"Error downloading model: {str(e)}"

    def remove_model(self, model_name: str) -> Tuple[bool, str]:
        """Remove an installed model"""
        try:
            result = subprocess.run(
                ['ollama', 'rm', model_name],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                # Update config
                self.config['installed_models'] = self.list_installed_models()
                if self.config.get('default_model') == model_name:
                    self.config['default_model'] = None
                self._save_config()

                return True, f"Successfully removed {model_name}"
            else:
                return False, f"Failed to remove model: {result.stderr}"

        except Exception as e:
            return False, f"Error removing model: {str(e)}"

    def get_default_model(self) -> Optional[str]:
        """Get the default model set by user"""
        return self.config.get('default_model')

    def set_default_model(self, model_name: str):
        """Set the default model"""
        if self.is_model_installed(model_name):
            self.config['default_model'] = model_name
            self._save_config()
            return True
        return False

    def get_recommended_model(self) -> str:
        """Get the recommended model based on system RAM"""
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024**3)

            if ram_gb >= 16:
                return 'llama3.1:8b'  # Best quality
            elif ram_gb >= 8:
                return 'llama3.1:8b'  # Still recommended
            else:
                return 'phi3:mini'  # Lightweight
        except ImportError:
            return 'llama3.1:8b'  # Default recommendation

    def get_system_info(self) -> Dict:
        """Get system information relevant to LLM usage"""
        info = {
            'os': platform.system(),
            'os_version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
        }

        try:
            import psutil
            info['ram_gb'] = round(psutil.virtual_memory().total / (1024**3), 1)
            info['ram_available_gb'] = round(psutil.virtual_memory().available / (1024**3), 1)
            info['cpu_count'] = psutil.cpu_count()
        except ImportError:
            info['ram_gb'] = 'Unknown'
            info['ram_available_gb'] = 'Unknown'
            info['cpu_count'] = 'Unknown'

        return info

    def get_ollama_install_instructions(self) -> str:
        """Get platform-specific Ollama installation instructions"""
        system = platform.system()

        if system == 'Windows':
            return """
# Install Ollama on Windows

1. Download Ollama from: https://ollama.ai/download/windows
2. Run the installer (OllamaSetup.exe)
3. Follow the installation wizard
4. Restart this application after installation

Or use PowerShell:
```powershell
winget install Ollama.Ollama
```
"""
        elif system == 'Darwin':  # macOS
            return """
# Install Ollama on macOS

1. Download Ollama from: https://ollama.ai/download/mac
2. Open the downloaded .dmg file
3. Drag Ollama to Applications
4. Launch Ollama from Applications
5. Restart this application after installation

Or use Homebrew:
```bash
brew install ollama
```
"""
        else:  # Linux
            return """
# Install Ollama on Linux

Run this command in terminal:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

Or download from: https://ollama.ai/download/linux

After installation, restart this application.
"""


def get_available_models() -> Dict:
    """Get dictionary of available models"""
    return ModelManager.MODELS


def download_model(model_name: str, progress_callback=None) -> Tuple[bool, str]:
    """
    Convenience function to download a model

    Args:
        model_name: Model to download
        progress_callback: Optional callback for progress updates

    Returns:
        Tuple of (success, message)
    """
    manager = ModelManager()
    return manager.download_model(model_name, progress_callback)
