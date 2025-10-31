"""
AI Module for EDA Tool
Handles local LLM integration, model management, and AI-powered features
"""

from .model_manager import ModelManager, get_available_models, download_model
from .llm_integration import LocalLLM, get_ai_response
from .context_builder import build_dataset_context, build_analysis_context
from .prompts import (
    CHAT_SYSTEM_PROMPT,
    INSIGHT_GENERATION_PROMPT,
    CODE_GENERATION_PROMPT,
    DATA_QUALITY_PROMPT
)

__all__ = [
    'ModelManager',
    'get_available_models',
    'download_model',
    'LocalLLM',
    'get_ai_response',
    'build_dataset_context',
    'build_analysis_context',
    'CHAT_SYSTEM_PROMPT',
    'INSIGHT_GENERATION_PROMPT',
    'CODE_GENERATION_PROMPT',
    'DATA_QUALITY_PROMPT'
]
