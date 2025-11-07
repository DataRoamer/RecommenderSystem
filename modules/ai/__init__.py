"""
AI Module for EDA Tool
Handles local LLM integration, model management, and AI-powered features
"""

from .model_manager import ModelManager, get_available_models, download_model
from .llm_integration import LocalLLM, get_ai_response, test_ollama_connection
from .context_builder import (
    build_dataset_context,
    build_analysis_context,
    build_code_generation_context,
    build_insight_context
)
from .prompts import (
    CHAT_SYSTEM_PROMPT,
    INSIGHT_GENERATION_PROMPT,
    CODE_GENERATION_PROMPT,
    DATA_QUALITY_PROMPT,
    format_chat_prompt,
    format_insight_prompt,
    format_code_prompt
)
from .ui_components import (
    display_ai_setup_wizard,
    display_model_settings,
    display_ai_status_badge,
    check_ai_prerequisites,
    display_ai_feature_guard
)
from .chat_assistant import (
    ChatAssistant,
    display_ai_chat,
    display_ai_insights,
    display_nl_query_translator
)
from .insights_generator import (
    InsightsGenerator,
    generate_insights_cached,
    clear_insights_cache
)
from .nl_query_translator import (
    NLQueryTranslator,
    execute_nl_query,
    add_query_to_history,
    get_query_history,
    clear_query_history
)
from .data_cleaning_advisor import (
    DataCleaningAdvisor,
    CleaningIssue,
    CleaningRecommendation,
    display_data_cleaning,
    get_cleaning_history,
    clear_cleaning_history
)

__all__ = [
    # Model Management
    'ModelManager',
    'get_available_models',
    'download_model',
    # LLM Integration
    'LocalLLM',
    'get_ai_response',
    'test_ollama_connection',
    # Context Building
    'build_dataset_context',
    'build_analysis_context',
    'build_code_generation_context',
    'build_insight_context',
    # Prompts
    'CHAT_SYSTEM_PROMPT',
    'INSIGHT_GENERATION_PROMPT',
    'CODE_GENERATION_PROMPT',
    'DATA_QUALITY_PROMPT',
    'format_chat_prompt',
    'format_insight_prompt',
    'format_code_prompt',
    # UI Components
    'display_ai_setup_wizard',
    'display_model_settings',
    'display_ai_status_badge',
    'check_ai_prerequisites',
    'display_ai_feature_guard',
    # Chat Assistant
    'ChatAssistant',
    'display_ai_chat',
    'display_ai_insights',
    'display_nl_query_translator',
    # Insights Generator
    'InsightsGenerator',
    'generate_insights_cached',
    'clear_insights_cache',
    # NL Query Translator
    'NLQueryTranslator',
    'execute_nl_query',
    'add_query_to_history',
    'get_query_history',
    'clear_query_history',
    # Data Cleaning Advisor
    'DataCleaningAdvisor',
    'CleaningIssue',
    'CleaningRecommendation',
    'display_data_cleaning',
    'get_cleaning_history',
    'clear_cleaning_history'
]
