"""
AI-Powered Insights Generator
Automatically analyzes data quality and generates actionable insights
"""

import streamlit as st
from typing import Dict, Any, Optional, List
from datetime import datetime
from .llm_integration import LocalLLM, AIResponse
from .context_builder import build_analysis_context
from .prompts import format_insight_prompt


class InsightsGenerator:
    """Generate AI-powered insights from data analysis"""

    def __init__(self, model_name: str = 'phi3:mini', temperature: float = 0.7):
        """
        Initialize Insights Generator

        Args:
            model_name: Name of the LLM model to use
            temperature: Sampling temperature for responses
        """
        self.llm = LocalLLM(model_name, temperature)

    def is_available(self) -> bool:
        """Check if AI is available"""
        return self.llm.is_available()

    def generate_insights(
        self,
        quality_report: Optional[Dict] = None,
        eda_report: Optional[Dict] = None,
        target_analysis: Optional[Dict] = None,
        dataset_info: Optional[Dict] = None
    ) -> AIResponse:
        """
        Generate insights from analysis reports

        Args:
            quality_report: Data quality report dictionary
            eda_report: EDA analysis report dictionary
            target_analysis: Target variable analysis dictionary
            dataset_info: Basic dataset information (name, shape, etc.)

        Returns:
            AIResponse object with generated insights
        """
        # Build context from available reports
        context = build_analysis_context(
            quality_report=quality_report,
            eda_report=eda_report,
            target_analysis=target_analysis
        )

        # Add dataset basic info if available
        if dataset_info:
            context = f"## Dataset Overview\n{self._format_dataset_info(dataset_info)}\n\n{context}"

        # Format the prompt
        prompt = format_insight_prompt(context)

        # Generate insights using LLM
        response = self.llm.generate(
            prompt=prompt,
            system_prompt="You are an expert data scientist analyzing datasets for ML readiness and data quality.",
            max_tokens=1000
        )

        return response

    def _format_dataset_info(self, dataset_info: Dict) -> str:
        """Format dataset basic information"""
        info_parts = []

        if 'name' in dataset_info:
            info_parts.append(f"**Dataset:** {dataset_info['name']}")

        if 'rows' in dataset_info and 'columns' in dataset_info:
            info_parts.append(f"**Shape:** {dataset_info['rows']} rows × {dataset_info['columns']} columns")

        if 'memory_mb' in dataset_info:
            info_parts.append(f"**Memory:** {dataset_info['memory_mb']:.2f} MB")

        return "\n".join(info_parts)


def generate_insights_cached(
    quality_report: Optional[Dict] = None,
    eda_report: Optional[Dict] = None,
    target_analysis: Optional[Dict] = None,
    dataset_info: Optional[Dict] = None,
    model_name: str = 'phi3:mini',
    force_refresh: bool = False
) -> Optional[AIResponse]:
    """
    Generate insights with caching to avoid redundant LLM calls

    Args:
        quality_report: Data quality report dictionary
        eda_report: EDA analysis report dictionary
        target_analysis: Target variable analysis dictionary
        dataset_info: Basic dataset information
        model_name: Model to use for generation
        force_refresh: Force regeneration even if cached

    Returns:
        AIResponse object with insights or None if error
    """
    # Initialize session state for caching
    if 'ai_insights_cache' not in st.session_state:
        st.session_state.ai_insights_cache = {}

    # Create cache key based on dataset and reports
    cache_key = _create_cache_key(dataset_info, quality_report, eda_report, target_analysis)

    # Check cache if not forcing refresh
    if not force_refresh and cache_key in st.session_state.ai_insights_cache:
        return st.session_state.ai_insights_cache[cache_key]

    # Generate new insights
    try:
        generator = InsightsGenerator(model_name=model_name)

        if not generator.is_available():
            return None

        response = generator.generate_insights(
            quality_report=quality_report,
            eda_report=eda_report,
            target_analysis=target_analysis,
            dataset_info=dataset_info
        )

        # Cache the result if successful
        if response.success:
            st.session_state.ai_insights_cache[cache_key] = response

        return response

    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")
        return None


def _create_cache_key(
    dataset_info: Optional[Dict],
    quality_report: Optional[Dict],
    eda_report: Optional[Dict],
    target_analysis: Optional[Dict]
) -> str:
    """Create a cache key based on the analysis inputs"""
    key_parts = []

    if dataset_info and 'name' in dataset_info:
        key_parts.append(f"ds:{dataset_info['name']}")

    if quality_report:
        key_parts.append("qr:yes")

    if eda_report:
        key_parts.append("eda:yes")

    if target_analysis:
        key_parts.append("ta:yes")

    return "_".join(key_parts) if key_parts else "default"


def clear_insights_cache():
    """Clear the insights cache"""
    if 'ai_insights_cache' in st.session_state:
        st.session_state.ai_insights_cache = {}
