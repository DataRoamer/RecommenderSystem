"""
Natural Language Query Translator
Converts natural language queries to executable pandas code
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import sys
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from .llm_integration import LocalLLM, AIResponse
from .context_builder import build_dataset_context
from .prompts import get_nl_query_prompt
import re


class NLQueryTranslator:
    """Translate natural language queries to pandas code and execute safely"""

    def __init__(self, model_name: str = 'phi3:mini', temperature: float = 0.3):
        """
        Initialize NL Query Translator

        Args:
            model_name: Name of the LLM model to use
            temperature: Sampling temperature (lower = more deterministic)
        """
        self.llm = LocalLLM(model_name, temperature)

    def is_available(self) -> bool:
        """Check if AI is available"""
        return self.llm.is_available()

    def translate_query(
        self,
        query: str,
        df: pd.DataFrame,
        dataset_name: str = "dataset"
    ) -> AIResponse:
        """
        Translate natural language query to pandas code

        Args:
            query: Natural language query
            df: DataFrame to operate on
            dataset_name: Name of the dataset

        Returns:
            AIResponse with generated code
        """
        # Build context about the dataset
        context = build_dataset_context(
            df=df,
            max_rows_sample=5,
            include_sample=True
        )

        # Add dataset name
        context = f"Dataset: {dataset_name}\n\n{context}"

        # Get the prompt
        prompt = get_nl_query_prompt(query=query, context=context)

        # Generate code
        response = self.llm.generate(
            prompt=prompt,
            system_prompt="You are an expert Python/pandas code generator. Generate clean, safe, executable code.",
            max_tokens=500
        )

        return response

    def execute_code_safely(
        self,
        code: str,
        df: pd.DataFrame
    ) -> Tuple[bool, Any, str]:
        """
        Execute generated code safely in a restricted environment

        Args:
            code: Python code to execute
            df: DataFrame to operate on

        Returns:
            Tuple of (success, result, error_message)
        """
        # Extract code from markdown if present
        code = self._extract_code_from_markdown(code)

        # Validate code for dangerous operations
        if not self._is_code_safe(code):
            return False, None, "Code contains potentially dangerous operations"

        # Create a safe execution environment
        safe_globals = {
            'pd': pd,
            'np': np,
            'df': df.copy(),  # Work on a copy to prevent modifications
            '__builtins__': {
                'len': len,
                'range': range,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'abs': abs,
                'max': max,
                'min': min,
                'sum': sum,
                'round': round,
                'sorted': sorted,
                'enumerate': enumerate,
                'zip': zip,
                'print': print,
            }
        }

        safe_locals = {}

        # Capture output
        output_buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = output_buffer

        try:
            # Execute the code
            exec(code, safe_globals, safe_locals)

            # Get the output
            output = output_buffer.getvalue()
            sys.stdout = old_stdout

            # Try to get the result from locals
            result = safe_locals.get('result', None)

            # If no result variable, try to get the last expression
            if result is None and output:
                result = output

            return True, result, ""

        except Exception as e:
            sys.stdout = old_stdout
            return False, None, str(e)

    def _extract_code_from_markdown(self, text: str) -> str:
        """Extract code from markdown code blocks"""
        # Look for ```python ... ``` blocks
        pattern = r'```python\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Look for ``` ... ``` blocks
        pattern = r'```\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            return matches[0].strip()

        # No markdown, return as is
        return text.strip()

    def _is_code_safe(self, code: str) -> bool:
        """
        Check if code is safe to execute

        Args:
            code: Code to validate

        Returns:
            True if safe, False otherwise
        """
        # List of dangerous keywords/operations
        dangerous_patterns = [
            r'\bimport\s+os\b',
            r'\bimport\s+sys\b',
            r'\bimport\s+subprocess\b',
            r'\b__import__\b',
            r'\beval\b',
            r'\bexec\b',
            r'\bcompile\b',
            r'\bopen\b',
            r'\bfile\b',
            r'\bwrite\b',
            r'\bdelete\b',
            r'\bremove\b',
            r'\brm\b',
            r'\bdrop\s+table\b',
            r'\btruncate\b',
            r'globals\(\)',
            r'locals\(\)',
            r'__',  # Dunder methods
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False

        return True


def execute_nl_query(
    query: str,
    df: pd.DataFrame,
    dataset_name: str = "dataset",
    model_name: str = 'phi3:mini'
) -> Dict[str, Any]:
    """
    Execute a natural language query on a DataFrame

    Args:
        query: Natural language query
        df: DataFrame to operate on
        dataset_name: Name of the dataset
        model_name: Model to use for code generation

    Returns:
        Dictionary with results
    """
    translator = NLQueryTranslator(model_name=model_name)

    if not translator.is_available():
        return {
            'success': False,
            'error': 'AI not available',
            'code': None,
            'result': None,
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'generation_time_ms': 0
        }

    # Translate query to code
    response = translator.translate_query(
        query=query,
        df=df,
        dataset_name=dataset_name
    )

    if not response.success:
        return {
            'success': False,
            'error': f'Code generation failed: {response.error}',
            'code': None,
            'result': None,
            'generation_time_ms': response.duration_ms,
            'query': query,
            'timestamp': datetime.now().isoformat()
        }

    # Execute the code
    success, result, error = translator.execute_code_safely(
        code=response.content,
        df=df
    )

    return {
        'success': success,
        'error': error if not success else None,
        'code': response.content,
        'result': result,
        'generation_time_ms': response.duration_ms,
        'query': query,
        'timestamp': datetime.now().isoformat()
    }


def add_query_to_history(query_result: Dict[str, Any]):
    """Add a query result to session history"""
    if 'nl_query_history' not in st.session_state:
        st.session_state.nl_query_history = []

    st.session_state.nl_query_history.append(query_result)

    # Limit history to last 20 queries
    if len(st.session_state.nl_query_history) > 20:
        st.session_state.nl_query_history = st.session_state.nl_query_history[-20:]


def get_query_history() -> List[Dict[str, Any]]:
    """Get query history from session state"""
    return st.session_state.get('nl_query_history', [])


def clear_query_history():
    """Clear query history"""
    if 'nl_query_history' in st.session_state:
        st.session_state.nl_query_history = []
