"""
AI Chat Assistant
Interactive chat interface for data analysis queries
"""

import streamlit as st
from datetime import datetime
from typing import List, Dict, Optional
from .llm_integration import LocalLLM, AIResponse
from .context_builder import build_dataset_context, build_analysis_context
from .prompts import CHAT_SYSTEM_PROMPT


class ChatAssistant:
    """AI Chat Assistant for data analysis"""

    def __init__(self, model_name: str = 'phi3:mini', temperature: float = 0.7):
        """
        Initialize Chat Assistant

        Args:
            model_name: Name of the LLM model to use
            temperature: Sampling temperature for responses
        """
        self.llm = LocalLLM(model_name, temperature)
        self.conversation_history: List[Dict[str, str]] = []

    def is_available(self) -> bool:
        """Check if AI is available"""
        return self.llm.is_available()

    def chat(
        self,
        user_message: str,
        dataset_context: Optional[str] = None,
        analysis_context: Optional[str] = None
    ) -> AIResponse:
        """
        Send a message to the AI and get a response

        Args:
            user_message: User's question or message
            dataset_context: Context about the dataset
            analysis_context: Context about analysis results

        Returns:
            AIResponse object with AI's reply
        """
        # Build enhanced system prompt with context
        system_prompt = CHAT_SYSTEM_PROMPT

        if dataset_context:
            system_prompt += f"\n\n## Current Dataset Context:\n{dataset_context}"

        if analysis_context:
            system_prompt += f"\n\n## Analysis Results:\n{analysis_context}"

        # Get response from LLM
        response = self.llm.chat(
            user_message=user_message,
            system_prompt=system_prompt,
            include_history=True
        )

        return response

    def clear_history(self):
        """Clear conversation history"""
        self.llm.clear_history()
        self.conversation_history = []

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.llm.get_history()


def display_ai_chat():
    """Display AI Chat interface in Streamlit"""

    st.markdown('<div class="section-header">💬 AI Chat Assistant</div>', unsafe_allow_html=True)

    # Check if AI is available
    from .model_manager import ModelManager
    manager = ModelManager()

    if not manager.is_ollama_installed():
        st.warning("⚠️ Ollama is not installed. Please set up AI features first.")
        if st.button("🤖 Go to AI Setup"):
            st.session_state.current_section = "ai_setup"
            st.rerun()
        return

    default_model = manager.get_default_model()
    if not default_model:
        st.warning("⚠️ No AI model is configured. Please set up a model first.")
        if st.button("🤖 Go to AI Setup"):
            st.session_state.current_section = "ai_setup"
            st.rerun()
        return

    # Initialize chat assistant in session state
    if 'chat_assistant' not in st.session_state:
        st.session_state.chat_assistant = ChatAssistant(
            model_name=default_model,
            temperature=0.7
        )

    # Initialize chat messages in session state
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []

    # Display welcome message if no messages yet
    if len(st.session_state.chat_messages) == 0:
        st.info("""
        👋 **Welcome to AI Chat Assistant!**

        I can help you with:
        - Understanding your data
        - Explaining analysis results
        - Suggesting data cleaning steps
        - Recommending feature engineering
        - Generating code
        - Answering questions about statistics

        💡 **Tip:** Upload a dataset first to get context-aware assistance!
        """)

    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "timestamp" in message:
                st.caption(f"_{message['timestamp']}_")

    # Chat input
    if prompt := st.chat_input("Ask me anything about your data..."):
        # Add user message to chat
        timestamp = datetime.now().strftime("%I:%M %p")
        st.session_state.chat_messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": timestamp
        })

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption(f"_{timestamp}_")

        # Build context if data is loaded
        dataset_context = None
        analysis_context = None

        if st.session_state.get('data_loaded', False):
            # Build dataset context
            dataset_context = f"Dataset: {st.session_state.filename}\n\n"
            dataset_context += build_dataset_context(
                df=st.session_state.df,
                max_rows_sample=5,
                include_sample=True
            )

            # Build analysis context if available
            if st.session_state.get('quality_report') or st.session_state.get('eda_report'):
                analysis_context = build_analysis_context(
                    quality_report=st.session_state.get('quality_report'),
                    eda_report=st.session_state.get('eda_report'),
                    target_analysis=st.session_state.get('target_analysis')
                )

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chat_assistant.chat(
                        user_message=prompt,
                        dataset_context=dataset_context,
                        analysis_context=analysis_context
                    )

                    if response.success:
                        ai_message = response.content
                        timestamp = datetime.now().strftime("%I:%M %p")

                        st.markdown(ai_message)
                        st.caption(f"_{timestamp} • {response.duration_ms:.0f}ms_")

                        # Add assistant message to chat
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": ai_message,
                            "timestamp": timestamp,
                            "duration_ms": response.duration_ms
                        })
                    else:
                        error_msg = f"❌ Error: {response.error}"
                        st.error(error_msg)
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "timestamp": datetime.now().strftime("%I:%M %p")
                        })

                except Exception as e:
                    error_msg = f"❌ Error communicating with AI: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": datetime.now().strftime("%I:%M %p")
                    })

    # Sidebar controls
    with st.sidebar:
        st.markdown("---")
        st.subheader("💬 Chat Controls")

        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_messages = []
            if 'chat_assistant' in st.session_state:
                st.session_state.chat_assistant.clear_history()
            st.success("Chat history cleared!")
            st.rerun()

        st.caption(f"Messages: {len(st.session_state.chat_messages)}")

        # Display current model
        st.info(f"🤖 Model: {default_model}")


def display_ai_insights():
    """Display auto-generated AI insights about the data"""

    st.markdown('<div class="section-header">🧠 AI-Generated Insights</div>', unsafe_allow_html=True)

    # Check if AI is available
    from .model_manager import ModelManager
    manager = ModelManager()

    if not manager.is_ollama_installed() or not manager.get_default_model():
        st.warning("⚠️ AI features not configured. Please set up AI first.")
        if st.button("🤖 Go to AI Setup"):
            st.session_state.current_section = "ai_setup"
            st.rerun()
        return

    # Check if data is loaded
    if not st.session_state.get('data_loaded', False):
        st.info("📁 Please upload a dataset first to generate insights.")
        return

    st.info("🚀 **Auto-Generated Insights** - Coming in the next phase!")
    st.markdown("""
    This feature will automatically analyze your data and provide:
    - Key findings and patterns
    - Potential data quality issues
    - Recommendations for preprocessing
    - Feature engineering suggestions
    - Model selection advice

    **Status:** Implementation in progress...
    """)
