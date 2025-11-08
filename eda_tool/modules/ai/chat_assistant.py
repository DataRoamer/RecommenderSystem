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
    from .insights_generator import generate_insights_cached, clear_insights_cache

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

    # Get default model
    default_model = manager.get_default_model()

    # Display info about the feature
    st.markdown("""
    **Auto-Generated Insights** uses AI to analyze your data and provide:
    - 🔍 Key findings and patterns
    - ⚠️ Potential data quality issues
    - 🛠️ Recommendations for preprocessing
    - ✨ Feature engineering suggestions
    - 🎯 ML readiness assessment
    """)

    # Refresh button
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("🔄 Regenerate", help="Generate fresh insights"):
            clear_insights_cache()
            st.rerun()
    with col2:
        auto_generate = st.checkbox("Auto-generate", value=True, help="Automatically generate insights when data is loaded")

    st.markdown("---")

    # Gather available reports
    quality_report = st.session_state.get('quality_report')
    eda_report = st.session_state.get('eda_report')
    target_analysis = st.session_state.get('target_analysis')

    # Check if any analysis is available
    if not quality_report and not eda_report:
        st.warning("⚠️ No analysis reports available yet. Please run Data Quality or EDA Analysis first.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Go to Data Quality"):
                st.session_state.current_section = "data_quality"
                st.rerun()
        with col2:
            if st.button("📈 Go to EDA"):
                st.session_state.current_section = "eda"
                st.rerun()
        return

    # Prepare dataset info
    dataset_info = {
        'name': st.session_state.get('filename', 'Unknown'),
        'rows': len(st.session_state.df) if st.session_state.get('df') is not None else 0,
        'columns': len(st.session_state.df.columns) if st.session_state.get('df') is not None else 0,
        'memory_mb': st.session_state.df.memory_usage(deep=True).sum() / (1024 * 1024) if st.session_state.get('df') is not None else 0
    }

    # Generate insights
    with st.spinner("🤖 Analyzing your data and generating insights..."):
        response = generate_insights_cached(
            quality_report=quality_report,
            eda_report=eda_report,
            target_analysis=target_analysis,
            dataset_info=dataset_info,
            model_name=default_model,
            force_refresh=False
        )

    # Display results
    if response is None:
        st.error("❌ Failed to generate insights. Please check if Ollama is running and the model is available.")
        return

    if not response.success:
        st.error(f"❌ Error generating insights: {response.error}")
        return

    # Display insights
    st.markdown("### 📊 Analysis Results")
    st.success("✅ Insights generated successfully!")

    # Display the insights in a nice format
    st.markdown(response.content)

    # Display metadata
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Generation Time", f"{response.duration_ms:.0f}ms")
    with col2:
        st.metric("Model Used", default_model)
    with col3:
        timestamp = datetime.now().strftime("%I:%M %p")
        st.metric("Generated At", timestamp)

    # Additional options
    with st.expander("📋 Analysis Details"):
        st.markdown("**Data Sources Used:**")
        sources = []
        if quality_report:
            sources.append("✅ Data Quality Report")
        if eda_report:
            sources.append("✅ EDA Analysis Report")
        if target_analysis:
            sources.append("✅ Target Variable Analysis")

        for source in sources:
            st.markdown(f"- {source}")

        st.markdown(f"\n**Dataset:** {dataset_info['name']}")
        st.markdown(f"**Shape:** {dataset_info['rows']} rows × {dataset_info['columns']} columns")


def display_nl_query_translator():
    """Display Natural Language Query Translator interface"""

    st.markdown('<div class="section-header">🔍 Natural Language Query</div>', unsafe_allow_html=True)

    # Check if AI is available
    from .model_manager import ModelManager
    from .nl_query_translator import (
        execute_nl_query,
        add_query_to_history,
        get_query_history,
        clear_query_history
    )

    manager = ModelManager()

    if not manager.is_ollama_installed() or not manager.get_default_model():
        st.warning("⚠️ AI features not configured. Please set up AI first.")
        if st.button("🤖 Go to AI Setup"):
            st.session_state.current_section = "ai_setup"
            st.rerun()
        return

    # Check if data is loaded
    if not st.session_state.get('data_loaded', False):
        st.info("📁 Please upload a dataset first to use natural language queries.")
        return

    # Get default model
    default_model = manager.get_default_model()

    # Display info
    st.markdown("""
    **Natural Language Query Translator** converts your questions into executable code:
    - 💬 Ask questions in plain English
    - 🐍 Get pandas code automatically
    - ▶️ Execute code safely
    - 📊 View results instantly
    """)

    # Example queries
    with st.expander("💡 Example Queries"):
        st.markdown("""
        Try asking:
        - "Show me the first 10 rows"
        - "What are the columns with missing values?"
        - "Show me descriptive statistics for numeric columns"
        - "Find all rows where age is greater than 30"
        - "Group by category and calculate mean values"
        - "Show me the correlation between price and rating"
        - "Find outliers in the salary column"
        - "Count unique values in each column"
        """)

    st.markdown("---")

    # Query input
    query = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="e.g., Show me rows with missing values in the 'age' column",
        key="nl_query_input"
    )

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        execute_button = st.button("🚀 Execute", type="primary")
    with col2:
        if st.button("🗑️ Clear History"):
            clear_query_history()
            st.success("History cleared!")
            st.rerun()

    # Execute query
    if execute_button and query:
        with st.spinner("🤖 Generating and executing code..."):
            result = execute_nl_query(
                query=query,
                df=st.session_state.df,
                dataset_name=st.session_state.get('filename', 'dataset'),
                model_name=default_model
            )

            # Add to history
            add_query_to_history(result)

            # Display result
            st.markdown("---")
            st.markdown("### 📊 Query Results")

            if result['success']:
                st.success("✅ Code generated and executed successfully!")

                # Display the generated code
                st.markdown("**Generated Code:**")
                st.code(result['code'], language='python')

                # Display the result
                st.markdown("**Output:**")

                # Check if result is a DataFrame
                if isinstance(result['result'], pd.DataFrame):
                    st.dataframe(result['result'], use_container_width=True)
                elif isinstance(result['result'], pd.Series):
                    st.dataframe(result['result'].to_frame(), use_container_width=True)
                else:
                    st.text(str(result['result']))

                # Metadata
                st.caption(f"⏱️ Generated in {result['generation_time_ms']:.0f}ms")

            else:
                st.error(f"❌ Error: {result['error']}")

                if result['code']:
                    st.markdown("**Generated Code (failed to execute):**")
                    st.code(result['code'], language='python')

    # Display query history
    history = get_query_history()

    if history:
        st.markdown("---")
        st.markdown("### 📜 Query History")

        # Show last 5 queries
        for i, query_result in enumerate(reversed(history[-5:])):
            with st.expander(f"Query {len(history) - i}: {query_result['query'][:50]}..."):
                st.markdown(f"**Query:** {query_result['query']}")

                if query_result['success']:
                    st.markdown("**Status:** ✅ Success")
                    st.code(query_result['code'], language='python')

                    # Show result preview
                    if query_result['result'] is not None:
                        st.markdown("**Result:**")
                        if isinstance(query_result['result'], (pd.DataFrame, pd.Series)):
                            st.dataframe(query_result['result'], use_container_width=True)
                        else:
                            st.text(str(query_result['result'])[:500])
                else:
                    st.markdown("**Status:** ❌ Failed")
                    st.error(query_result['error'])

                st.caption(f"🕐 {query_result['timestamp']}")

        st.caption(f"Showing last 5 of {len(history)} queries")
    elif execute_button:
        pass  # Don't show empty history message if just executed
    else:
        st.info("💡 No queries yet. Try asking a question about your data!")
