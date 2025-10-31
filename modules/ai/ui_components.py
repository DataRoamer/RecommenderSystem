"""
UI Components for AI Features
Streamlit components for model management and AI setup
"""

import streamlit as st
from typing import Optional, Tuple
from .model_manager import ModelManager, get_available_models
from .llm_integration import test_ollama_connection


def display_ai_setup_wizard():
    """
    Display AI setup wizard for first-time users
    Guides through Ollama installation and model download
    """
    st.markdown("# 🤖 AI Features Setup")
    st.markdown("Welcome to the AI-powered EDA Tool! Let's get your local AI assistant set up.")
    st.markdown("---")

    manager = ModelManager()

    # Step 1: Check Ollama installation
    st.subheader("📥 Step 1: Ollama Installation")

    ollama_installed = manager.is_ollama_installed()

    if ollama_installed:
        version = manager.get_ollama_version()
        st.success(f"✅ Ollama is installed! Version: {version}")

        # Test connection
        is_running, message = test_ollama_connection()
        if is_running:
            st.success(f"✅ {message}")
        else:
            st.warning(f"⚠️ {message}")
            st.info("💡 Please start Ollama service and refresh this page.")
            return

    else:
        st.error("❌ Ollama is not installed on your system.")
        st.markdown("""
        **Ollama** is required to run local AI models. It's free, open-source, and runs entirely on your computer.

        **Why Ollama?**
        - 🔒 Complete privacy - your data never leaves your machine
        - 💰 Zero ongoing costs - no API fees
        - ⚡ Fast local inference
        - 🌐 Works offline
        """)

        with st.expander("📖 Installation Instructions", expanded=True):
            instructions = manager.get_ollama_install_instructions()
            st.markdown(instructions)

        st.warning("⚠️ After installing Ollama, please restart this application.")
        return

    st.markdown("---")

    # Step 2: System Information
    st.subheader("💻 Step 2: System Information")

    system_info = manager.get_system_info()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Operating System", system_info['os'])
    with col2:
        ram_available = system_info.get('ram_available_gb', 'Unknown')
        if ram_available != 'Unknown':
            st.metric("Available RAM", f"{ram_available} GB")
        else:
            st.metric("Total RAM", system_info.get('ram_gb', 'Unknown'))
    with col3:
        st.metric("CPU Cores", system_info.get('cpu_count', 'Unknown'))

    # RAM-based recommendation
    ram_gb = system_info.get('ram_gb', 0)
    if ram_gb != 'Unknown' and isinstance(ram_gb, (int, float)):
        if ram_gb >= 16:
            ram_status = "🟢 Excellent - Can run all models"
        elif ram_gb >= 8:
            ram_status = "🟡 Good - Recommended for 8B models"
        elif ram_gb >= 4:
            ram_status = "🟠 Limited - Only lightweight models recommended"
        else:
            ram_status = "🔴 Low - May have performance issues"

        st.info(f"**RAM Status:** {ram_status}")

    st.markdown("---")

    # Step 3: Model Selection and Installation
    st.subheader("🎯 Step 3: Choose Your AI Model")

    installed_models = manager.list_installed_models()

    if installed_models:
        st.success(f"✅ You have {len(installed_models)} model(s) installed:")
        for model in installed_models:
            st.write(f"  • {model}")

        st.markdown("**Add more models below or skip to the next section.**")

    models = get_available_models()

    # Display model cards
    for model_key, model_info in models.items():
        is_installed = manager.is_model_installed(model_key)
        is_recommended = model_info.get('recommended', False)

        # Model card
        with st.expander(
            f"{'⭐ ' if is_recommended else ''}{model_info['name']} "
            f"({'✅ Installed' if is_installed else '⬇️ Not Installed'})",
            expanded=(is_recommended and not is_installed)
        ):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**{model_info['description']}**")
                st.write(f"📦 **Size:** {model_info['size']}")
                st.write(f"💾 **RAM Required:** {model_info['ram_required']}")

                # Quality and speed ratings
                quality_stars = "⭐" * model_info['quality']
                speed_stars = "⚡" * model_info['speed']

                st.write(f"🎯 **Quality:** {quality_stars}")
                st.write(f"⚡ **Speed:** {speed_stars}")

            with col2:
                if is_installed:
                    st.success("✅ Installed")

                    if st.button(f"🗑️ Remove", key=f"remove_{model_key}"):
                        with st.spinner(f"Removing {model_info['name']}..."):
                            success, message = manager.remove_model(model_key)
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)

                    if st.button(f"🎯 Set as Default", key=f"default_{model_key}"):
                        if manager.set_default_model(model_key):
                            st.success(f"✅ {model_info['name']} set as default")
                            st.rerun()

                else:
                    if st.button(f"⬇️ Download {model_info['size']}", key=f"download_{model_key}"):
                        download_model_with_progress(manager, model_key, model_info['name'])

    # Default model selection
    if installed_models:
        st.markdown("---")
        st.subheader("⚙️ Default Model")

        default_model = manager.get_default_model()
        if default_model:
            st.info(f"**Current default:** {default_model}")
        else:
            st.warning("No default model set. Please set one above.")

    st.markdown("---")

    # Completion check
    if installed_models:
        st.success("🎉 **Setup Complete!** You're ready to use AI features.")
        st.info("💡 Navigate to other sections using the sidebar to start using AI-powered analysis.")
    else:
        st.warning("⚠️ **Setup Incomplete:** Please download at least one model to use AI features.")


def download_model_with_progress(manager: ModelManager, model_key: str, model_name: str):
    """
    Download a model with progress tracking

    Args:
        manager: ModelManager instance
        model_key: Model identifier (e.g., 'llama3.1:8b')
        model_name: Display name
    """
    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    progress_bar = progress_placeholder.progress(0)
    status_placeholder.info(f"Starting download of {model_name}...")

    def progress_callback(line: str):
        """Callback for download progress"""
        status_placeholder.text(line)
        # Simple progress simulation (Ollama doesn't provide percentage)
        # In a real implementation, you'd parse the output for progress info

    try:
        success, message = manager.download_model(model_key, progress_callback)

        if success:
            progress_bar.progress(100)
            status_placeholder.empty()
            progress_placeholder.empty()
            st.success(f"✅ {message}")
            st.balloons()
            st.rerun()
        else:
            progress_placeholder.empty()
            status_placeholder.empty()
            st.error(f"❌ {message}")

    except Exception as e:
        progress_placeholder.empty()
        status_placeholder.empty()
        st.error(f"❌ Error downloading model: {str(e)}")


def display_model_settings():
    """
    Display model settings and configuration
    For use in sidebar or settings page
    """
    manager = ModelManager()

    st.markdown("### 🤖 AI Model Settings")

    # Check status
    if not manager.is_ollama_installed():
        st.error("❌ Ollama not installed")
        if st.button("📖 View Setup Instructions"):
            st.session_state.current_section = "ai_setup"
            st.rerun()
        return

    is_running, message = test_ollama_connection()
    if not is_running:
        st.warning(f"⚠️ {message}")
        st.info("Please start Ollama service")
        return

    # Installed models
    installed_models = manager.list_installed_models()

    if not installed_models:
        st.warning("No models installed")
        if st.button("⬇️ Download Models"):
            st.session_state.current_section = "ai_setup"
            st.rerun()
        return

    # Model selector
    default_model = manager.get_default_model()
    current_index = 0

    if default_model and default_model in installed_models:
        current_index = installed_models.index(default_model)

    selected_model = st.selectbox(
        "Active Model",
        installed_models,
        index=current_index,
        key="ai_model_selector"
    )

    if selected_model != default_model:
        if st.button("✅ Set as Default"):
            manager.set_default_model(selected_model)
            st.success(f"Set {selected_model} as default")
            st.rerun()

    # Temperature setting
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Lower = more focused, Higher = more creative"
    )

    # Save to session state
    st.session_state.ai_temperature = temperature
    st.session_state.ai_model = selected_model

    # Quick stats
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Models Installed", len(installed_models))
    with col2:
        if 'ai_queries_count' in st.session_state:
            st.metric("Queries Run", st.session_state.ai_queries_count)
        else:
            st.metric("Queries Run", 0)

    # Link to full setup
    if st.button("⚙️ Manage Models"):
        st.session_state.current_section = "ai_setup"
        st.rerun()


def display_ai_status_badge():
    """
    Display AI availability status badge
    For use in sidebar or header
    """
    manager = ModelManager()

    if not manager.is_ollama_installed():
        st.sidebar.error("🤖 AI: Not Available")
        return

    is_running, _ = test_ollama_connection()

    if is_running:
        installed_models = manager.list_installed_models()
        if installed_models:
            default_model = manager.get_default_model()
            if default_model:
                # Extract short name (e.g., "llama3.1:8b" -> "Llama 3.1")
                short_name = default_model.replace(":", " ").title()
                st.sidebar.success(f"🤖 AI: {short_name}")
            else:
                st.sidebar.success(f"🤖 AI: Ready ({len(installed_models)} models)")
        else:
            st.sidebar.warning("🤖 AI: No models")
    else:
        st.sidebar.warning("🤖 AI: Service offline")


def check_ai_prerequisites() -> Tuple[bool, str]:
    """
    Check if AI features are ready to use

    Returns:
        Tuple of (is_ready: bool, message: str)
    """
    manager = ModelManager()

    if not manager.is_ollama_installed():
        return False, "Ollama is not installed. Please visit AI Setup to get started."

    is_running, message = test_ollama_connection()
    if not is_running:
        return False, f"Ollama service is not running. {message}"

    installed_models = manager.list_installed_models()
    if not installed_models:
        return False, "No AI models installed. Please visit AI Setup to download a model."

    default_model = manager.get_default_model()
    if not default_model:
        return False, "No default model set. Please set one in AI Setup."

    return True, f"AI ready with {default_model}"


def display_ai_feature_guard(feature_name: str = "AI Features"):
    """
    Display a guard message if AI is not ready
    Use this at the top of AI-powered features

    Args:
        feature_name: Name of the feature requiring AI

    Returns:
        True if AI is ready, False otherwise
    """
    is_ready, message = check_ai_prerequisites()

    if not is_ready:
        st.warning(f"⚠️ {feature_name} requires AI setup")
        st.info(message)

        if st.button("🚀 Go to AI Setup"):
            st.session_state.current_section = "ai_setup"
            st.rerun()

        return False

    return True
