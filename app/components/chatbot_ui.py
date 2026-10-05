"""
Chatbot UI component for Streamlit dashboard.
"""

import streamlit as st
from chatbot.chatbot import OdishaFloodChatbot


def render_chatbot_component(bot: OdishaFloodChatbot):
    """Render interactive AI Flood Chatbot assistant in Streamlit."""
    st.markdown("### 💬 Intelligent Odisha Flood & Climate Assistant")
    st.markdown(
        "Ask natural language questions about next-day flood forecasts, 24-year historical rainfall observations, "
        "model explanations (SHAP), and meteorological science."
    )
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 **Namaskar!** I am your **Odisha Flood Early Warning Assistant**.\n\n"
                    "I am powered by the validated 24-year historical dataset (2001–2024), "
                    "calibrated machine learning models, and explainability algorithms.\n\n"
                    "Try asking me:\n"
                    "- 🔮 *'Will Cuttack have a flood tomorrow?'*\n"
                    "- 📊 *'What was the rainfall in Cuttack in July 2023?'*\n"
                    "- ⚡ *'Show the highest rainfall events recorded in Odisha'*\n"
                    "- 🧠 *'What is rainfall downscaling and how does it help?'*"
                )
            }
        ]
        
    # Quick question prompt buttons
    st.markdown("##### ⚡ Quick Queries:")
    q_cols = st.columns(4)
    quick_queries = [
        "Will Cuttack have a flood tomorrow?",
        "What was the rainfall in Cuttack in July 2023?",
        "Show 24-year flood summary for Puri",
        "What is rainfall downscaling?"
    ]
    
    for i, q_text in enumerate(quick_queries):
        if q_cols[i].button(q_text, key=f"quick_q_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q_text})
            with st.spinner("Analyzing dataset and executing models..."):
                resp = bot.respond(q_text)
                st.session_state.messages.append({"role": "assistant", "content": resp})
            st.rerun()
            
    st.markdown("---")
    
    # Render chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Chat user input
    if prompt := st.chat_input("Ask a question about flood prediction, historical rain, or explanations..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Querying dataset and models..."):
                response_text = bot.respond(prompt)
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
    if st.button("🧹 Clear Conversation History", type="secondary"):
        st.session_state.messages = []
        st.rerun()
