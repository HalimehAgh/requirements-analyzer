"""
Streamlit UI for the Requirements Quality Analyzer.
"""

import os
import streamlit as st
from graph import pipeline
from utils.guardrails import validate_input

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Requirements Quality Analyzer",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Requirements Quality Analyzer")
st.caption("A multi-agent system built with LangGraph that analyzes software requirements for quality issues.")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This tool runs your requirements through a pipeline of 4 specialized AI agents:
    
    - 🔴 **Ambiguity Agent** — finds vague language
    - 🟡 **Consistency Agent** — detects contradictions
    - 🔵 **Completeness Agent** — flags missing information
    - 🟢 **Testability Agent** — checks if requirements are verifiable
    
    A **Reporter Agent** then synthesizes all findings into a final report.
    """)

    st.divider()
    st.header("⚙️ Settings")
    llm_provider = st.selectbox(
        "LLM Provider",
        options=["anthropic", "openai"],
        index=0
    )
    os.environ["LLM_PROVIDER"] = llm_provider

    st.divider()
    st.markdown("Built with [LangGraph](https://langchain-ai.github.io/langgraph/)")

# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------

st.subheader("📄 Input Requirements")

# Initialize session state
if "requirements_text" not in st.session_state:
    st.session_state.requirements_text = ""

col1, col2 = st.columns([1, 5])
with col1:
    if st.button("Load Sample"):
        try:
            with open("sample_inputs/sample_requirements.md", "r") as f:
                st.session_state.requirements_text = f.read()
        except FileNotFoundError:
            st.warning("Sample file not found.")

requirements_text = st.text_area(
    label="Paste your requirements document here",
    value=st.session_state.requirements_text,
    height=250,
    placeholder="REQ-1: The system shall allow users to log in...\nREQ-2: The application should respond quickly..."
)

analyze_btn = st.button("🚀 Analyze Requirements", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

if analyze_btn:
    if not requirements_text.strip():
        st.error("Please enter some requirements text first.")
        st.stop()

    # Input guardrail
    with st.spinner("Validating input..."):
        is_valid, reason = validate_input(requirements_text)

    if not is_valid:
        st.error(f"⛔ Invalid input: {reason}")
        st.stop()

    # Run the pipeline with progress tracking
    st.divider()
    st.subheader("⚙️ Running Analysis Pipeline")

    progress = st.progress(0, text="Starting pipeline...")
    status = st.empty()

    steps = {
        "ambiguity":    (20,  "🔴 Running Ambiguity Agent..."),
        "consistency":  (40,  "🟡 Running Consistency Agent..."),
        "completeness": (60,  "🔵 Running Completeness Agent..."),
        "testability":  (80,  "🟢 Running Testability Agent..."),
        "reporter":     (90,  "📝 Generating Report..."),
        "validator":    (100, "✅ Validating Report..."),
    }

    try:
        final_state = {}

        for event in pipeline.stream({"requirements_text": requirements_text}):
            node_name = list(event.keys())[0]

            if node_name in steps:
                pct, msg = steps[node_name]
                progress.progress(pct, text=msg)
                status.markdown(f"**{msg}**")

            node_output = event.get(node_name, {})
            if isinstance(node_output, dict):
                final_state.update(node_output)

        progress.progress(100, text="Done!")
        status.empty()

    except ValueError as e:
        st.error(f"Pipeline error: {e}")
        st.stop()

    # ---------------------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------------------

    st.divider()
    st.subheader("📊 Analysis Report")

    report = final_state.get("final_report")

    if report:
        st.markdown(report)

        st.download_button(
            label="⬇️ Download Report",
            data=report,
            file_name="requirements_quality_report.md",
            mime="text/markdown"
        )
    else:
        st.error("Report could not be generated. Please try again.")