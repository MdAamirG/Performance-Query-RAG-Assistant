"""Streamlit demo UI for the Solartis performance analysis system."""
import sys

import streamlit as st

sys.path.insert(0, "src")

from main import PerformanceAnalysisSystem


st.set_page_config(
    page_title="Solartis Performance Analyzer",
    page_icon="⚡",
    layout="wide",
)


@st.cache_resource
def get_system():
    return PerformanceAnalysisSystem()


SAMPLE_QUERIES = [
    "Why is my JSON filtering query slow?",
    "Why is SELECT * from a large table expensive?",
    "What causes a latency spike from 1 sec to 50 sec?",
    "How do I optimize a high-frequency config lookup query?",
]


st.title("Solartis Performance Analyzer")
st.caption("RAG + Gemini demo for query performance analysis")

with st.sidebar:
    st.subheader("Sample Queries")
    selected_sample = st.radio("Pick a query", SAMPLE_QUERIES, index=0)
    st.markdown("Use these to demo the flow quickly.")

st.markdown(
    """
This tool retrieves similar performance cases from a local knowledge base,
then uses Gemini to explain the likely root cause and suggest optimizations.
"""
)

query = st.text_area("Enter a performance question", value=selected_sample, height=120)

run_button = st.button("Analyze Query", type="primary")

if run_button and query.strip():
    with st.spinner("Retrieving cases and analyzing with Gemini..."):
        system = get_system()
        retrieved_cases = system.rag.retrieve_similar_cases(query, k=3)
        result = system.analyze_query(query)
        analysis = result.get("analysis", {})

    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.subheader("Analysis")
        st.markdown(f"**Problem:** {analysis.get('problem', 'N/A')}")
        st.markdown(f"**Priority:** {analysis.get('priority', 'N/A').upper()}")
        st.markdown(f"**Confidence:** {analysis.get('confidence', 'N/A').upper()}")
        expected_improvement = analysis.get("expected_improvement")
        if expected_improvement:
            st.markdown(f"**Expected Improvement:** {expected_improvement}")
        else:
            st.markdown(f"**Expected Improvement:** {analysis.get('expected_improvement_percent', 0)}%")

        st.markdown("### Root Causes")
        for cause in analysis.get("root_causes", []):
            st.write(f"- {cause}")

        st.markdown("### Suggestions")
        for suggestion in analysis.get("suggestions", []):
            st.write(f"- {suggestion}")

    with col2:
        st.subheader("Retrieved Similar Cases")
        if retrieved_cases:
            for case in retrieved_cases:
                with st.container(border=True):
                    st.markdown(f"**{case['name']}**")
                    st.caption(f"Similarity: {case.get('similarity_score', 0):.2%}")
                    st.write(case["query"])
        else:
            st.info("No similar cases found.")

    with st.expander("Show raw response"):
        st.json(result)

elif run_button:
    st.warning("Please enter a query first.")
