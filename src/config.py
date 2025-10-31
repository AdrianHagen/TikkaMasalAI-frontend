import os

import streamlit as st

DEV_BASE_URL = "http://127.0.0.1:8000/"
PROD_BASE_URL = "http://35.233.22.145:8000/"

URL = DEV_BASE_URL

DEFAULT_API_URL = os.path.join(URL, "predict")
DEFAULT_LLM_URL = os.path.join(URL, "llm/generate")
DEFAULT_EXPLAIN_URL = os.path.join(URL, "predict/explain")


def get_api_url() -> str:
    """Allow overriding the API URL via Streamlit secrets or query params."""
    # Prioritize secrets so deployments can configure without code changes.
    try:
        api_url = st.secrets.get("api_url", DEFAULT_API_URL)
    except Exception:
        api_url = DEFAULT_API_URL
    # Allow quick overrides via query param for local testing.
    api_override = st.query_params.get("api_url")
    if isinstance(api_override, list) and api_override:
        api_url = api_override[-1]
    elif isinstance(api_override, str) and api_override:
        api_url = api_override
    return api_url


def get_llm_url() -> str:
    """Allow overriding the LLM URL via Streamlit secrets or query params."""
    try:
        llm_url = st.secrets.get("llm_url", DEFAULT_LLM_URL)
    except Exception:
        llm_url = DEFAULT_LLM_URL
    llm_override = st.query_params.get("llm_url")
    if isinstance(llm_override, list) and llm_override:
        llm_url = llm_override[-1]
    elif isinstance(llm_override, str) and llm_override:
        llm_url = llm_override
    return llm_url


def get_explain_url() -> str:
    """Allow overriding the explain URL via Streamlit secrets or query params."""
    try:
        explain_url = st.secrets.get("explain_url", DEFAULT_EXPLAIN_URL)
    except Exception:
        explain_url = DEFAULT_EXPLAIN_URL
    explain_override = st.query_params.get("explain_url")
    if isinstance(explain_override, list) and explain_override:
        explain_url = explain_override[-1]
    elif isinstance(explain_override, str) and explain_override:
        explain_url = explain_override
    return explain_url
