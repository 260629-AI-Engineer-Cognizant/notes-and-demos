"""
We want to create a way to interact with our data driven application
that is nice to look at but I don't want to write any frontend code.
Solution: Streamlit

Streamlit is a python framework for designing simple frontends using
exclusively python code. It'll create a React-based frontend for your
application and is the choice of many developers that are working on
AI-based apps

Quick Notes:
Streamlit works with widgets
Streamlit will rerun the page script any time a widget is changed, so
we'll have to be aware of how it affects memory
"""

import streamlit as st

from models import get_settings

st.set_page_config(
    page_title="Bedrock Battle",
    page_icon="🥊",
    layout="wide",
)

settings = get_settings()

# Now we're providing details to the page
st.title("🥊 Bedrock Battle Demo")

st.write(
    "Compare Bedrock-powered Langchain Workflows"
    "without building another RAG Application"
)

