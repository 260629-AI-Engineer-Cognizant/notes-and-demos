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

from uuid import uuid4
from chains import ROLE_PROMPTS, build_showdown_chain, build_judge_chain, build_memory_chain
from models import get_settings, create_chat_model
from langchain_core.chat_history import InMemoryChatMessageHistory

st.set_page_config(
    page_title="Bedrock Battle",
    page_icon="🥊",
    layout="wide",
)

settings = get_settings()

# Streamlit reruns this script any time a widget is affected
# It'll keep track of the widget values but it'll rerun all of the other
# variable so they usually get axed.
# How do we fix this? Streamlit STATE

if "showdown_results" not in st.session_state:
    st.session_state.showdown_results = None

# Recall that state allows us to keep track of variable during reruns
# So we need this to store our values
if "chat_display" not in st.session_state:
    st.session_state.chat_display=[]

if "memory_store" not in st.session_state:
    st.session_state.memory_store = {}

if "memory_session_id" not in st.session_state:
    st.session_state.memory_session_id = str(uuid4())


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in st.session_state.memory_store:
        st.session_state.memory_store[session_id] = InMemoryChatMessageHistory()

    return st.session_state.memory_store[session_id]

# =======================================================
# Page Layout Below
# =======================================================

# Now we're providing details to the page
st.title("🥊 Bedrock Battle Demo")

st.write(
    "Compare Bedrock-powered Langchain Workflows without building another RAG Application"
)

# We'll make some tabs to control what page we're looking at
# This is similar to routing in a Single Page Application

showdown_tab, memory_tab = st.tabs(
    [
        "Model Showdown",
        "Memory Chat"
    ]
)

# Create a Sidebar to allow us to affect the different models and config
# choices

with st.sidebar:
    st.header("Bedrock Settings")
    st.caption("Change this if your AWS Region or model Ids Differ")

    region = st.text_input("AWS Region", value=settings.region)
    model_a_id = st.text_input("Model A Id", value=settings.model_a_id)
    model_b_id = st.text_input("Model B Id", value=settings.model_b_id)
    judge_model_id = st.text_input("Judge Model Id", value=settings.judge_model_id)

    # Sliders for other settings
    temperature = st.slider(
        "Temperature",
        min_value= 0.0,
        max_value= 1.0,
        value=.7,
        step=.1,
        help="Higher values usually produce more varied responses"
    )

    max_tokens = st.slider(
        "Maximum Output Tokens",
        min_value= 100,
        max_value= 800,
        value=350,
        step=50
    )

with showdown_tab:
    st.subheader("Showdown")

    challenge = st.text_area(
        "Challenge",
        value=(
            "Invent a ridiculous but genuinely useful product for people who are folding laundry"
        ),
        height=110
    )

    # Role selection box
    role_a = st.selectbox(
        "Model A Role",
        list(ROLE_PROMPTS.keys())
    )

    role_b = st.selectbox(
        "Model B Role",
        list(ROLE_PROMPTS.keys()),
        index=1
    )

    # Let's add a check box to determine if they want a final verdict
    run_judge = st.checkbox(
        "Ask another chain to judge the responses",
        value=True
    )

    # Add on a button and give it functionality
    if st.button("Start the Showdown", type="primary"):
        # Create a model
        model_a = create_chat_model(
            model_a_id,
            region,
            temperature=temperature,
            max_tokens=max_tokens
        )

        model_b = create_chat_model(
            model_b_id,
            region,
            temperature=temperature,
            max_tokens=max_tokens
        )
        # Create the Chain
        # contestant_a = build_contestant_chain(
        #     model_a,
        #     role_a
        # )
        showdown_chain = build_showdown_chain(
            model_a,
            model_b,
            role_a, role_b
        )

        # Invoke the Chain with the challenge
        responses = showdown_chain.invoke(
            {"challenge": challenge}
        )

        # Below is the code for the verdict if desired
        verdict = None

        if run_judge:
            judge_model = create_chat_model(
                judge_model_id,
                region,
                temperature=.1,
                max_tokens=250
            )

            # Build the chain
            judge_chain = build_judge_chain(
                judge_model,
            )

            verdict = judge_chain.invoke(
                {
                    "challenge": challenge,
                    "response_a": responses["response_a"],
                    "response_b": responses["response_b"],
                }
            )

        # Store response in session state so it doesn't leave
        st.session_state.showdown_results = {
            "role_a": role_a,
            "role_b": role_b,
            "response_a": responses["response_a"],
            "response_b": responses["response_b"],
            "verdict": verdict
        }

        # I could add on another chain here and invoke them both manually
        # but it's time for a slightly different architecture.

        #Now at this point there are 2 responses so let's change how we
        # print this on the page


    # We can now search the session state and only show the columns if the results
    # are in
    results = st.session_state.showdown_results

    if results:
        # Create some columns to store the data
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(f"### Model A - {role_a}")
            st.write(results["response_a"])

        with col_b:
            st.markdown(f"### Model B - {role_b}")
            st.write(results["response_b"])

        # Display Verdict
        if results["verdict"]:
            st.markdown(f"### Judge's Verdict")
            st.write(results["verdict"])

with memory_tab:
    st.subheader("Memory Chat")
    st.write("Tell the assistant you name or a preference, then ask about it in a later message")

    # Render the messages
    for message in st.session_state.chat_display:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    user_input = st.chat_input("Talk to the memory enabled assistant")

    if user_input:
        st.session_state.chat_display.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        # Create a model and use the memory chain
        model = create_chat_model(
            model_a_id,
            region,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Build our chain
        memory_chain = build_memory_chain(
            model,
            get_session_history
        )

        # Invoke the chain to get a response
        response = memory_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {
                    "session_id": st.session_state.memory_session_id
                }
            }
        )

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.chat_display.append({
            "role": "assistant",
            "content": response
        })
