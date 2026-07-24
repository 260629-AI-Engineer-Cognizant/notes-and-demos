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

from chains import ROLE_PROMPTS, build_showdown_chain, build_judge_chain
from models import get_settings, create_chat_model

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

# Now we're providing details to the page
st.title("🥊 Bedrock Battle Demo")

st.write(
    "Compare Bedrock-powered Langchain Workflows without building another RAG Application"
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

st.subheader("Single Contestant")

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