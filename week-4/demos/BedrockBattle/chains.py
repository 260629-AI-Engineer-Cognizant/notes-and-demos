"""
This function is going to include all of the various pieces
to create our LangChain chains. It'll mostly be functions that
will be able to be used to return chains, which then can be invoked
"""

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel

# Constants get all caps
ROLE_PROMPTS = {
    "Creative Inventor": (
        "Be imaginative, surprising, playful and specific."
        "The answer should still be meaningful"
    ),
    "Practical Engineer":(
        "Prioritize feasibility, tradeoffs, cost and clear implementation steps"
    )
}


# Create a function that returns a CHAIN
# This will not call the chain or the model, calling this function
# Builds the chain which is later invoked

def build_contestant_chain(model: Any, role_name: str):
    role_instructions = ROLE_PROMPTS[role_name]

    # Generate a prompt for our LLM
    prompt=ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are competing in a friendly AI model showcase"
                f"Your role name is {role_name}"
                f"{role_instructions}"
                "Respond in no more than 180 words"
            ),
            (
                "human",
                "Challenge: \n{challenge}"
            )
        ]
    )

    return prompt | model | StrOutputParser()

# This function will allow us to create a chain that runs 2 different
# llms in parallel

def build_showdown_chain(
        model_a: Any,
        model_b: Any,
        role_a: str,
        role_b: str
):
    # Create a parallel chain that uses the existing function above
    return RunnableParallel(
        response_a = build_contestant_chain(model_a, role_a),
        response_b = build_contestant_chain(model_b, role_b)
    )

def build_judge_chain(model: Any):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a fair judge for a friendly AI Showdown."
                "Evaluate how well each response satisfies the original"
                "challenge. Choose one winner and explain in 2-4 sentences."
                "Do not judge based on model names"
            ),
            (
                "human",
                """
                Original Challenge:
                {challenge}
                
                Response A:
                {response_a}
                
                Response B:
                {response_b}
                
                Return:
                Winner A or B
                Reason: 2-4 Sentences
                """
            )
        ]
    )

    return prompt | model | StrOutputParser()