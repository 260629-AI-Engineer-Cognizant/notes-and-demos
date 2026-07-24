import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse

# This imports will allow us to read from the env file and
# will allow us to create a Bedrock model with the Converse API

load_dotenv()

# Creating a data class for our basic settings from the env file
@dataclass(frozen=True)
# The above marks the class as Frozen or effectively readonly
class BedrockSettings:

    region: str
    model_a_id: str
    model_b_id: str
    judge_model_id: str

def get_settings() -> BedrockSettings:

    # This fetches the values from the loaded env file
    # and sets them in the BedrockSetting object
    return BedrockSettings(
        region=os.getenv("AWS_REGION", "us-east-1"),
        model_a_id=os.getenv("AWS_MODEL_A_ID", "amazon.nova-micro-v1:0"),
        model_b_id=os.getenv("AWS_MODEL_B_ID","amazon.nova-lite-v1:0"),
        judge_model_id=os.getenv("AWS_JUDGE_MODEL_ID","amazon.nova-lite-v1:0"),
    )

def create_chat_model(
        model_id: str,
        region: str,
        *,
        temperature: float = .7,
        max_tokens: int = 350
) -> ChatBedrockConverse:
    # Create a langchain chat model backed by Amazon Bedrock Converse API

    # The credential details like our bearer token will automatically
    # be read by boto3

    return ChatBedrockConverse(
        model_id = model_id,
        region_name=region,
        temperature=temperature,
        max_tokens=max_tokens
    )