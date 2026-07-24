from models import create_chat_model, get_settings

def main():
    settings = get_settings()

    print("Calling Bedrock through converse API")

    # Create our first chat model
    model = create_chat_model(
        settings.model_a_id,
        settings.region,
        temperature=.2,
        max_tokens=100,
    )

    # Call our model using our standard langchain pieces
    # We usually use invoke to call a model
    response = model.invoke(
        "In two sentences, explain why LangChain is useful outside of RAG"
    )

    print(response.text)

if __name__ == '__main__':
    main()