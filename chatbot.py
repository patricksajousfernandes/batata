import os
import json
from dotenv import load_dotenv
import boto3
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
from langchain_community.llms import Bedrock
from langchain_community.chat_models import BedrockChat
from langchain import hub
from langchain_aws import ChatBedrock

# Load environment variables from .env file
load_dotenv()

# Instantiate Bedrock clients with the profile name from environment variables
profile_name = os.getenv("profile_name")
boto3.setup_default_session(profile_name=profile_name)
bedrock = boto3.client("bedrock-runtime", "us-east-1")
bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", "us-east-1")

# Retrieve knowledge base ID and LLM model type from environment variables
knowledge_base_id = os.getenv("knowledge_base_id")
llm_model = os.getenv("llm_model")  # "amazon-titan" or "anthropic-claude"

def get_contexts(query, kb_id, number_of_results=5):
    """
    Retrieve contexts for a given query from the knowledge base using the Langchain retriever.
    """
    retriever = AmazonKnowledgeBasesRetriever(
        credentials_profile_name=profile_name,
        knowledge_base_id=kb_id,
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": number_of_results}},
    )
    contexts = retriever.invoke(input=query)
    return contexts

def call_titan(query, contexts):
    """
    Call the Amazon Titan LLM with the given query and contexts.
    """
    # Model configuration
    model_kwargs = {
        "max_tokens": 3000,
        "temperature": 0.1,
        "top_p": 0.9
    }

    # Instantiate Bedrock LLM
    llm = ChatBedrock(
        client=bedrock, model_id="anthropic.claude-3-5-sonnet-20240620-v1:0", model_kwargs=model_kwargs
    )

    # Retrieve prompt from Langchain Hub
    prompt = hub.pull("texte/awsprompt")

    # Create document combination and retrieval chains
    runnable = prompt | llm

    # Invoke the chain with the query
    response = runnable.invoke({"input": query, "context": contexts})

    # Assuming response is an object with a 'content' attribute or similar
    if hasattr(response, 'content'):
        return response.content
    else:
        return "Response format not recognized"

def answer_query(user_input):
    """
    Retrieve contexts for the user query from the knowledge base and call the LLM for a response.
    """
    # Retrieve contexts for the user input from the knowledge base
    user_contexts = get_contexts(user_input, knowledge_base_id)

    # Call the appropriate LLM based on the environment variable
    if llm_model == "amazon-titan":
        answer = call_titan(user_input, user_contexts)
        return answer
    else:
        return "Unsupported LLM model"


