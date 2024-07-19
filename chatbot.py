import os
import json
from dotenv import load_dotenv
from itertools import chain
import boto3
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import Bedrock
from langchain_community.chat_models import BedrockChat
from langchain import hub

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
        "maxTokenCount": 4096,
        "stopSequences": [],
        "temperature": 0,
        "topP": 1,
    }

    # Instantiate Bedrock LLM
    llm = Bedrock(
        client=bedrock, model_id="amazon.titan-text-express-v1", model_kwargs=model_kwargs
    )

    # Retrieve prompt from Langchain Hub
    prompt = hub.pull("texte/awsprompt")

    # Create document combination and retrieval chains
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(contexts, combine_docs_chain)

    # Invoke the chain with the query
    response = chain.invoke({"input": query})

    # Extract and return the answer
    answer = response['output']
    return answer

def answer_query(user_input):
    """
    Retrieve contexts for the user query from the knowledge base and call the LLM for a response.
    """
    # Retrieve contexts for the user input from the knowledge base
    user_contexts = get_contexts(user_input, knowledge_base_id)

    # Call the appropriate LLM based on the environment variable
    if llm_model == "amazon-titan":
        answer = call_titan(user_input, user_contexts)
    else:
        return "Unsupported LLM model"

    # Return the final response to the user
    if "result" in answer:
        return answer["result"]
    else:
        return "No response from model"
