import logging
import streamlit as st
from dotenv import load_dotenv
import os
from openai import OpenAI
from groq import Groq
from langsmith.wrappers import wrap_openai

logger = logging.getLogger()

@st.cache_resource(ttl=14400)
def load_oai_model(api_key: str) -> OpenAI:
    """
    Load OpenAI API client.

    Returns:
        OpenAI: The OpenAI API client.
    """
    # Load API key from environment variable

    # Create OpenAI API client
    openai_client = OpenAI(api_key=api_key)
    openai_client = wrap_openai(openai_client)

    logger.info("OpenAI Client set up")

    return openai_client

@st.cache_resource(ttl=7200)
def load_groq_model() -> Groq:
    """
    Load OpenAI API client.

    Returns:
        OpenAI: The OpenAI API client.
    """
    # Load API key from environment variable
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    # Create Groq API client
    groq_client = Groq(api_key=api_key)

    logger.info("Groq Client set up")

    return groq_client


def load_api_clients(use_oai: bool = True, openai_api_key: str = None) -> tuple[OpenAI, OpenAI | Groq]:
    """
    Load API clients.


    Returns:
        tuple[OpenAI, OpenAI ]: A tuple containing the retrieval client and generation client.
    """
    if use_oai:
        gen_client = load_oai_model(api_key=openai_api_key)
    else:
        gen_client = load_groq_model()

    return gen_client