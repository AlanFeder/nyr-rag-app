import logging
import streamlit as st
from pyprojroot import here
from dotenv import load_dotenv
import os
from openai import OpenAI
import pandas as pd
import pickle
from groq import Groq
from cohere import Client as Cohere

logger = logging.getLogger()

@st.cache_resource(ttl=7200)
def load_oai_model() -> OpenAI:
    """
    Create API clients.

    Returns:
        OpenAI: The API client.
    """
    # Load API keys from environment variable
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Create API client
    openai_client = OpenAI(api_key=openai_api_key)

    logger.info("Openai Client set up")

    return openai_client

@st.cache_resource(ttl=7200)
def load_groq() -> Groq:
    """
    Create API clients.

    Returns:
        Groq: The Groq client.
    """
    # Load API keys from environment variable
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    # Create API client
    openai_client = Groq(api_key=api_key)

    logger.info("Groq set up")

    return openai_client


@st.cache_resource(ttl=7200)
def load_cohere() -> Cohere:
    """
    Create API clients.

    Returns:
        Groq: The Cohere client.
    """
    # Load API keys from environment variable
    load_dotenv()
    api_key = os.getenv("CO_API_KEY")

    # Create API client
    cohere_client = Cohere(api_key)

    logger.info("Cohere set up")

    return cohere_client


@st.cache_resource(ttl=7200)
def load_cohere() -> Cohere:
    """
    Load embedding models and create API client.

    Returns:
        Cohere: The Cohere client.
    """
    # Load API keys from environment variable
    load_dotenv()
    api_key = os.getenv("CO_API_KEY")

    # Create API client
    groq_client = Cohere(api_key=api_key)

    logger.info("Cohere set up")

    return groq_client


def load_api_clients(use_oai: bool = False) -> tuple[OpenAI | Cohere, OpenAI | Groq]:
    if use_oai:
        openai_client = load_oai_model()
        ret_client = gen_client = openai_client
    else:
        # ret_client = load_cohere()
        ret_client = load_oai_model()
        gen_client = load_groq()
    
    return ret_client, gen_client


@st.cache_data(ttl=7200)
def import_data():
    fp_data = here() / 'data'
    df_talks = pd.read_parquet(fp_data / 'talks_on_youtube.parquet')
    with open(fp_data / 'transcripts.pkl', 'rb') as f1:
        transcript_dicts = pickle.load(f1)
    with open(fp_data / 'full_embeds_oai.pkl', 'rb') as f2:
        full_embeds_oai = pickle.load(f2)

    logging.info('Loaded files')

    return df_talks, transcript_dicts, full_embeds_oai

