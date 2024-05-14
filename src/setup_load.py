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
    Load OpenAI API client.

    Returns:
        OpenAI: The OpenAI API client.
    """
    # Load API key from environment variable
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Create OpenAI API client
    openai_client = OpenAI(api_key=openai_api_key)

    logger.info("OpenAI Client set up")

    return openai_client

@st.cache_resource(ttl=7200)
def load_groq() -> Groq:
    """
    Load Groq API client.

    Returns:
        Groq: The Groq API client.
    """
    # Load API key from environment variable
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    # Create Groq API client
    groq_client = Groq(api_key=api_key)

    logger.info("Groq Client set up")

    return groq_client

@st.cache_resource(ttl=7200)
def load_cohere() -> Cohere:
    """
    Load Cohere API client.

    Returns:
        Cohere: The Cohere API client.
    """
    # Load API key from environment variable
    load_dotenv()
    api_key = os.getenv("CO_API_KEY")

    # Create Cohere API client
    cohere_client = Cohere(api_key)

    logger.info("Cohere Client set up")

    return cohere_client

def load_api_clients(use_oai: bool = False) -> tuple[OpenAI | Cohere, OpenAI | Groq]:
    """
    Load API clients based on the use_oai flag.

    Args:
        use_oai (bool): Flag to determine whether to use OpenAI or alternate clients.

    Returns:
        tuple[OpenAI | Cohere, OpenAI | Groq]: A tuple containing the retrieval client and generation client.
    """
    if use_oai:
        openai_client = load_oai_model()
        ret_client = gen_client = openai_client
    else:
        # ret_client = load_cohere()
        ret_client = load_oai_model()
        gen_client = load_groq()
    
    return ret_client, gen_client

@st.cache_data(ttl=7200)
def import_data() -> tuple[pd.DataFrame, dict, dict]:
    """
    Import data from files.

    Returns:
        tuple[pd.DataFrame, dict, dict]: A tuple containing the talks dataframe, transcript dictionaries, and full embeddings.
    """
    fp_data = here() / 'data'
    df_talks = pd.read_parquet(fp_data / 'talks_on_youtube.parquet')
    with open(fp_data / 'transcripts.pkl', 'rb') as f1:
        transcript_dicts = pickle.load(f1)
    with open(fp_data / 'full_embeds_oai.pkl', 'rb') as f2:
        full_embeds_oai = pickle.load(f2)

    logging.info('Loaded files')

    return df_talks, transcript_dicts, full_embeds_oai
