import logging
import streamlit as st
from pyprojroot import here
from dotenv import load_dotenv
import os
from openai import OpenAI
from groq import Groq
import pandas as pd
import pickle
from langsmith.wrappers import wrap_openai

logger = logging.getLogger()

@st.cache_resource(ttl=14400)
def load_oai_model(api_key: str = None) -> OpenAI:
    """
    Load OpenAI API client.

    Returns:
        OpenAI: The OpenAI API client.
    """
    # Load API key from environment variable
    if not api_key: 
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
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
        openai_client = load_oai_model(api_key=openai_api_key)
        ret_client = gen_client = openai_client
    else:
        ret_client = load_oai_model(api_key=openai_api_key)
        gen_client = load_groq_model()

    return ret_client, gen_client

@st.cache_data(ttl=14400)
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
    with open(fp_data / 'transcripts_40seconds.pkl', 'rb') as f1:
        transcripts_40seconds = pickle.load(f1) 
    with open(fp_data / 'full_embeds.pkl', 'rb') as f2:
        full_embeds = pickle.load(f2)

    logger.info('Loaded files')

    return df_talks, transcript_dicts, transcripts_40seconds, full_embeds
