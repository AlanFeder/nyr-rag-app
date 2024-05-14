import logging
import streamlit as st
from pyprojroot import here
from dotenv import load_dotenv
import os
from openai import OpenAI
import pandas as pd
import pickle

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


def load_api_clients() -> tuple[OpenAI, OpenAI]:
    """
    Load API clients.


    Returns:
        tuple[OpenAI, OpenAI ]: A tuple containing the retrieval client and generation client.
    """
    openai_client = load_oai_model()
    ret_client = gen_client = openai_client
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
    with open(fp_data / 'transcripts_40seconds.pkl', 'rb') as f1:
        transcripts_40seconds = pickle.load(f1) 
    with open(fp_data / 'full_embeds.pkl', 'rb') as f2:
        full_embeds = pickle.load(f2)

    logging.info('Loaded files')

    return df_talks, transcript_dicts, transcripts_40seconds, full_embeds
