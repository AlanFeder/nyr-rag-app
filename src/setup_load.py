import logging
import streamlit as st
from typing import Any
from dotenv import load_dotenv
import os
from openai import OpenAI
from groq import Groq
import pandas as pd
import pickle
from langsmith.wrappers import wrap_openai
import boto3
import io

logger = logging.getLogger()

@st.cache_resource(ttl=14400)
def load_oai_model(api_key: str = None, use_oai: bool = True) -> OpenAI:
    """
    Load OpenAI API client.

    Returns:
        OpenAI: The OpenAI API client.
    """
    # Load API key from environment variable
    if (not api_key) & (not use_oai): 
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
        openai_client = load_oai_model(api_key=openai_api_key, use_oai=True)
        ret_client = gen_client = openai_client
    else:
        ret_client = load_oai_model(api_key=openai_api_key, use_oai=False)
        gen_client = load_groq_model()

    return ret_client, gen_client

@st.cache_resource(ttl=14400)
def initialize_boto3():
    return boto3.client('s3')


# Read pickle files
def read_pickle_from_s3(s3, bucket_name: str, key: str) -> Any:
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    return pickle.loads(obj['Body'].read())

@st.cache_data(ttl=14400)
def import_data() -> tuple[pd.DataFrame, dict, dict]:
    """
    Import data from files.

    Returns:
        tuple[pd.DataFrame, dict, dict]: A tuple containing the talks dataframe, transcript dictionaries, and full embeddings.
    """
 
    s3 = initialize_boto3()
    bucket_name = "nyr-rag-app"

    try:
        # Read parquet file
        obj = s3.get_object(Bucket=bucket_name, Key='talks_on_youtube.parquet')
        df_talks = pd.read_parquet(io.BytesIO(obj['Body'].read()))

        transcript_dicts = read_pickle_from_s3(s3, bucket_name, 'transcripts.pkl')
        transcripts_40seconds = read_pickle_from_s3(s3, bucket_name, 'transcripts_40seconds.pkl')
        full_embeds = read_pickle_from_s3(s3, bucket_name, 'full_embeds.pkl')

        logger.info('Loaded files')

        return df_talks, transcript_dicts, transcripts_40seconds, full_embeds
    except Exception as e:
        logger.error(f"Error loading data from S3: {str(e)}")
        raise