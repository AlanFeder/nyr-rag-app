import logging
from typing import Any
from dotenv import load_dotenv
import os
from openai import OpenAI
import pandas as pd
import pickle
import boto3
import io
from botocore import UNSIGNED
from botocore.client import Config

logger = logging.getLogger(__name__)

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

    logger.info("OpenAI Client set up")

    return openai_client

def initialize_boto3():
    return boto3.client('s3', config=Config(signature_version=UNSIGNED))


# Read pickle files
def read_pickle_from_s3(s3, bucket_name: str, key: str) -> Any:
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    return pickle.loads(obj['Body'].read())

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