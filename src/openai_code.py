import logging
from openai import OpenAI
import numpy as np

logger = logging.getLogger()

def do_1_oai_embed(lt: str, openai_client: OpenAI) -> tuple[list[list[float]], int]:
    """
    Generate embeddings using the OpenAI API for a single list of texts.

    Args:
        lt (str): A text to generate embeddings for.
        openai_client (OpenAI): The OpenAI PI client.

    Returns:
        tuple[list[list[float]], int]: A tuple containing the generated embeddings and the total number of tokens.
    """
    embed_response = openai_client.embeddings.create(
        input=lt,
        model='text-embedding-3-small',
    )
    here_embed = np.array(embed_response.data[0].embedding)
    n_toks = embed_response.usage.total_tokens
    logger.info(f'Embedded {lt}')
    return here_embed, n_toks
