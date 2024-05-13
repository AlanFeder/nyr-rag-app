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

def calc_cost(response1) -> int:
    prompt_tokens = response1.usage.prompt_tokens
    completion_tokens = response1.usage.completion_tokens

    cost_cents = (prompt_tokens + (3 * completion_tokens)) / 1000

    return cost_cents


def do_1_oai_query(system_prompt: str, user_prompt: str, openai_client: OpenAI) -> tuple[str, float]:

    messages1 = [
        {'role':'system', 'content':system_prompt},
        {'role':'user',   'content':user_prompt}
    ]
    response1 = openai_client.chat.completions.create(
        messages=messages1,
        model='gpt-4-turbo',
        seed=18,
        temperature=0
    )

    content_out = response1.choices[0].message.content
    cost_cents = calc_cost(response1)

    return content_out, cost_cents
