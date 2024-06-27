import logging
import tiktoken

logger = logging.getLogger()

def calc_n_tokens(text_in: str) -> int:
    """
    Calculate the number of tokens in the input text using the 'o200k_base' encoding.

    Args:
        text_in (str): The input text.

    Returns:
        int: The number of tokens in the input text.
    """
    tok_model = tiktoken.get_encoding('o200k_base')
    token_ids = tok_model.encode(text=text_in)
    n_tokens = len(token_ids)

    logger.info(f'{n_tokens} counted')

    return n_tokens

def calc_cost(prompt_tokens: int, completion_tokens: int, embedding_tokens: int) -> float:
    """
    Calculate the cost in cents based on the number of prompt, completion, and embedding tokens.

    Args:
        prompt_tokens (int): The number of tokens in the prompt.
        completion_tokens (int): The number of tokens in the completion.
        embedding_tokens (int): The number of tokens in the embedding.

    Returns:
        float: The cost in cents.
    """
    prompt_cost = prompt_tokens / 2000
    completion_cost = 3 * completion_tokens / 2000
    embedding_cost = embedding_tokens / 500000

    cost_cents = prompt_cost + completion_cost + embedding_cost

    logger.info(f'Costs: Embedding: {embedding_cost}. Prompt: {prompt_cost}. Completion: {completion_cost}. Total Cost: {cost_cents}')

    return cost_cents
