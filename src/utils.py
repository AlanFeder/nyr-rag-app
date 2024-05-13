import logging
import tiktoken
import numpy as np

logger = logging.getLogger()

def dict_to_list_and_array(data: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """
    Convert a dictionary to a list of keys and a numpy array of values.

    Args:
        data (dict[str, Any]): The input dictionary.

    Returns:
        tuple[list[str], np.ndarray]: A tuple containing a list of keys and a numpy array of values.
    """
    # Extract the keys from the dictionary and convert them to a list
    keys = list(data.keys())
    
    # Extract the values from the dictionary, convert them to a list, and then to a numpy array
    values = np.array(list(data.values()))
    
    return keys, values

def calc_n_tokens(text_in: str) -> int:
    tok_model = tiktoken.get_encoding('cl100k_base')
    token_ids = tok_model.encode(text=text_in)
    return len(token_ids)

def calc_cost(prompt_tokens: int, completion_tokens: int) -> int:

    cost_cents = (prompt_tokens + (3 * completion_tokens)) / 1000

    return cost_cents