import logging
import tiktoken
import numpy as np

logger = logging.getLogger()

def dict_to_list_and_array(data: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """
    Convert a dictionary to a list of keys and a numpy array of values.

    Args:
        data (dict[str, np.ndarray]): The input dictionary with string keys and numpy array values.

    Returns:
        tuple[list[str], np.ndarray]: A tuple containing a list of keys and a numpy array of values.
    """
    # Extract the keys from the dictionary and convert them to a list
    keys = list(data.keys())
    
    # Extract the values from the dictionary and convert them to a numpy array
    values = np.array(list(data.values()))
    
    return keys, values

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
    return len(token_ids)

def calc_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate the cost in cents based on the number of prompt and completion tokens.

    Args:
        prompt_tokens (int): The number of tokens in the prompt.
        completion_tokens (int): The number of tokens in the completion.

    Returns:
        float: The cost in cents.
    """
    cost_cents = (prompt_tokens + (3 * completion_tokens)) / 2000

    return cost_cents