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

def split_into_consecutive(arr: np.ndarray) -> list[np.ndarray]:
    """ 
    Split a numpy array into a list of arrays, where each sub-element contains consecutive parts.

    Args:
        arr (np.ndarray): The input numpy array.

    Returns:
        List[np.ndarray]: A list of numpy arrays, where each sub-element contains consecutive parts.
    
    ---

    I got the following function from the following GPT prompt
    > I have a numpy array of numbers.  some are consecutive, some are not (e.g. 3, 4, 5, 6, 10, 11, 12, 13, 19, 20, 21).  How can I split it into a list of arrays, where each sub-element is just the consecutive parts (e.g. [[3,4,5,6],[10,11,12,13],[19,20,21]])?

    The goal is so that if there are consecutive minutes, and I look at the before-and-after, they are combined as needed
    """
    # List to store the result
    result = []
    # Temporary list to store current sequence
    temp = [arr[0]]

    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1] + 1:
            # Continue the current sequence
            temp.append(arr[i])
        else:
            # Start a new sequence
            result.append(np.array(temp))
            temp = [arr[i]]

    # Append the last sequence
    result.append(np.array(temp))

    logger.info(f'{len(arr)} elements split into {len(result)} lists')

    return result
