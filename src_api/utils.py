import logging
import numpy as np

logger = logging.getLogger(__name__)


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
