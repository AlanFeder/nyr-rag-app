import logging
import requests
import json

logger = logging.getLogger(__name__)

def run_remote_llama(
    llm_model_name: str, 
    messages_list: list, 
    ip_address: str = '3.88.116.167'#'18.214.91.237'
) -> dict:
    """Sends a chat request to a remote LLAMA server.

    Args:
        llm_model_name: The name of the LLM (Large Language Model) to use on the remote server.
        messages_list: A list of messages representing the chat history.
        ip_address: The IP address of the remote server (default: '18.214.91.237').

    Returns:
        dict: The parsed response dictionary from the remote LLAMA server.
    """

    url = f"http://{ip_address}:11434/api/chat"  # Construct the API endpoint URL

    input_data = {
        'model': llm_model_name,
        'messages': messages_list,
        'stream': False,  # Get a complete response, not a stream
        'options': {
            "seed": 18,  # Set a seed for deterministic output
            "temperature": 0  # Generate responses with low randomness
        }
    }

    response = requests.post(url, json=input_data)  # Send the POST request
    if response.status_code == 200:
        logging.info("EC2 Ran Correctly, with a response")
    else:
        logging.error(f'API Ran with error code {response.status_code}')
    response_text = response.text
    result1 = json.loads(response_text)  # Parse the JSON response

    return result1

def do_generation(query0: str, context0: str, model: str) -> str:
    """Generates a response to the query based on retrieved context.

    Args:
        query0: The user's query.
        context0: The retrieved context from the knowledge base.
        model: The name of the language generation model to use.

    Returns:
        str: The generated response.
    """
    system_prompt = """\
You are a chatbot on a cybersecurity software called Ryskview. \
You will answer questions about the knowledge base on the Ryskview \
program about how to use the program.  Information will be included \
based on a retrieval process, and you will receive the most likely \
elements that can answer.  Each document will have an <h1> tag for the \
top-level category and an <h2> for article title. \
If you do not know the answer, do not \
make up an answer - just say you don't know. If the context does not \
include the answer, then say you don't know.  Be polite.\
"""

    user_prompt = f'''QUERY: {query0}\n
======\n
CONTEXT: {context0}\n
======\n
Remember, if you do not know, do not make up an answer.
The query was {query0}'''
    messages = [{'role':'system', 'content':system_prompt},
                {'role':'user','content':user_prompt}]
    try:
        logger.info(f'start running {model} ')
        result1 = run_remote_llama(llm_model_name=model, messages_list=messages)
    except Exception as e:
        logger.error(f"Error {e} running model {model}, user_prompt: {user_prompt}")
    logger.info('Generation finished')
    logger.info(f"eval seconds: {round(result1['prompt_eval_duration']/1e9, 1)}")
    logger.info(f"gen seconds: {round(result1['eval_duration']/1e9, 1)}")
    logger.info(f"total seconds: {round(result1['total_duration']/1e9, 1)}")
    logger.info(f"# tokens out: {result1['eval_count']}")

    out_text = result1['message']['content']

    return out_text

# if __name__ == '__main__':
    # logger = logging.getLogger(__name__)
