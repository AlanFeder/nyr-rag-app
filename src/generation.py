import logging
from .openai_code import do_1_oai_query
from openai import OpenAI

logger = logging.getLogger()

def make_user_prompt(question, keep_texts):
    user_prompt = f'''
Question: {question}
==============================
'''
    list_strs = []
    for i, text0 in enumerate(keep_texts.values()):
        list_strs.append(f'Video Transcript {i+1}\n{text0}')
    user_prompt += '\n---\n'.join(list_strs)
    user_prompt += '''
==============================
After analyzing the above video transcripts, please provide a helpful answer to my question. Remember to stay within two paragraphs
Address the response to me directly.  Do not use any information not explicitly supported by the transcripts.'''
    return user_prompt


def do_generation(query0: str, keep_texts: dict, openai_client: OpenAI) -> str:
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

    user_prompt = make_user_prompt(query0, keep_texts)
    try:
        content_out, cost_cents = do_1_oai_query(system_prompt, user_prompt, openai_client)
    except Exception as e:
        logger.error(f"Error {e} running user_prompt: {user_prompt}")
    logger.info('Generation finished')
    logger.info(f"cost: {cost_cents}")

    return content_out, cost_cents
