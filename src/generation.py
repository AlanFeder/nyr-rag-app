import logging
from .openai_code import do_1_oai_query
from openai import OpenAI

logger = logging.getLogger()

SYSTEM_PROMPT = '''
You are an AI assistant that helps answer questions by searching through video transcripts. 
I have retrieved the two transcripts most likely to answer the user's question.
Carefully read through the transcripts to find information that helps answer the question. 
Be brief - your response should not be more than two paragraphs.
Only use information directly stated in the provided transcripts to answer the question. 
Do not add any information or make any claims that are not explicitly supported by the transcripts.
If the transcripts do not contain enough information to answer the question, state that you do not have enough information to provide a complete answer.
Format the response clearly.  If only one of the transcripts answers the question, don't reference the other and don't explain why its content is irrelevant.
'''

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


# def do_generation(query0: str, keep_texts: dict, openai_client: OpenAI) -> str:
#     """Generates a response to the query based on retrieved context.

#     Args:
#         query0: The user's query.
#         context0: The retrieved context from the knowledge base.
#         model: The name of the language generation model to use.

#     Returns:
#         str: The generated response.
#     """

#     user_prompt = make_user_prompt(query0, keep_texts)
#     try:
#         content_out, cost_cents = do_1_oai_query(SYSTEM_PROMPT, user_prompt, openai_client)
#     except Exception as e:
#         logger.error(f"Error {e} running user_prompt: {user_prompt}")
#     logger.info('Generation finished')
#     logger.info(f"cost: {cost_cents}")

#     return content_out, cost_cents
