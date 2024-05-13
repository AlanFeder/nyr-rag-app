import logging
from .openai_code import do_1_oai_query_stream, set_messages, OpenAI

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

def do_stream_generation(query1: str, keep_texts: dict, openai_client: OpenAI):
    user_prompt = make_user_prompt(query1, keep_texts=keep_texts)
    messages1, prompt_tokens = set_messages(SYSTEM_PROMPT, user_prompt)
    response = do_1_oai_query_stream(messages1, openai_client)

    return response, prompt_tokens
