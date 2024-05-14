import logging
from .setup_load import OpenAI, Groq
from .utils import calc_n_tokens


logger = logging.getLogger()


def set_messages(system_prompt: str, user_prompt: str) -> dict:
    messages1 = [
        {'role':'system', 'content':system_prompt},
        {'role':'user',   'content':user_prompt}
    ]
    n_system_tokens = calc_n_tokens(system_prompt)
    n_user_tokens = calc_n_tokens(user_prompt)
    n_input_tokens = n_system_tokens + n_user_tokens
    return messages1, n_input_tokens


SYSTEM_PROMPT = '''
You are an AI assistant that helps answer questions by searching through video transcripts. 
I have retrieved the transcripts most likely to answer the user's question.
Carefully read through the transcripts to find information that helps answer the question. 
Be brief - your response should not be more than two paragraphs.
Only use information directly stated in the provided transcripts to answer the question. 
Do not add any information or make any claims that are not explicitly supported by the transcripts.
If the transcripts do not contain enough information to answer the question, state that you do not have enough information to provide a complete answer.
Format the response clearly.  If only one of the transcripts answers the question, don't reference the other and don't explain why its content is irrelevant.
Do not speak in the first person. DO NOT write a letter, make an introduction, or salutation
'''

def make_user_prompt(question, keep_texts):
    user_prompt = f'''
Question: {question}
==============================
'''
    list_strs = []
    for i, tx_val in enumerate(keep_texts.values()):
        text0 = tx_val['text']
        speaker_name = tx_val['Speaker']
        list_strs.append(f'Video Transcript {i+1}\nSpeaker: {speaker_name}\n{text0}')
    user_prompt += '\n---\n'.join(list_strs)
    user_prompt += '''
==============================
After analyzing the above video transcripts, please provide a helpful answer to my question. Remember to stay within two paragraphs
Address the response to me directly.  Do not use any information not explicitly supported by the transcripts.'''
    return user_prompt

def do_1_query_stream(messages1: dict, gen_client: OpenAI | Groq) -> tuple[str, float]:

    if isinstance(gen_client, OpenAI):
        # model1 = 'gpt-4-turbo'
        model1 = 'gpt-4o'
        # model1 = 'gpt-3.5-turbo'
    elif isinstance(gen_client, Groq):
        model1 = 'llama3-8b-8192'
    else:
        logger.error("There is some problem with the generator client")
        raise Exception("There is some problem with the generator client")
    response1 = gen_client.chat.completions.create(
        messages=messages1,
        model=model1,
        seed=18,
        temperature=0,
        stream=True
    )

    return response1