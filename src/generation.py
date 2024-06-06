import logging
from .setup_load import OpenAI, Groq
from .utils import calc_n_tokens
from langsmith import traceable

logger = logging.getLogger()

SYSTEM_PROMPT = '''
You are an AI assistant that helps answer questions by searching through video transcripts. 
I have retrieved the transcripts most likely to answer the user's question.
Carefully read through the transcripts to find information that helps answer the question. 
Be brief - your response should not be more than two paragraphs.
Only use information directly stated in the provided transcripts to answer the question. 
Do not add any information or make any claims that are not explicitly supported by the transcripts.
If the transcripts do not contain enough information to answer the question, state that you do not have enough information to provide a complete answer.
Format the response clearly.  If only one of the transcripts answers the question, don't reference the other and don't explain why its content is irrelevant.
Do not speak in the first person. DO NOT write a letter, make an introduction, or salutation.
Reference the speaker's name when you say what they said.
'''

def set_messages(system_prompt: str, user_prompt: str) -> tuple[list[dict[str, str]], int]:
    """
    Set the messages for the chat completion.

    Args:
        system_prompt (str): The system prompt.
        user_prompt (str): The user prompt.

    Returns:
        tuple[list[dict[str, str]], int]: A tuple containing the messages and the total number of input tokens.
    """
    messages1 = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ]
    n_system_tokens = calc_n_tokens(system_prompt)
    n_user_tokens = calc_n_tokens(user_prompt)
    n_input_tokens = n_system_tokens + n_user_tokens
    logger.info(f"System Prompt is {n_system_tokens} tokens, User Prompt is {n_user_tokens} tokens")
    return messages1, n_input_tokens

def make_user_prompt(question: str, keep_texts: dict[str, dict[str, str]], is_oai: bool = True) -> str:
    """
    Create the user prompt based on the question and the retrieved transcripts.

    Args:
        question (str): The user's question.
        keep_texts (dict[str, dict[str, str]]): The retrieved transcripts.

    Returns:
        str: The user prompt.
    """
    user_prompt = f'''
Question: {question}
==============================
'''
    if len(keep_texts) > 0:
        list_strs = []
        for i, tx_val in enumerate(keep_texts.values()):
            if is_oai:
                text0 = tx_val['text']
            else:
                text0 = tx_val['relevant_text']
            speaker_name = tx_val['Speaker']
            list_strs.append(f'Video Transcript {i+1}\nSpeaker: {speaker_name}\n{text0}')
        user_prompt += '\n---\n'.join(list_strs)
        user_prompt += '''
==============================
After analyzing the above video transcripts, please provide a helpful answer to my question. Remember to stay within two paragraphs
Address the response to me directly.  Do not use any information not explicitly supported by the transcripts. Remember to reference the speaker's name.'''
    else:
        # If no relevant transcripts are found, generate a default response
        user_prompt += "No relevant video transcripts were found.  Please just return a result that says something like 'I'm sorry, but the answer to [Question] was not found in the transcripts from the New York R Conference'"
    # logger.info(f'User prompt: {user_prompt}')
    return user_prompt

@traceable
def do_1_query(messages1: list[dict[str, str]], gen_client: OpenAI | Groq) -> str: 
    """
    Generate a response using the specified chat completion model.

    Args:
        messages1 (list[dict[str, str]]): The messages for the chat completion.
        gen_client (OpenAI): The generation client (OpenAI).

    Returns:
        Stream: The generated response stream.
    """
    if isinstance(gen_client, OpenAI):
        model1 = 'gpt-4o'
    elif isinstance(gen_client, Groq):
        model1 = 'llama3-8b-8192'
    else:
        logger.error("There is some problem with the generator client")
        raise Exception("There is some problem with the generator client")

    # Generate the response using the specified model
    response1 = gen_client.chat.completions.create(
        messages=messages1,
        model=model1,
        seed=18,
        temperature=0,
        stream=False
    )

    text1 = text_from_response(response1)

    return text1

def text_from_response(response1) -> str:
    return response1.choices[0].message.content


def do_generation(query1: str, keep_texts: dict, gen_client: OpenAI | Groq) -> tuple[str, int]:
    """
    Generate the chatbot response using the specified generation client.

    Args:
        query1 (str): The user's query.
        keep_texts (dict[str, dict[str, str]]): The retrieved relevant texts.
        gen_client (OpenAI): The generation client (OpenAI).

    Returns:
        tuple[Stream, int]: A tuple containing the generated response stream and the number of prompt tokens.
    """
    is_oai = True if isinstance(gen_client, OpenAI) else False
    user_prompt = make_user_prompt(query1, keep_texts=keep_texts, is_oai=is_oai)
    messages1, prompt_tokens = set_messages(SYSTEM_PROMPT, user_prompt)
    response = do_1_query(messages1, gen_client)

    return response, prompt_tokens
