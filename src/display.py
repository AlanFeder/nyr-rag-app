import streamlit as st
from typing import Any
import logging
from .retrieval import do_retrieval
from .generation import do_generation
from .setup_load import load_api_clients
from .utils import calc_cost, calc_n_tokens
import os

logger = logging.getLogger()

def display_generation(text_response: Any) -> int:
    """
    Display the chatbot response.

    Args:
        stream_response (Stream): The Stream generator object.

    Returns:
        int: The number of completion tokens.
    """

    text_out = st.write_stream(text_response)
    st.session_state['response_out'] = text_out
    logger.info("Printing completed")
    completion_tokens = calc_n_tokens(text_out)
    return completion_tokens

def display_cost(cost_cents: float) -> None:
    """
    Display the cost of the retrieval and generation.

    Args:
        cost_cents (float): The cost in cents.
    """
    st.caption(f'This cost approximately {cost_cents:.01f}¢')
    logger.info("Cost displayed")

def display_context(keep_texts: dict[str, dict[str, str]]) -> None:
    """
    Display the RAG-identified relevant videos.

    Args:
        keep_texts (dict[str, dict[str, str]]): The retrieved relevant texts.
    """
    st.divider()
    st.subheader('RAG-identified relevant videos')
    n_vids = len(keep_texts)
    if n_vids == 0:
        st.markdown("No relevant videos identified")
    elif n_vids == 1:
        _, vid_c1, _ = st.columns(3)
        vid_containers = [vid_c1]
    elif n_vids == 2:
        _, vid_c1, vid_c2, _ = st.columns([1/6, 1/3, 1/3, 1/6])
        vid_containers = [vid_c1, vid_c2]
    elif n_vids > 2:
        vid_containers = st.columns(n_vids)
    for i, (vid_id, vid_info) in enumerate(keep_texts.items()):
        vid_container = vid_containers[i]
        with vid_container:
            vid_title = vid_info['Title']
            vid_speaker = vid_info['Speaker']
            sim_score = 100 * vid_info['score']
            vid_url = vid_info['VideoURL']
            vid_start = int(vid_info['best_video_start'])
            st.markdown(f"**{vid_title}**\n\n*{vid_speaker}*\n\nYear: {vid_id[4:8]}")
            st.caption(f"Similarity Score: {sim_score:.0f}/100")
            st.video(vid_url, start_time=vid_start)
            with st.expander(label='Transcript', expanded=False):
                st.markdown(vid_info['text'])
    logger.info("Context displayed")


def display_footer() -> None:
    """
    Display the footer section of the Streamlit app.
    """
    st.divider()

    st.caption('''This streamlit app was created for Alan Feder's [talk at the 10th Anniversary New York R Conference](https://rstats.ai/nyr.html). \n\n The slides used are [here](https://bit.ly/nyr-rag). \n\n The Github repository that houses all the code is [here](https://github.com/AlanFeder/nyr-rag-app) -- feel free to fork it and use it on your own!''')

    st.divider()

    st.subheader('Contact me!')
    st.image('AJF_Headshot.jpg', width=60)
    st.markdown('[Email](mailto:AlanFeder@gmail.com) | [Website](https://www.alanfeder.com/) | [LinkedIn](https://www.linkedin.com/in/alanfeder/) | [GitHub](https://github.com/AlanFeder)')
    logger.info("footer displayed")

def make_app(n_results: int) -> None:
    """
    Create the core Streamlit application for the knowledge base QA system.

    Args:
        n_results (int): The number of documents to retrieve.
    """
    logger.info("Start building streamlit app")

    # Configure Streamlit page settings
    st.set_page_config(
        page_title='RAG-time in the Big Apple', 
        page_icon='favicon_io/favicon.ico',
        layout="wide",
        initial_sidebar_state="auto",
        menu_items=None
    )
    with st.sidebar: 
        use_oai = st.radio(
            label = "Do you want to generate with GPT (More accurate) or Open-source Llama3 (free)?",
            options = [True, False],
            format_func=lambda x: 'ChatGPT' if x else 'Llama3-8b',
            index=0)
        if use_oai:
            openai_api_key = st.text_input(
                label="Input your OpenAI API Key (don't worry, this isn't stored anywhere)",
                type='password'
            )
            st.markdown('''If you don't have an OpenAI API key, you can sign up [here](https://platform.openai.com/account/api-keys).''')
        else:
            openai_api_key = None
            st.markdown("Rate limits may be applied to this app due to its use of [Groq](https://groq.com/)")

    st.title("Chat With a Decade of Previous NYR Talks")

    placeholder1 = 'e.g. What is the tidyverse?'
    chat_container = st.container()
    videos_container = st.container()
    footer_container = st.container()

    with footer_container:
        display_footer()

    if use_oai and not openai_api_key:
        with chat_container:
            st.warning("Please either put in an OpenAI API Key (and then press enter) or just use Llama3-8b, which isn't quite as good but is good enough.")
    else:
        st.markdown("What question do you want to ask of previous speakers?")

        with chat_container:
            if prompt1 := st.chat_input(placeholder=placeholder1, key='input1'):
                st.session_state['prompt'] = prompt1
                ret_client, gen_client = load_api_clients(use_oai=use_oai, openai_api_key=openai_api_key)
                keep_texts = do_retrieval(query0=prompt1, n_results=n_results, api_client=ret_client)
                st.session_state['keep_texts'] = keep_texts
                stream_response, prompt_tokens = do_generation(prompt1, keep_texts, gen_client)

                with st.chat_message("user"):
                    st.markdown(st.session_state['prompt'])

                with videos_container:
                    display_context(st.session_state['keep_texts'])

                with st.chat_message("assistant"):
                    completion_tokens = display_generation(stream_response)
                embedding_tokens = calc_n_tokens(prompt1)
                if not use_oai:
                    completion_tokens = prompt_tokens = 0
                cost_cents = calc_cost(prompt_tokens, completion_tokens, embedding_tokens)
                st.session_state['cost_cents'] = cost_cents
                display_cost(cost_cents)

    logger.info("You're done!")
