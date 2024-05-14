import streamlit as st
import logging
from .retrieval import do_retrieval
from .generation import do_stream_generation, Stream
from .setup_load import load_api_clients
from .utils import calc_cost, calc_n_tokens

logger = logging.getLogger()

def display_stream_generation(stream_response: Stream) -> int:
    """
    Display the chatbot response.

    Args:
        stream_response (Stream): The Stream generator object.

    Returns:
        int: The number of completion tokens.
    """
    text_out = st.write_stream(stream_response)
    completion_tokens = calc_n_tokens(text_out)
    return completion_tokens

def display_cost(cost_cents: float) -> None:
    """
    Display the cost of the retrieval and generation.

    Args:
        cost_cents(float): The cost in cents
    """
    st.caption(f'This cost approximately {cost_cents:.01f}¢')

def display_context(keep_texts: dict) -> None:
    """
    Display the RAG-identified relevant videos.

    Args:
        keep_texts (dict): The retrieved relevant texts.
    """
    st.divider()
    st.subheader('RAG-identified relevant videos')
    n_vids = len(keep_texts)
    size1 = min(1 / n_vids, 1 / 3)
    if n_vids == 1:
        _, vid_c1, _ = st.columns(3)
        vid_containers = [vid_c1]
    elif n_vids == 2:
        _, vid_c1, vid_c2, _ = st.columns([size1 / 2, size1, size1, size1 / 2])
        vid_containers = [vid_c1, vid_c2]
    else:
        vid_containers = st.columns([size1] * n_vids)
    for i, (vid_id, vid_info) in enumerate(keep_texts.items()):
        vid_container = vid_containers[i]
        with vid_container:
            st.markdown(f"**{vid_info['Title']}**\n\n*{vid_info['Speaker']}*\n\nYear: {vid_id[4:8]}")
            st.caption(f"Similarity Score: {100*vid_info['score']:.0f}/100")
            st.video(vid_info['VideoURL'])

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

    use_oai = True
    st.title("Chat With a Decade of Previous NYR Talks")

    st.markdown("What question do you want to ask of previous speakers?")

    placeholder1 = 'e.g. What is the tidyverse?'
    chat_container = st.container()
    videos_container = st.container()

    with chat_container:
        if prompt1 := st.chat_input(placeholder=placeholder1, key='input1'):
            with st.chat_message("user"):
                st.markdown(prompt1)
            ret_client, gen_client = load_api_clients()
            keep_texts = do_retrieval(query0=prompt1, n_results=n_results, api_client=ret_client)
            with videos_container:
                display_context(keep_texts)
            stream_response, prompt_tokens = do_stream_generation(prompt1, keep_texts, gen_client)
            with st.chat_message("assistant"):
                completion_tokens = display_stream_generation(stream_response)
                embedding_tokens = calc_n_tokens(prompt1)
                cost_cents = calc_cost(prompt_tokens, completion_tokens, embedding_tokens)
                display_cost(cost_cents)
