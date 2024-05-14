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

def display_cost(prompt_tokens: int, completion_tokens: int, cost_cents_ret: float) -> None:
    """
    Display the cost of the retrieval and generation.

    Args:
        prompt_tokens (int): The number of prompt tokens.
        completion_tokens (int): The number of completion tokens.
        cost_cents_ret (float): The cost in cents for the retrieval phase.
    """
    cost_cents_gen = calc_cost(prompt_tokens, completion_tokens)
    cost_cents = cost_cents_ret + cost_cents_gen
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
    size1 = 100 / n_vids
    size2 = [size1] * n_vids
    vid_containers = st.columns(size2)
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

    query1 = st.text_input(
        label='Question:',
        placeholder=placeholder1,
        key='input1',
        type='default'
    )

    run_rag = st.button(
        label = 'Ask my question',
        key='button1',
    )

    if run_rag:
        if len(query1) < 2:
            logger.error("You need to ask a question to get an answer")
            st.error("You need to ask a question to get an answer")
        else:
            st.header("Chatbot Response")
            logger.info(f"Received query: {query1}")
            ret_client, gen_client = load_api_clients(use_oai=use_oai)
            keep_texts, cost_cents_ret = do_retrieval(query0=query1, n_results=n_results, api_client=ret_client)
            out_container = st.container()
            display_context(keep_texts)
            stream_response, prompt_tokens = do_stream_generation(query1, keep_texts, gen_client)
            with out_container:
                completion_tokens = display_stream_generation(stream_response)
                display_cost(prompt_tokens, completion_tokens, cost_cents_ret)
