import streamlit as st
import logging
from .retrieval import do_retrieval
from .generation import do_stream_generation
from .setup_load import load_oai_model
from .openai_code import calc_n_tokens, calc_cost, OpenAI

logger = logging.getLogger()

def do_and_display_generation(query1: str, keep_texts: dict, openai_client: OpenAI, cost_cents_ret: int) -> None:
    response, prompt_tokens = do_stream_generation(query1, keep_texts, openai_client)
    text_out = st.write_stream(response)
    completion_tokens = calc_n_tokens(text_out)
    cost_cents_gen = calc_cost(prompt_tokens, completion_tokens)
    cost_cents = cost_cents_ret + cost_cents_gen
    st.caption(f'This cost approximately {cost_cents:.01f}¢')

def display_context(keep_texts: dict) -> None:
    """Displays the chatbot response and optionally the retrieved sources and context.

    Args:
        text_out: The generated chatbot response.
        sources: A list of source URLs.
        context0: The retrieved context used for generation.
        display_sources: Whether to display sources and context (default: False).
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
    """Creates the core Streamlit application for the knowledge base QA system.

    Args:
        n_results: The number of documents to retrieve.
        display_sources: Whether to display retrieved sources and context 
                         (default: False).
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
        if len(query1)<2:
            logger.error("You need to ask a question to get an answer")
            st.error("You need to ask a question to get an answer")
        else:
            st.header("Chatbot Response")
            with st.spinner('''Please be patient. Our LLM is taking a while to get an answer'''):
                logger.info(f"Received query: {query1}")
                openai_client = load_oai_model()
                keep_texts, cost_cents_ret = do_retrieval(query0=query1, n_results=n_results, openai_client=openai_client)
                out_container = st.container()
                display_context(keep_texts)
                with out_container:
                    do_and_display_generation(query1, keep_texts, openai_client, cost_cents_ret)
