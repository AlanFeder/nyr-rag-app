import streamlit as st
import logging
from .workflow import do_rag
from .utils import process_content_for_printing

logger = logging.getLogger()

def display_results(text_out: str, keep_texts: dict, cost_cents: float) -> None:
    """Displays the chatbot response and optionally the retrieved sources and context.

    Args:
        text_out: The generated chatbot response.
        sources: A list of source URLs.
        context0: The retrieved context used for generation.
        display_sources: Whether to display sources and context (default: False).
    """

    st.header("Chatbot Response")
    st.markdown(text_out)

    st.caption(f'This cost approximately {cost_cents:.01f}¢')
    st.divider()
    st.subheader('RAG-identified relevant videos')
    n_vids = len(keep_texts)
    size1 = 100 / n_vids
    size2 = [size1] * n_vids
    vid_containers = st.columns(size2)
    for i, (vid_id, vid_info) in enumerate(keep_texts):
        vid_container = vid_containers[i]
        with vid_container:
            st.markdown(f"**{vid_info['Title']}**\n\n*{vid_info['Speaker']}*\n\nYear: {vid_info['id0'][4:8]}")
            st.caption(f"Similarity Score: {100*vid_info['score']:.0f}/100")
            st.video(vid_info['VideoURL'])



    # st.header("Retrieved Sources")
    # st.subheader("Links")
    # st.markdown('\n\n'.join(set(sources)))  # Display unique source URLs
    # st.subheader("Retrieved Content")
    # context1 = process_content_for_printing(context0)
    # st.markdown(context1)


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
    query1 = st.text_input(
        label='Question:',
        placeholder='e.g. What is the tidyverse?',
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
            with st.spinner('''\
        Please be patient. Our LLM is taking a while to get an answer'''):
                try:
                    text_out, keep_texts, cost_cents = do_rag(
                        query0=query1, n_results=n_results
                    )
                except Exception as e:
                    st.error(f"An error {e} occurred while processing your request.")  # User-friendly error

                display_results(text_out, keep_texts, cost_cents)
