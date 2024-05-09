import streamlit as st
import logging
from .workflow import do_rag
from .utils import process_content_for_printing

logger = logging.getLogger(__name__)

def display_results(
    text_out: str, 
    sources: list[str], 
    context0: str, 
    display_sources: bool = False
) -> None:
    """Displays the chatbot response and optionally the retrieved sources and context.

    Args:
        text_out: The generated chatbot response.
        sources: A list of source URLs.
        context0: The retrieved context used for generation.
        display_sources: Whether to display sources and context (default: False).
    """

    st.header("Chatbot Response")
    st.markdown(text_out)
    if display_sources:
        st.divider()
        st.header("Retrieved Sources")
        st.subheader("Links")
        st.markdown('\n\n'.join(set(sources)))  # Display unique source URLs
        st.subheader("Retrieved Content")
        context1 = process_content_for_printing(context0)
        st.markdown(context1)


def make_app(
    model_name: str, 
    model: str, 
    n_results: int,  
    display_sources: bool = False 
) -> None:
    """Creates the core Streamlit application for the knowledge base QA system.

    Args:
        model_name: The name of the embedding model used for retrieval.
        model: The name of the generative model.
        n_results: The number of documents to retrieve.
        display_sources: Whether to display retrieved sources and context 
                         (default: False).
    """
    logger.info("Start building streamlit app")
# Configure Streamlit page settings
    st.set_page_config(
        page_title='Initial RAG - Q&A on Knowledge Base', 
        page_icon='favicon_io/favicon.ico',
        layout="wide",
        initial_sidebar_state="auto",
        menu_items=None
    )


    st.title("Question the Knowledge Base with Generative AI")

    st.markdown("What question do you want to ask of the knowledge base?")
    query1 = st.text_input(
        label='Question:',
        placeholder='e.g. What is a POAM?',
        key='input1',
        type='default'
    )

    run_rag = st.button(
        label = 'Ask the AI my question',
        key='button1',
    )

    if run_rag:
        if len(query1)<2:
            logger.error("You need to ask a question to get an answer")
            st.error("You need to ask a question to get an answer")
        else:
            with st.spinner('''\
        Please be patient. Our 🦙 is taking a while to get an answer'''):
                try:
                    text_out, context0, sources = do_rag(
                        query0=query1, n_results=n_results, model_name=model_name, model=model
                    )
                except Exception as e:
                    st.error(f"An error {e} occurred while processing your request.")  # User-friendly error

                display_results(text_out, sources, context0, display_sources=display_sources)

# if __name__ == "__main__":
#     logger = logging.getLogger(__name__)
