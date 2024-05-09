import logging
from src.logging_setup import setup_logging
from src.display import make_app

if __name__ == "__main__":

    # At the beginning of your Streamlit app script
    setup_logging()
    logger = logging.getLogger(__name__)

    # Initialize variables and models
    model_name = "avsolatorio/GIST-Embedding-v0"
    model='mistral'
    n_results = 4
    
    # run app
    make_app(model_name=model_name, model=model, n_results=n_results, display_sources=True)
