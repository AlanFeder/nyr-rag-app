import logging
from src.logging_setup import setup_logging
from src.display import make_app

if __name__ == "__main__":

    # At the beginning of your Streamlit app script
    setup_logging()
    logger = logging.getLogger()

    # Initialize variables and models
    n_results = 4
    
    # run app
    make_app(n_results=n_results, display_sources=True)
