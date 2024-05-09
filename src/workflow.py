import logging
from .retrieval import do_retrieval
from .generation import do_generation

logger = logging.getLogger(__name__)

def do_rag(query0: str, n_results: int, model_name: str, model: str) -> tuple[str, str, list[str]]:
    """Performs retrieval and generation (RAG) to answer the user's query.

    Args:
        query0: The user's query.
        n_results: The number of documents to retrieve.
        model_name: The name of the embedding model for the collection.
        model: The name of the language generation model to use.

    Returns:
        tuple[str, str, list[str]]:
            - gen_text: The generated response to the query.
            - context0: The raw retrieved context used for generation.
            - sources: A list of source URLs for the retrieved context.
    """
    # try:
    logger.info(f"Received query: {query0}")
    context0, sources = do_retrieval(query0=query0, n_results=n_results, model_name=model_name)
    gen_text = do_generation(query0=query0, context0=context0, model=model)
    logger.info(f"RAG process completed successfully")
    return gen_text, context0, sources
    # except Exception as e:
        # logger.error(f"Error processing query: {query0}. Context was {context0}, Error: {str(e)}")


# if __name__ == '__main__':
#     logger = logging.getLogger(__name__)
