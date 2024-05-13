import logging
from .retrieval import do_retrieval
from .generation import do_generation
from .setup_load import load_oai_model

logger = logging.getLogger()

def do_rag(query0: str, n_results: int) -> tuple[str, str, list[str]]:
    """Performs retrieval and generation (RAG) to answer the user's query.

    Args:
        query0: The user's query.
        n_results: The number of documents to retrieve.

    Returns:
        tuple[str, str, list[str]]:
            - gen_text: The generated response to the query.
            - context0: The raw retrieved context used for generation.
            - sources: A list of source URLs for the retrieved context.
    """
    logger.info(f"Received query: {query0}")
    openai_client = load_oai_model()
    keep_texts, cost_cents_emb = do_retrieval(query0=query0, n_results=n_results, openai_client=openai_client)
    gen_text, cost_cents_gen = do_generation(query0=query0, keep_texts=keep_texts, openai_client=openai_client)
    cost_cents = cost_cents_gen + cost_cents_emb
    logger.info(f"RAG process completed successfully")
    return gen_text, keep_texts, cost_cents
