import logging
import numpy as np
import pandas as pd

from .utils import dict_to_list_and_array
from .setup_load import import_data, OpenAI, Cohere

logger = logging.getLogger()

def sort_docs(full_embeds_oai: dict, arr_q: np.ndarray):
    talk_embeds = full_embeds_oai['abstract']
    video_ids, arr_embed = dict_to_list_and_array(talk_embeds)
    cos_sims = np.dot(arr_embed, arr_q)
    best_match_video_ids = np.argsort(-cos_sims)
    top_vids = np.array(video_ids)[best_match_video_ids]
    df_sorted = pd.DataFrame({'id0':top_vids, 'score':-np.sort(-cos_sims)})
    return df_sorted

def limit_docs(df_sorted: pd.DataFrame, df_talks: pd.DataFrame, n_results: int, transcript_dicts: dict):
    df_sorted = df_sorted.merge(df_talks)
    df_top = df_sorted.iloc[:n_results].copy()
    keep_texts = df_top.set_index('id0').to_dict(orient='index')
    for id0 in keep_texts:
        keep_texts[id0]['text'] = transcript_dicts[id0]['text']
    return keep_texts


def do_1_embed(lt: str, emb_client: OpenAI | Cohere ) -> tuple[list[list[float]], int]:
    """
    Generate embeddings using the OpenAI API for a single list of texts.

    Args:
        lt (str): A text to generate embeddings for.
        openai_client (OpenAI): The OpenAI API client.

    Returns:
        tuple[list[list[float]], int]: A tuple containing the generated embeddings and the total number of tokens.
    """
    # import streamlit as st
    # st.write(type(emb_client))
    if isinstance(emb_client, OpenAI):
        embed_response = emb_client.embeddings.create(
            input=lt,
            model='text-embedding-3-small',
        )
        here_embed = np.array(embed_response.data[0].embedding)
        n_toks = embed_response.usage.total_tokens
    elif isinstance(emb_client, Cohere):
        co_response = emb_client.embed(
            texts=lt,
            model="embed-english-v3.0",
            input_type="search_document"
        )
        here_embed = np.array(co_response.embeddings).flatten()
        n_toks = co_response.meta.billed_units.input_tokens
    else:
        logger.error("There is some problem with the embedding client")
        raise Exception("There is some problem with the embedding client")
    logger.info(f'Embedded {lt}')
    return here_embed, n_toks

def do_retrieval(query0: str, n_results: int, api_client) -> tuple[str, list[str]]:
    """Retrieves relevant documents from the ChromaDB collection.

    Args:
        query0: The user's query.
        n_results: The number of documents to retrieve.

    Returns:
        tuple[str, list[str]]:
            - A concatenated string of retrieved document content (docs).
            - A list of corresponding document sources (URLs).
    """
    logger.info(f"Starting document retrieval for query: {query0}")
    try: 
        df_talks, transcript_dicts, full_embeds_oai = import_data()
        arr_q, n_emb_toks = do_1_embed(query0, api_client)
        df_sorted = sort_docs(full_embeds_oai, arr_q)
        keep_texts = limit_docs(df_sorted, df_talks, n_results, transcript_dicts)
        cost_cents = 2 * n_emb_toks / 10_000
    except Exception as e:
        logger.error(f"Error during document retrieval for query: {query0}, Error: {str(e)}")
        raise 
    return keep_texts, cost_cents


