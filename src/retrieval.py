from pyprojroot import here
import numpy as np
import pandas as pd

import logging
from .utils import dict_to_list_and_array
from .openai_code import do_1_oai_embed
from .setup_load import import_data
from openai import OpenAI

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

def do_retrieval(query0: str, n_results: int, openai_client: OpenAI) -> tuple[str, list[str]]:
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
        arr_q, n_emb_toks = do_1_oai_embed(query0, openai_client)
        df_sorted = sort_docs(full_embeds_oai, arr_q)
        keep_texts = limit_docs(df_sorted, df_talks, n_results, transcript_dicts)
        cost_cents = 2 * n_emb_toks / 10_000
    except Exception as e:
        logger.error(f"Error during document retrieval for query: {query0}, Error: {str(e)}")
        raise 
    return keep_texts, cost_cents
