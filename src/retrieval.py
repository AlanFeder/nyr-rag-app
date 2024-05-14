import logging
import numpy as np
import pandas as pd

from .utils import dict_to_list_and_array
from .setup_load import import_data, OpenAI

logger = logging.getLogger()

def sort_docs(full_embeds_oai: dict, arr_q: np.ndarray) -> pd.DataFrame:
    """
    Sort documents based on their cosine similarity to the query embedding.

    Args:
        full_embeds_oai (dict): Dictionary containing document embeddings.
        arr_q (np.ndarray): Query embedding.

    Returns:
        pd.DataFrame: Sorted dataframe containing document IDs and similarity scores.
    """
    # Extract talk embeddings and convert to list and array
    talk_embeds = full_embeds_oai['abstract']
    video_ids, arr_embed = dict_to_list_and_array(talk_embeds)

    # Calculate cosine similarities between query embedding and document embeddings
    cos_sims = np.dot(arr_embed, arr_q)

    # Get the indices of the best matching video IDs
    best_match_video_ids = np.argsort(-cos_sims)

    # Get the top video IDs based on the best match indices
    top_vids = np.array(video_ids)[best_match_video_ids]

    # Create a sorted dataframe with video IDs and similarity scores
    df_sorted = pd.DataFrame({'id0': top_vids, 'score': -np.sort(-cos_sims)})

    return df_sorted

def limit_docs(df_sorted: pd.DataFrame, df_talks: pd.DataFrame, n_results: int, transcript_dicts: dict) -> dict:
    """
    Limit the retrieved documents based on a score threshold and return the top documents.

    Args:
        df_sorted (pd.DataFrame): Sorted dataframe containing document IDs and similarity scores.
        df_talks (pd.DataFrame): Dataframe containing talk information.
        n_results (int): Number of top documents to retrieve.
        transcript_dicts (dict): Dictionary containing transcript text for each document ID.

    Returns:
        dict: Dictionary containing the top documents with their IDs, scores, and text.
    """
    # Merge the sorted dataframe with the talks dataframe
    df_sorted = df_sorted.merge(df_talks)

    # Get the top n_results documents
    df_top = df_sorted.iloc[:n_results].copy()

    # Get the top score and calculate the score threshold
    top_score = df_top['score'].iloc[0]
    score_thresh = min(0.6, top_score - 0.05)

    # Filter the top documents based on the score threshold
    df_top = df_top.loc[df_top['score'] > score_thresh]

    # Create a dictionary of the top documents with their IDs, scores, and text
    keep_texts = df_top.set_index('id0').to_dict(orient='index')
    for id0 in keep_texts:
        keep_texts[id0]['text'] = transcript_dicts[id0]['text']

    return keep_texts

def do_1_embed(lt: str, emb_client: OpenAI) -> tuple[np.ndarray, int]:
    """
    Generate embeddings using the OpenAI API for a single text.

    Args:
        lt (str): A text to generate embeddings for.
        emb_client (OpenAI ): The embedding API client (OpenAI ).

    Returns:
        tuple[np.ndarray, int]: A tuple containing the generated embeddings and the total number of tokens.
    """
    if isinstance(emb_client, OpenAI):
        # Generate embeddings using OpenAI API
        embed_response = emb_client.embeddings.create(
            input=lt,
            model='text-embedding-3-small',
        )
        here_embed = np.array(embed_response.data[0].embedding)
        n_toks = embed_response.usage.total_tokens
    else:
        logger.error("There is some problem with the embedding client")
        raise Exception("There is some problem with the embedding client")
    
    logger.info(f'Embedded {lt}')
    return here_embed, n_toks

def do_retrieval(query0: str, n_results: int, api_client: OpenAI) -> tuple[dict, float]:
    """
    Retrieve relevant documents based on the user's query.

    Args:
        query0 (str): The user's query.
        n_results (int): The number of documents to retrieve.
        api_client (OpenAI): The API client (OpenAI ) for generating embeddings.

    Returns:
        tuple[dict, float]: A tuple containing the retrieved documents and the cost in cents.
    """
    logger.info(f"Starting document retrieval for query: {query0}")
    try:
        # Import data
        df_talks, transcript_dicts, full_embeds_oai = import_data()
        
        # Generate embeddings for the query
        arr_q, n_emb_toks = do_1_embed(query0, api_client)
        
        # Sort documents based on their cosine similarity to the query embedding
        df_sorted = sort_docs(full_embeds_oai, arr_q)
        
        # Limit the retrieved documents based on a score threshold
        keep_texts = limit_docs(df_sorted, df_talks, n_results, transcript_dicts)
        
        # Calculate the cost in cents
        cost_cents = 2 * n_emb_toks / 10_000
    except Exception as e:
        logger.error(f"Error during document retrieval for query: {query0}, Error: {str(e)}")
        raise

    return keep_texts, cost_cents
