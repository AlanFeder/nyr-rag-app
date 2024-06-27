import logging
from itertools import chain
import numpy as np
import pandas as pd
from .utils import dict_to_list_and_array, split_into_consecutive
from .setup_load import import_data, OpenAI

logger = logging.getLogger(__name__)

def do_sort(embed_dict: dict[str, np.ndarray], arr_q: np.ndarray) -> pd.DataFrame:
    """
    Sort documents based on their cosine similarity to the query embedding.

    Args:
        embed_dict (dict[str, np.ndarray]): Dictionary containing document embeddings.
        arr_q (np.ndarray): Query embedding.

    Returns:
        pd.DataFrame: Sorted dataframe containing document IDs and similarity scores.
    """
    # Extract talk embeddings and convert to list and array
    video_ids, arr_embed = dict_to_list_and_array(embed_dict)

    # Calculate cosine similarities between query embedding and document embeddings
    cos_sims = np.dot(arr_embed, arr_q)

    # Get the indices of the best matching video IDs
    best_match_video_ids = np.argsort(-cos_sims)

    # Get the top video IDs based on the best match indices
    top_vids = np.array(video_ids)[best_match_video_ids]

    # Create a sorted dataframe with video IDs and similarity scores
    df_sorted = pd.DataFrame({'id0': top_vids, 'score': -np.sort(-cos_sims)})

    return df_sorted


def sort_docs(full_embeds: dict[str, dict[str, np.ndarray]], arr_q: np.ndarray) -> pd.DataFrame:
    """
    Sort documents based on their cosine similarity to the query embedding.

    Args:
        full_embeds (dict[str, dict[str, np.ndarray]]): Dictionary containing document embeddings.
        arr_q (np.ndarray): Query embedding.

    Returns:
        pd.DataFrame: Sorted dataframe containing document IDs and similarity scores.
    """
    # Extract talk embeddings and convert to list and array
    abstract_embeds = full_embeds['abstract']

    sorted_abstracts = do_sort(abstract_embeds, arr_q)

    logger.info("abstracts sorted")

    return sorted_abstracts

def sort_within_doc(full_embeds: dict[str, dict[str, np.ndarray]], arr_q: np.ndarray, video_id: str) -> pd.DataFrame:
    """
    Sort segments within a document based on their cosine similarity to the query embedding.

    Args:
        full_embeds (dict[str, dict[str, np.ndarray]]): Dictionary containing document embeddings.
        arr_q (np.ndarray): Query embedding.
        video_id (str): ID of the video/document.

    Returns:
        pd.DataFrame: Sorted dataframe containing segment IDs and similarity scores.
    """
    # Extract talk embeddings and convert to list and array
    seg_embeds = full_embeds['seg']
    these_seg_embeds = seg_embeds[video_id]

    sorted_segs = do_sort(these_seg_embeds, arr_q)

    logger.info(f'Segments sorted within {video_id}')

    return sorted_segs

def limit_docs(df_sorted: pd.DataFrame, df_talks: pd.DataFrame, n_results: int, transcript_dicts: dict[str, dict]) -> dict[str, dict]:
    """
    Limit the retrieved documents based on a score threshold and return the top documents.

    Args:
        df_sorted (pd.DataFrame): Sorted dataframe containing document IDs and similarity scores.
        df_talks (pd.DataFrame): Dataframe containing talk information.
        n_results (int): Number of top documents to retrieve.
        transcript_dicts (dict[str, dict]): Dictionary containing transcript text for each document ID.

    Returns:
        dict[str, dict]: Dictionary containing the top documents with their IDs, scores, and text.
    """
    # Merge the sorted dataframe with the talks dataframe
    df_sorted = df_sorted.merge(df_talks)

    # Get the top n_results documents
    df_top = df_sorted.iloc[:n_results].copy()

    # Get the top score and calculate the score threshold
    top_score = df_top['score'].iloc[0]
    score_thresh = max(min(0.6, top_score - 0.05), 0.2)

    # Filter the top documents based on the score threshold
    df_top = df_top.loc[df_top['score'] > score_thresh]

    # Create a dictionary of the top documents with their IDs, scores, and text
    keep_texts = df_top.set_index('id0').to_dict(orient='index')
    for id0 in keep_texts:
        keep_texts[id0]['text'] = transcript_dicts[id0]['text']

    logger.info(f"{len(keep_texts)} videos kept")

    return keep_texts

def do_1_embed(lt: str, emb_client: OpenAI) -> np.ndarray:
    """
    Generate embeddings using the OpenAI API for a single text.

    Args:
        lt (str): A text to generate embeddings for.
        emb_client (OpenAI): The embedding API client (OpenAI).

    Returns:
        np.ndarray: The generated embeddings.
    """
    if isinstance(emb_client, OpenAI):
        # Generate embeddings using OpenAI API
        embed_response = emb_client.embeddings.create(
            input=lt,
            model='text-embedding-3-small',
        )
        here_embed = np.array(embed_response.data[0].embedding)
    else:
        logger.error("There is some problem with the embedding client")
        raise Exception("There is some problem with the embedding client")
    
    logger.info(f'Embedded {lt}')
    return here_embed

def do_retrieval(query0: str, n_results: int, api_client: OpenAI) -> dict[str, dict]:
    """
    Retrieve relevant documents based on the user's query.

    Args:
        query0 (str): The user's query.
        n_results (int): The number of documents to retrieve.
        api_client (OpenAI): The API client (OpenAI) for generating embeddings.

    Returns:
        dict[str, dict]: The retrieved documents.
    """
    logger.info(f"Starting document retrieval for query: {query0}")
    try:
        # Import data
        df_talks, transcript_dicts, transcripts_40seconds, full_embeds = import_data()
        
        # Generate embeddings for the query
        arr_q = do_1_embed(query0, api_client)
        
        # Sort documents based on their cosine similarity to the query embedding
        df_sorted = sort_docs(full_embeds, arr_q)
        
        # Limit the retrieved documents based on a score threshold
        keep_texts = limit_docs(df_sorted, df_talks, n_results, transcript_dicts)
        n_vids = len(keep_texts)
        if n_vids >= 1:
            n_chunks_per_vid = 50 // n_vids
            
        for video_id in keep_texts:
            df_sorted_chunks = sort_within_doc(full_embeds, arr_q, video_id)
            tx_chunk_info = transcripts_40seconds[video_id]
            tx_segs_info = transcript_dicts[video_id]['segments']
            df_chunk_info = pd.DataFrame(tx_chunk_info).T.reset_index().rename({'index':'id0'}, axis=1)
            df_sorted_chunks = df_sorted_chunks.merge(df_chunk_info)
            df_keep_chunks = df_sorted_chunks.iloc[:n_chunks_per_vid].copy()
            top_chunk_start = df_keep_chunks.iloc[0]['segment_start']
            segment_ids = df_keep_chunks['segment_ids']
            unique_seg_ids = set(chain.from_iterable(segment_ids))
            seg_id_grps = split_into_consecutive(np.array(list(unique_seg_ids)))
            text_grps_list = []
            for seg_id_grp in seg_id_grps:
                text_seg_list = []
                for seg_id in seg_id_grp:
                    text_seg_list.append(tx_segs_info[seg_id]['segment_text'])
                text_seg = ' '.join(text_seg_list)
                text_grps_list.append(text_seg)
            relevant_text = '\n...\n'.join(text_grps_list)
            keep_texts[video_id]['best_video_start'] = top_chunk_start
            keep_texts[video_id]['relevant_text'] = relevant_text
    except Exception as e:
        logger.error(f"Error during document retrieval for query: {query0}, Error: {str(e)}")
        raise

    logger.info("Retrieval Done")

    return keep_texts

