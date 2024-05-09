import streamlit as st
from pyprojroot import here
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# import chromadb
# from chromadb.utils import embedding_functions
import logging
from .utils import ld2dl

logger = logging.getLogger(__name__)


# @st.cache_resource(ttl=360) # Cache the ChromaDB client for efficiency
# def load_chromadb_client(folder_name: str = 'chromadb_004') -> chromadb.PersistentClient:
#     """Loads and initializes the ChromaDB client.

#     Args: 
#         folder_name: The name of the folder that contains the relevant
#             ChromaDB client (optional, defaults to 'chromadb_004').

#     Returns:
#         chromadb.PersistentClient: The initialized ChromaDB client.
#     """
#     try:
#         client = chromadb.PersistentClient(path=str(here() / folder_name)) 
#         logger.info("ChromaDB client loaded successfully")
#         return client
#     except Exception as e:
#         logger.error(f"Failed to load ChromaDB client: {e}")
#         raise  # Re-raise the exception to handle it elsewhere

# def choose_collection(
#     model_name: str, 
#     collection_name: str = 'ryskview-knowledgebase-v4',
#     folder_name: str = 'chromadb_004'
# ) -> chromadb.Collection:
#     """Selects a ChromaDB collection based on model and collection name.

#     Args:
#         model_name: The name of the embedding model for the collection.
#         collection_name: The name of the ChromaDB collection (optional, 
#             defaults to 'ryskview-knowledgebase-v3').
#         folder_name: The name of the folder that contains the relevant
#             ChromaDB client (optional, defaults to 'chromadb_003').

#     Returns:
#         chromadb.Collection: The specified ChromaDB collection ready for use.
#     """

#     client = load_chromadb_client(folder_name=folder_name)  # Load the ChromaDB client

#     emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(
#         model_name=model_name, 
#         normalize_embeddings=True  # Ensure embedding vectors are normalized
#     )

#     collection = client.get_collection(
#         collection_name, 
#         embedding_function=emb_func  # Assign the embedding function to the collection
#     )

#     return collection 

def make_str(elt0: dict) -> str:
    """Constructs a formatted string representation of a retrieved document.

    Args:
        elt0: A dictionary containing document metadata and text content.

    Returns:
        str: A formatted string with section, header, and document chunk.
    """   
    try:
        metadata0 = elt0['metadatas']
        text0 = elt0['documents']
        section0 = metadata0['section']
        header0 = metadata0['header']
        str0 = f'<h1>SECTION: {section0}</h1>\n<h2>HEADER: {header0}</h2>\nCHUNK: {text0}'
        return(str0)
    except KeyError as e:
        logger.error(f"Error constructing string from document: Missing key {e}")
        return "Error in document structure."

@st.cache_data
def load_pickle_db(file_name: str = 'pickle_db_01.pkl'):
    fp1 = here()/file_name
    with open(fp1, 'rb') as f1:
        pickle_db = pickle.load(f1)
    logger.info("Pickle DB loaded")
    return(pickle_db)

@st.cache_data
def load_sbert_model(model_name: str):
    return SentenceTransformer(model_name_or_path=model_name)

def embed_query(query0: str, model_name: str):
    sbert_model = load_sbert_model(model_name=model_name)
    query_embed = sbert_model.encode(query0, normalize_embeddings=True)
    return query_embed

def get_top_docs(pickle_db: list[dict], query_embed:np.ndarray, n_results:int) -> list[dict]:
    np_db = np.stack([e['embeddings'] for e in pickle_db])
    cos_sims = np.dot(np_db, query_embed)
    rank_ids = np.argsort(-cos_sims)
    top_idx = rank_ids[:n_results]
    top_elts = []
    for i in top_idx:
        elt0 = pickle_db[i]
        elt0['distance'] = cos_sims[i]
        top_elts.append(elt0)
    return top_elts



def do_retrieval(query0: str, n_results: int, model_name: str) -> tuple[str, list[str]]:
    """Retrieves relevant documents from the ChromaDB collection.

    Args:
        query0: The user's query.
        n_results: The number of documents to retrieve.
        model_name: The name of the embedding model for the collection.

    Returns:
        tuple[str, list[str]]:
            - A concatenated string of retrieved document content (docs).
            - A list of corresponding document sources (URLs).
    """
    logger.info(f"Starting document retrieval for query: {query0}")
    try: 
        # collection = choose_collection(model_name=model_name)
        # top_n = collection.query(query_texts=query0, n_results=n_results)
        # top_n = {k:v[0] for k, v in top_n.items() if v is not None}
        pickle_db = load_pickle_db(here()/'pickle_db_01.pkl')
        query_embed = embed_query(query0=query0, model_name=model_name)
        top_n = get_top_docs(pickle_db=pickle_db, query_embed=query_embed, n_results=n_results)
        top_n2 = ld2dl(top_n)
        logger.info(f"Document IDs: {', '.join(top_n2['id'])}")
        logger.info(f"Distances: {', '.join([str(k)[:6] for k in top_n2['distance']])}")
        # top_n2 = [dict(zip(top_n,t)) for t in zip(*top_n.values())]
        strings = [make_str(elt0=elt0) for elt0 in top_n]
        sources = [elt0['metadatas']['url'] for elt0 in top_n]
        docs = '\n\n---\n\n'.join(strings)
        logger.info(f"Document retrieval successful for query: {query0}")
        return(docs, sources)
    except Exception as e:
        logger.error(f"Error during document retrieval for query: {query0}, Error: {str(e)}")
        return ("", [])

