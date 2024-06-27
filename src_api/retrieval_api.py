from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging
from .retrieval import do_retrieval
from .setup_load import load_oai_model
import numpy as np
from contextlib import asynccontextmanager


# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing application components...")
    # Add any necessary initialization code here
    # For example:
    # await database.connect()
    # load_models()
    logger.info("Application startup complete.")
    
    yield  # This is where the application runs
    
    # Shutdown
    logger.info("Shutting down application...")
    # Add any necessary cleanup code here
    # For example:
    # await database.disconnect()
    logger.info("Application shutdown complete.")

app = FastAPI(lifespan=lifespan)



class Query(BaseModel):
    query: str
    n_results: int

class RetrievalResponse(BaseModel):
    keep_texts: Dict[str, Dict[str, Any]]

    class Config:
        json_encoders = {
            np.integer: lambda x: int(x),
            np.floating: lambda x: float(x),
            np.ndarray: lambda x: x.tolist(),
        }

@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(query: Query):
    try:
        logger.info(f"Received query: {query.query}, n_results: {query.n_results}")

        ret_client = load_oai_model()
        keep_texts = do_retrieval(query.query, query.n_results, ret_client)
                   # Log the types of values in keep_texts
        for key, value in keep_texts.items():
            logger.debug(f"keep_texts[{key}] types: {type(value)}")
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    logger.debug(f"keep_texts[{key}][{sub_key}] type: {type(sub_value)}")
           
        return RetrievalResponse(keep_texts=keep_texts)
    except Exception as e:
        logger.error(f"Error during retrieval: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
