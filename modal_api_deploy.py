from modal import Image, App, asgi_app, Secret, Mount
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

stub = App("nyr-rag-app")

image = Image.debian_slim().pip_install(
    "fastapi",
    "pydantic",
    "numpy",
    "pandas",
    "python-dotenv",
    "openai",
    "boto3",
    "pyarrow"
)

# Mount the src_api directory
src_api_path = os.path.join(os.path.dirname(__file__), "src_api")
src_api_mount = Mount.from_local_dir(src_api_path, remote_path="/root/src_api")

@stub.function(
    image=image,
    secrets=[Secret.from_name("nyr-rag-r")],
    mounts=[src_api_mount]
)
@asgi_app()
def fastapi_app():
    import sys
    sys.path.append("/root")  # Add the root directory to Python path
    from src_api.retrieval_api import app
    return app
