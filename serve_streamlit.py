# ---
# deploy: true
# cmd: ["modal", "serve", "10_integrations/streamlit/serve_streamlit.py"]
# ---
#
# # Run and share Streamlit apps
#
# This example shows you how to run a Streamlit app with `modal serve`, and then deploy it as a serverless web app.
#
# ![example streamlit app](./streamlit.png)
#
# This example is structured as two files:
#
# 1. This module, which defines the Modal objects (name the script `serve_streamlit.py` locally).
# 2. `app.py`, which is any Streamlit script to be mounted into the Modal
# function ([download script](https://github.com/modal-labs/modal-examples/blob/main/10_integrations/streamlit/app.py)).

import shlex
import subprocess
from pathlib import Path

import modal

# ## Define container dependencies
#
# The `app.py` script imports three third-party packages, so we include these in the example's
# image definition.

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "tiktoken~=0.7.0",
    "openai~=1.28.1",
    "groq~=0.5.0",
    "langsmith~=0.1.73",
    "streamlit~=1.34.0",
    "boto3~=1.34.133", 
    "botocore~=1.34.133", 
    "pandas~=2.2.2",
    "python-dotenv~=1.0.1"
)

app = modal.App(name="nyr-rag-app", image=image)

# ## Mounting the `app.py` script
#
# We can just mount the `app.py` script inside the container at a pre-defined path using a Modal
# [`Mount`](https://modal.com/docs/guide/local-data#mounting-directories).

streamlit_script_local_path = Path(__file__).parent / "app.py"
streamlit_script_remote_path = Path("/root/app.py")

if not streamlit_script_local_path.exists():
    raise RuntimeError(
        "app.py not found! Place the script with your streamlit app in the same directory."
    )

streamlit_script_mount = modal.Mount.from_local_file(
    streamlit_script_local_path,
    streamlit_script_remote_path,
)

streamlit_hs_local_path = Path(__file__).parent / "AJF_Headshot.jpg"
streamlit_hs_remote_path = Path("/root/AJF_Headshot.jpg")

if not streamlit_hs_local_path.exists():
    raise RuntimeError(
        "headshot not found! Place the script with your streamlit app in the same directory."
    )

streamlit_hs_mount = modal.Mount.from_local_file(
    streamlit_hs_local_path,
    streamlit_hs_remote_path,
)


streamlit_env_local_path = Path(__file__).parent / ".env"
streamlit_env_remote_path = Path("/root/.env")

if not streamlit_env_local_path.exists():
    raise RuntimeError(
        "env not found! Place the script with your streamlit app in the same directory."
    )

streamlit_env_mount = modal.Mount.from_local_file(
    streamlit_env_local_path,
    streamlit_env_remote_path,
)

streamlit_src_local_path = Path(__file__).parent / "src"
streamlit_src_remote_path = Path("/root/src")

if not streamlit_src_local_path.exists():
    raise RuntimeError(
        "src not found! Place the script with your streamlit app in the same directory."
    )

streamlit_src_mount = modal.Mount.from_local_dir(
    local_path=streamlit_src_local_path,
    remote_path=streamlit_src_remote_path,
)


# ## Spawning the Streamlit server
#
# Inside the container, we will run the Streamlit server in a background subprocess using
# `subprocess.Popen`. We also expose port 8000 using the `@web_server` decorator.


@app.function(
    allow_concurrent_inputs=100,
    mounts=[streamlit_script_mount, streamlit_src_mount, streamlit_hs_mount, streamlit_env_mount],
)
@modal.web_server(8000)
def run():
    target = shlex.quote(str(streamlit_script_remote_path))
    cmd = f"streamlit run {target} --server.port 8000 --server.enableCORS=false --server.enableXsrfProtection=false"
    subprocess.Popen(cmd, shell=True)


# ## Iterate and Deploy
#
# While you're iterating on your screamlit app, you can run it "ephemerally" with `modal serve`. This will
# run a local process that watches your files and updates the app if anything changes.
#
# ```shell
# modal serve serve_streamlit.py
# ```
#
# Once you're happy with your changes, you can deploy your application with
#
# ```shell
# modal deploy serve_streamlit.py
# ```
#
# If successful, this will print a URL for your app, that you can navigate to from
# your browser 🎉 .
