# New York R Conference RAG app

Use Python 3.11 or 3.12:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app reads conference data from the existing `nyr-rag-app` S3 bucket using
Boto3's standard AWS credential configuration. Users can enter their OpenAI key
in the sidebar. The Groq option also requires server-side `OPENAI_API_KEY` and
`GROQ_API_KEY` environment variables (or an untracked `.env` file). Optional
LangSmith tracing continues to use the standard LangSmith environment settings.

## Dependency maintenance

`requirements.in` lists the packages used by the app plus security minimums;
`requirements.txt` pins their complete resolved dependency graph. NumPy stays on
1.26 and pandas on 2.x to preserve compatibility with existing serialized data.
PyArrow stays on its patched 23.x line because PyArrow 26 requires NumPy 2 at
runtime. To regenerate with [uv](https://docs.astral.sh/uv/):

```sh
uv pip compile requirements.in --python-version 3.11 --universal --upgrade --no-annotate --output-file requirements.txt
pip install -r requirements.txt
python -m pip check
LANGSMITH_TRACING=false python -P -m unittest discover -s tests -v
```

Tests use local data and mock cloud responses; they do not require AWS or API
credentials. They check Streamlit startup, OpenAI/Groq SDK and tracing-wrapper
compatibility, S3/Parquet data loading, retrieval, and streaming responses.
GitHub Actions runs them on Python 3.11 and 3.12 and audits every pinned package
with pip-audit. Dependabot groups related version and security updates; the
workflow also runs weekly to detect new advisories.
