"""Exercise upgraded SDKs and the app without cloud credentials or API calls."""
import io
import os
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import generation, retrieval, setup_load


class CompatibilityTests(unittest.TestCase):
    def test_app_starts_without_cloud_credentials(self):
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        self.assertEqual([], [error.message for error in app.exception])
        self.assertEqual(1, len(app.warning))

    def test_clients_initialize_with_current_tracing_wrapper(self):
        client = setup_load.load_oai_model("test-placeholder")
        self.assertIsInstance(client, setup_load.OpenAI)
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-placeholder"}):
            self.assertIsInstance(setup_load.load_groq_model(), setup_load.Groq)

    def test_s3_data_round_trip_and_retrieval(self):
        talks = pd.DataFrame([{"id0": "talk", "Speaker": "Test speaker"}])
        content = {
            "talks_on_youtube.parquet": talks.to_parquet(index=False),
            "transcripts.pkl": pickle.dumps({"talk": {
                "text": "A complete talk", "segments": {
                    0: {"segment_text": "First segment"},
                    1: {"segment_text": "Second segment"},
                },
            }}),
            "transcripts_40seconds.pkl": pickle.dumps({"talk": {
                "chunk": {"segment_start": 12, "segment_ids": [0, 1]},
            }}),
            "full_embeds.pkl": pickle.dumps({
                "abstract": {"talk": np.array([1.0, 0.0])},
                "seg": {"talk": {"chunk": np.array([1.0, 0.0])}},
            }),
        }
        s3 = MagicMock()
        s3.get_object.side_effect = lambda **kwargs: {"Body": io.BytesIO(content[kwargs["Key"]])}
        setup_load.import_data.clear()
        with patch.object(setup_load, "initialize_boto3", return_value=s3):
            loaded = setup_load.import_data()
        setup_load.import_data.clear()
        pd.testing.assert_frame_equal(loaded[0], talks)
        client = MagicMock(spec=setup_load.OpenAI)
        client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=[1.0, 0.0])]
        )
        with patch.object(retrieval, "import_data", return_value=loaded):
            matches = retrieval.do_retrieval("question", 4, client)
        self.assertEqual("First segment Second segment", matches["talk"]["relevant_text"])
        self.assertEqual(12, matches["talk"]["best_video_start"])

    def test_chat_calls_accept_current_sdk_and_stream_formats(self):
        messages = [{"role": "user", "content": "Test"}]
        openai_client = MagicMock(spec=setup_load.OpenAI)
        openai_client.chat.completions.create.return_value = iter(["answer"])
        self.assertEqual(["answer"], list(generation.do_1_query(messages, openai_client)))
        groq_client = MagicMock(spec=setup_load.Groq)
        groq_client.chat.completions.create.return_value = iter([
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="answer"))]),
        ])
        self.assertEqual(["", "answer"], list(generation.do_1_query(messages, groq_client)))


if __name__ == "__main__":
    unittest.main()
