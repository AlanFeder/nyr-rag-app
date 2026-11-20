"""Verify application requests against the upgraded SDK without network access."""
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import httpx2
import numpy as np
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import generation, retrieval, setup_load
from langsmith.wrappers import wrap_openai


class SDKTransportTests(unittest.TestCase):
    @patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
    def test_embedding_and_streaming_with_the_real_sdk(self):
        requests = []

        def respond(request):
            self.assertEqual("Bearer offline-placeholder", request.headers["authorization"])
            body = json.loads(request.content)
            requests.append((request.url.path, body))
            if request.url.path == "/v1/embeddings":
                return httpx2.Response(200, json={
                    "object": "list",
                    "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 0.0]}],
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                })
            self.assertEqual("/v1/chat/completions", request.url.path)
            self.assertTrue(body["stream"])
            event = {
                "id": "chatcmpl-offline", "object": "chat.completion.chunk",
                "created": 0, "model": body["model"],
                "choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": None}],
            }
            return httpx2.Response(
                200, headers={"content-type": "text/event-stream"},
                content=(
                    "data: " + json.dumps({**event, "choices": [
                        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                    ]}) + "\n\n"
                    "data: " + json.dumps(event) + "\n\n"
                    "data: " + json.dumps({**event, "choices": [], "usage": {
                        "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2
                    }}) + "\n\ndata: [DONE]\n\n"
                ).encode(),
            )

        with OpenAI(
            api_key="offline-placeholder", max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
        ) as client:
            client = wrap_openai(client)
            np.testing.assert_array_equal([1.0, 0.0], retrieval.do_1_embed("question", client))
            messages = [{"role": "user", "content": "question"}]
            chunks = list(generation.do_1_query(messages, client))
            self.assertEqual("answer", "".join(
                choice.delta.content or "" for chunk in chunks for choice in chunk.choices
            ))
        self.assertEqual("question", requests[0][1]["input"])
        self.assertEqual(messages, requests[1][1]["messages"])


