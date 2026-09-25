from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from docchat.config import Settings
from docchat.llm import CALL_LOG, LLM, LLMError, extract_json


class MockOllama:
    def __init__(self, replies, caps=("completion", "vision", "thinking"), ctx_len=262144):
        self.replies = list(replies)
        self.caps = list(caps)
        self.ctx_len = ctx_len
        self.calls = []

    def show(self, model):
        return SimpleNamespace(
            capabilities=self.caps, modelinfo={"qwen35.context_length": self.ctx_len}
        )

    def chat(self, stream=False, **kwargs):
        self.calls.append(kwargs)
        text = self.replies.pop(0)
        if stream:
            words = text.split(" ")
            parts = [SimpleNamespace(message=SimpleNamespace(content=w + " ", thinking=None),
                                     prompt_eval_count=None, eval_count=None, done_reason=None)
                     for w in words[:-1]]  # fmt: skip
            parts.append(SimpleNamespace(message=SimpleNamespace(content=words[-1], thinking=None),
                                         prompt_eval_count=12, eval_count=len(words), done_reason="stop"))  # fmt: skip
            return iter(parts)
        return SimpleNamespace(message=SimpleNamespace(content=text, thinking="hmm"),
                               prompt_eval_count=12, eval_count=3, done_reason="stop")  # fmt: skip


def make(replies, **kw) -> tuple[LLM, MockOllama]:
    client = MockOllama(replies, **kw)
    return LLM(client, Settings(_env_file=None, num_ctx=8192)), client


def test_num_ctx_and_think_always_sent():
    llm, client = make(["<think>internal</think>Answer"])
    res = llm.chat([{"role": "user", "content": "hi"}], system="sys", think=False)
    sent = client.calls[0]
    assert sent["options"]["num_ctx"] == 8192
    assert sent["think"] is False
    assert sent["messages"][0] == {"role": "system", "content": "sys"}
    assert res.content == "Answer"  # stray think tags stripped


def test_think_not_sent_to_non_thinking_models():
    llm, client = make(["ok"], caps=("completion",))
    llm.chat([{"role": "user", "content": "hi"}], think=True)
    assert "think" not in client.calls[0]


def test_num_ctx_clamped_to_model_context():
    llm, client = make(["ok"], ctx_len=4096)
    llm.chat([{"role": "user", "content": "hi"}])
    assert client.calls[0]["options"]["num_ctx"] == 4096


def test_streaming_callbacks_and_call_log():
    llm, _ = make(["one two three"])
    tokens, log = [], []
    token = CALL_LOG.set(log)
    try:
        res = llm.chat([{"role": "user", "content": "hi"}], on_token=tokens.append, tag="synthesis")
    finally:
        CALL_LOG.reset(token)
    assert "".join(tokens) == "one two three" == res.content
    assert log[0].tag == "synthesis" and log[0].prompt_tokens == 12


class Answer(BaseModel):
    value: int


def test_json_retries_with_validation_error():
    llm, client = make(['{"value": "not a number"}', '```json\n{"value": 5}\n```'])
    out = llm.json([{"role": "user", "content": "q"}], Answer)
    assert out.value == 5
    assert client.calls[0]["format"] == Answer.model_json_schema()
    assert client.calls[0]["think"] is False and client.calls[0]["options"]["temperature"] == 0.0
    assert "not valid JSON" in client.calls[1]["messages"][-1]["content"]


def test_json_gives_up_after_two_attempts():
    llm, _ = make(["nope", "still nope"])
    with pytest.raises(LLMError):
        llm.json([{"role": "user", "content": "q"}], Answer)


def test_probe_downgrades_to_plain_json_mode():
    llm, _ = make(["bad", "bad"])
    assert llm.probe_json() is False
    assert llm.json_mode == "json"


def test_images_are_encoded(tmp_path):
    from PIL import Image

    img = tmp_path / "x.png"
    Image.new("RGB", (3000, 1000), "white").save(img)
    llm, client = make(["seen"])
    llm.chat([{"role": "user", "content": "what is this?"}], images=[img])
    encoded = client.calls[0]["messages"][-1]["images"][0]
    assert isinstance(encoded, str) and len(encoded) > 100


def test_extract_json_tolerates_chatter():
    assert extract_json('Sure! Here it is: {"a": 1} hope that helps') == {"a": 1}
    assert extract_json("<think>x</think>\n[1, 2]") == [1, 2]
