"""Request handling of the HTTP server without a model: option names reach the prompt, and responses and the model
list conform to the official System One OpenAPI schema (tests/data/systemone_openapi.json)."""
import asyncio
import json
from pathlib import Path

import jsonschema
import pytest

from basal.server import Server, models_payload, named_options, to_items

SPEC = json.loads((Path(__file__).parent / "data/systemone_openapi.json").read_text())


def schema(name):
    s = json.loads(json.dumps(SPEC["components"]["schemas"][name]).replace("#/components/schemas/", "#/$defs/"))
    s["$defs"] = json.loads(json.dumps(SPEC["components"]["schemas"]).replace("#/components/schemas/", "#/$defs/"))
    return s


class Tok:  # character-level stand-in for the tokenizer
    def apply_chat_template(self, msgs, **_):
        return "".join(f"<{m['role']}>{m['content']}" for m in msgs)

    def __call__(self, text, add_special_tokens=False):
        return type("E", (), {"input_ids": [ord(c) for c in text]})()


class FakeServer(Server):
    """Server.decide with a stub backend that returns fixed position probabilities for every job."""
    def __init__(self, pos_probs):
        self.name, self.temps, self.tok, self.letters, self.pos_probs = "basal-test", {}, Tok(), {}, pos_probs
        self.orders = 2
        self.backend = type("B", (), {"policies": {}})()

    async def _run(self, body):
        self.queue = asyncio.Queue()

        async def worker():
            while True:
                prompts, ids, fut, _ = await self.queue.get()
                fut.set_result([self.pos_probs[: len(i)] for i in ids])
        w = asyncio.get_running_loop().create_task(worker())
        try:
            return await self.decide(body)
        finally:
            w.cancel()


def test_named_options_keep_keys_when_needed():
    assert named_options({"returns": "Returns", "it": "IT"}) == (["returns", "it"], ["Returns", "IT"])
    assert named_options({"option_1": "Yes", "opcja_2": "Nie", "0": "x"})[1] == ["Yes", "Nie", "x"]  # placeholders
    assert named_options({"in_time": "Filed in time"})[1] == ["Filed in time"]           # key words in text
    assert named_options({"late": "Filed after the deadline"})[1] == ["late: Filed after the deadline"]
    keys, opts = named_options({"approve": {"requires_manager": False}, "reject": {"requires_manager": False}})
    assert opts == ['approve: {"requires_manager": false}', 'reject: {"requires_manager": false}']
    assert named_options({"a": "same", "b": "same"})[1] == ["a: same", "b: same"]
    assert named_options({"x": None, "y": "why"})[1] == ["x", "why"]


def test_structured_choices_give_distinct_prompts():
    q = to_items("Action requested: reject.", {"action": {"type": "choice", "instructions": "Return the action.",
                 "criteria": {"approve": {"requires_manager": False}, "reject": {"requires_manager": False}}}})[0]
    assert len(set(q["options"])) == 2 and q["options"][1].startswith("reject")


@pytest.mark.parametrize("q", [
    {"type": "choice", "instructions": "Dept?", "criteria": {"returns": "Returns", "it": "IT"}},
    {"type": "noul", "instructions": "Damaged?"},
    {"type": "score", "instructions": "Urgency?", "criteria": ["low", "mid", "high"]},
])
def test_response_conforms_to_official_schema(q):
    srv = FakeServer([0.7, 0.2, 0.1])
    r = asyncio.run(srv._run({"model": "basal", "state": "Parcel arrived damaged.", "questions": {"q": q}}))
    jsonschema.validate(r, schema("SystemOneResponse"))
    assert isinstance(r["usage"]["input_tokens"], int)


def test_model_list_conforms_to_official_schema():
    jsonschema.validate(models_payload("basal-1.0-4.5B", "fast", {"0.99": None}), schema("ModelMetadataList"))


def test_semantic_keys_of_unique_strings_reach_the_prompt():
    """Swapping which key owns which description must change what the model sees (regression)."""
    a = to_items("Action requested: reject.", {"q": {"type": "choice", "instructions": "Return the requested action.",
                 "criteria": {"approve": "Handled by Alice", "reject": "Handled by Bob"}}})[0]
    b = to_items("Action requested: reject.", {"q": {"type": "choice", "instructions": "Return the requested action.",
                 "criteria": {"reject": "Handled by Alice", "approve": "Handled by Bob"}}})[0]
    assert a["options"] == ["approve: Handled by Alice", "reject: Handled by Bob"]
    assert b["options"] == ["reject: Handled by Alice", "approve: Handled by Bob"]
    s = to_items("x", {"q": {"type": "score", "instructions": "Level?", "criteria": {"low": "minor", "high": "severe"}}})[0]
    assert s["options"] == ["low: minor", "high: severe"]
