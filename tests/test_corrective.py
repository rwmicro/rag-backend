"""CorrectiveLoop unit tests.

Pure-Python: uses fakes for Chunk / ConfidenceEvaluator / LLMGenerator so the
loop can be exercised without torch/numpy/langchain in the env.
"""

import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.corrective import CorrectiveLoop


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeChunk:
    chunk_id: str
    content: str = ""


class _Level(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class _FakeEval:
    confidence_score: float
    confidence_level: _Level
    issues: List[str]


class _ScriptedEvaluator:
    """Yields a pre-canned evaluation for each call."""

    def __init__(self, scripted: List[_FakeEval]):
        self._scripted = list(scripted)
        self.calls: List[Tuple[List[Tuple[_FakeChunk, float]], str]] = []

    def evaluate(self, results, query, **_kwargs):
        self.calls.append((list(results), query))
        return self._scripted.pop(0)


class _ScriptedLLM:
    def __init__(self, response: str = "rewritten query"):
        self.response = response
        self.prompts: List[str] = []

    def generate(self, prompt: str, **_kwargs) -> str:
        self.prompts.append(prompt)
        return self.response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_high_confidence_skips_retry():
    """No rewrite when initial confidence is high."""
    initial = [(_FakeChunk("a"), 0.9), (_FakeChunk("b"), 0.7)]
    evaluator = _ScriptedEvaluator(
        [_FakeEval(0.85, _Level.HIGH, [])]
    )
    llm = _ScriptedLLM("should not be called")

    loop = CorrectiveLoop(evaluator=evaluator, llm_generator=llm, max_attempts=1)
    out, trace = loop.run("the query", initial, retry_retrieve=lambda q: [])

    assert out == initial
    assert trace.triggered is False
    assert trace.final_strategy == "initial"
    assert llm.prompts == []  # rewrite never called
    assert len(trace.attempts) == 1


def test_low_confidence_triggers_rewrite_and_merges():
    """Low confidence → LLM rewrite → retry → merged result list."""
    initial = [(_FakeChunk("a"), 0.4)]
    retry = [(_FakeChunk("b"), 0.8), (_FakeChunk("a"), 0.2)]
    evaluator = _ScriptedEvaluator(
        [
            _FakeEval(0.3, _Level.LOW, ["Low top score"]),
            _FakeEval(0.75, _Level.HIGH, []),
        ]
    )
    llm = _ScriptedLLM("specific rewritten query")

    loop = CorrectiveLoop(evaluator=evaluator, llm_generator=llm, max_attempts=1)
    out, trace = loop.run("vague q", initial, retry_retrieve=lambda q: retry)

    assert trace.triggered is True
    assert trace.rewritten_query == "specific rewritten query"
    assert trace.final_strategy == "merged"
    assert len(trace.attempts) == 2
    assert llm.prompts and "vague q" in llm.prompts[0]

    # max-merge: chunk "a" keeps 0.4 (initial > retry); "b" gets 0.8.
    by_id = {c.chunk_id: s for c, s in out}
    assert by_id == {"a": 0.4, "b": 0.8}
    # Sorted descending
    assert [c.chunk_id for c, _ in out] == ["b", "a"]


def test_max_attempts_zero_disables_loop():
    """max_attempts=0 means no retry even with low confidence."""
    initial = [(_FakeChunk("a"), 0.1)]
    evaluator = _ScriptedEvaluator([_FakeEval(0.1, _Level.VERY_LOW, ["nothing"])])
    llm = _ScriptedLLM()

    loop = CorrectiveLoop(evaluator=evaluator, llm_generator=llm, max_attempts=0)
    out, trace = loop.run("q", initial, retry_retrieve=lambda q: [])

    assert out == initial
    assert trace.triggered is False
    assert llm.prompts == []


def test_empty_initial_uses_retry_results():
    """When initial is empty, the retry results become the answer."""
    initial: list = []
    retry = [(_FakeChunk("x"), 0.6), (_FakeChunk("y"), 0.5)]
    evaluator = _ScriptedEvaluator(
        [
            _FakeEval(0.0, _Level.VERY_LOW, ["No results returned"]),
            _FakeEval(0.6, _Level.MEDIUM, []),
        ]
    )
    llm = _ScriptedLLM("better query")

    loop = CorrectiveLoop(evaluator=evaluator, llm_generator=llm, max_attempts=1)
    out, trace = loop.run("q", initial, retry_retrieve=lambda q: retry, top_k=2)

    assert trace.final_strategy == "rewritten"
    assert [c.chunk_id for c, _ in out] == ["x", "y"]


def test_retry_failure_falls_back_to_initial():
    """If the retry retrieval raises, return the initial unchanged."""
    initial = [(_FakeChunk("a"), 0.3)]
    evaluator = _ScriptedEvaluator([_FakeEval(0.3, _Level.LOW, ["weak"])])
    llm = _ScriptedLLM("rewritten q")

    def _boom(_q):
        raise RuntimeError("retriever offline")

    loop = CorrectiveLoop(evaluator=evaluator, llm_generator=llm)
    out, trace = loop.run("q", initial, retry_retrieve=_boom)

    assert out == initial
    assert trace.triggered is True
    assert trace.rewritten_query == "rewritten q"
    # Only the initial attempt was logged before the retry exception.
    assert len(trace.attempts) == 1


def test_rewrite_identical_to_query_skips_retry():
    """If the LLM echoes the input, don't retry — would just waste a round-trip."""
    initial = [(_FakeChunk("a"), 0.4)]
    evaluator = _ScriptedEvaluator([_FakeEval(0.3, _Level.LOW, ["weak"])])
    llm = _ScriptedLLM("the same query")

    loop = CorrectiveLoop(evaluator=evaluator, llm_generator=llm)
    out, trace = loop.run("the same query", initial, retry_retrieve=lambda q: [])

    assert out == initial
    assert trace.triggered is True
    assert trace.rewritten_query is None  # no usable rewrite


def test_trigger_level_medium_catches_medium_confidence():
    """Higher trigger level → retries on merely-medium results, not just low."""
    initial = [(_FakeChunk("a"), 0.5)]
    retry = [(_FakeChunk("b"), 0.85)]
    evaluator = _ScriptedEvaluator(
        [
            _FakeEval(0.55, _Level.MEDIUM, ["could be better"]),
            _FakeEval(0.85, _Level.HIGH, []),
        ]
    )
    llm = _ScriptedLLM("sharper q")

    loop = CorrectiveLoop(
        evaluator=evaluator, llm_generator=llm, trigger_level="medium"
    )
    out, trace = loop.run("q", initial, retry_retrieve=lambda q: retry)

    assert trace.triggered is True
    assert trace.final_strategy == "merged"
    assert {c.chunk_id for c, _ in out} == {"a", "b"}


def test_top_k_caps_merged_results():
    initial = [(_FakeChunk(f"a{i}"), 0.4 - i * 0.01) for i in range(5)]
    retry = [(_FakeChunk(f"b{i}"), 0.6 - i * 0.01) for i in range(5)]
    evaluator = _ScriptedEvaluator(
        [
            _FakeEval(0.3, _Level.LOW, ["weak"]),
            _FakeEval(0.7, _Level.HIGH, []),
        ]
    )
    llm = _ScriptedLLM("better q")

    loop = CorrectiveLoop(evaluator=evaluator, llm_generator=llm)
    out, _ = loop.run("q", initial, retry_retrieve=lambda q: retry, top_k=3)

    assert len(out) == 3
    # Top-3 should all be from `retry` since its scores dominate.
    assert all(c.chunk_id.startswith("b") for c, _ in out)


def test_trace_to_dict_serializable():
    """Trace must be JSON-friendly so query_log can persist it."""
    import json

    initial = [(_FakeChunk("a"), 0.3)]
    evaluator = _ScriptedEvaluator(
        [
            _FakeEval(0.3, _Level.LOW, ["low score"]),
            _FakeEval(0.7, _Level.HIGH, []),
        ]
    )
    llm = _ScriptedLLM("rewritten")

    loop = CorrectiveLoop(evaluator=evaluator, llm_generator=llm)
    _, trace = loop.run(
        "q", initial, retry_retrieve=lambda q: [(_FakeChunk("b"), 0.7)]
    )

    payload = json.dumps(trace.to_dict())  # must not raise
    decoded = json.loads(payload)
    assert decoded["triggered"] is True
    assert decoded["final_strategy"] == "merged"
    assert len(decoded["attempts"]) == 2
