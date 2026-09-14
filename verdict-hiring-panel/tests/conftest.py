"""Shared pytest fixtures for the Verdict test suite.

Every test in this suite runs in "local fallback" mode: no live API keys are
configured, so app.py's own fallback logic (fallback_opinion, fallback_debate,
etc.) is exercised deterministically. This is intentional — it means the suite
never makes real network calls, never needs secrets to run, and is fast and
reproducible in CI. The independence/citation/state-change guarantees we test
here hold in both fallback and live modes because they're enforced in shared
code (validate_evidence_result, provider_for, run_panel), not inside the
per-provider branches.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `import app` work when pytest is invoked from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402  (import after sys.path fixup, by design)


@pytest.fixture(autouse=True)
def no_api_keys(monkeypatch):
    """Guarantee every test exercises the local fallback path deterministically,
    regardless of what a developer happens to have in their real .env file."""
    for name in (
        "GROQ_API_KEY_TECHNICAL",
        "GEMINI_API_KEY_CULTURE",
        "GEMINI_API_KEY_HIRING",
        "GROQ_API_KEY_SKEPTIC",
        "GEMINI_API_KEY_ADJUDICATOR",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def sample_evidence():
    """A small, realistic evidence packet: one clean technical/impact signal,
    and one explicit ownership-risk phrase ("only observed... did not own it")
    so the Skeptic persona and the debate's unresolved-risk path both trigger,
    the same way they would against a real transcript."""
    return [
        {
            "id": "E-01",
            "source": "Candidate Resume · resume.txt",
            "source_kind": "candidate_resume",
            "quote": "Built and shipped a Python API used in production.",
            "verified": True,
        },
        {
            "id": "E-02",
            "source": "Candidate Transcript · transcript.txt",
            "source_kind": "candidate_transcript",
            "quote": "I partnered with the team but only observed the deployment, I did not own it.",
            "verified": True,
        },
        {
            "id": "E-03",
            "source": "Candidate Resume · resume.txt",
            "source_kind": "candidate_resume",
            "quote": "Collaborated with stakeholders to reduce conflict during a migration.",
            "verified": True,
        },
    ]


@pytest.fixture
def sample_sources():
    """Raw source text, as it would look after ingest() extracted it from an
    uploaded resume/transcript, before build_evidence() turns it into cited
    evidence records."""
    return {
        "Candidate Resume · resume.txt": {
            "text": (
                "Built and shipped a Python API used in production. "
                "Collaborated with stakeholders to reduce conflict during a migration."
            ),
            "kind": "candidate_resume",
        },
        "Candidate Transcript · transcript.txt": {
            "text": (
                "I partnered with the team but only observed the deployment, "
                "I did not own it fully."
            ),
            "kind": "candidate_transcript",
        },
    }


@pytest.fixture
def client():
    app.app.config.update(TESTING=True)
    return app.app.test_client()
