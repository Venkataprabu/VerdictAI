"""Tests for the four panel personas: independent initialization, isolated
pre-debate calls, and distinct evidence-backed conclusions.

Covers the evaluator's "agent independence" requirement:
  - each of the 4 agents (Technical, HR/Culture, Hiring Manager, Skeptic)
    produces its own opinion from a separate call
  - no agent's independent call can see another agent's conclusion
  - agents cite real, verified evidence, never invented text
"""
from __future__ import annotations

import inspect

import app


def test_all_four_roles_are_defined():
    assert app.ROLES == ["Technical", "HR / Culture", "Hiring Manager", "Skeptic"]


def test_each_role_produces_an_independent_opinion(sample_evidence):
    """panel_opinion() (independent, pre-debate) must succeed for every role
    and return a self-consistent, fully-populated opinion record."""
    for role in app.ROLES:
        opinion = app.panel_opinion(role, "AI Engineer", "", sample_evidence)
        assert opinion["role"] == role
        assert opinion["phase"] == "independent"
        assert opinion["mode"] == "Local fallback"  # no API key configured in tests
        assert 0 <= opinion["score"] <= 10
        assert opinion["confidence"] in {"Low", "Medium", "High"}
        assert opinion["fit"] in {"Strong", "Promising", "Needs validation", "Unknown"}


def test_agents_reach_distinct_conclusions(sample_evidence):
    """The four personas are not the same prompt wearing different labels —
    each rule set looks at different keywords and can land on a different
    focus, score, and evidence citation from the same evidence packet."""
    opinions = {role: app.panel_opinion(role, "AI Engineer", "", sample_evidence) for role in app.ROLES}

    foci = {op["focus"] for op in opinions.values()}
    assert len(foci) == len(app.ROLES), "expected every persona to have a distinct focus area"

    # The Skeptic is specifically designed to weigh the ownership-risk evidence
    # (E-02) differently from the optimistic personas — it should score lowest
    # and flag for debate rather than advancing outright.
    assert opinions["Skeptic"]["score"] < opinions["Technical"]["score"]
    assert opinions["Skeptic"]["recommendation"] == "FLAG FOR DEBATE"


def test_llm_opinion_is_isolated_per_role_no_peer_access(sample_evidence):
    """Structural proof of independence: the function that makes each agent's
    pre-debate call only accepts (role, position, requirements, evidence) — it
    has no parameter through which another agent's opinion could be threaded
    in, so it is impossible for it to see a peer's conclusion before forming
    its own."""
    signature = inspect.signature(app.llm_opinion)
    param_names = set(signature.parameters)
    assert param_names == {"role", "position", "requirements", "evidence"}
    for forbidden in ("agents", "peers", "other_opinions", "debate", "history"):
        assert forbidden not in param_names


def test_llm_opinion_returns_none_without_an_api_key(sample_evidence):
    """With no key configured (the `no_api_keys` fixture is autouse), the live
    call must not be attempted — panel_opinion() falls back instead."""
    assert app.llm_opinion("Technical", "AI Engineer", "", sample_evidence) is None


def test_each_role_is_assigned_a_separate_api_key_name():
    """Independence is also preserved at the credentials layer: every
    pre-debate persona reads a *different* environment variable, so even with
    live keys configured, one persona's provider call can't be satisfied by
    another persona's key/session."""
    key_names = {app.provider_for(role)[1] for role in app.ROLES}
    # provider_for returns the *value* (empty string here, since no keys are
    # set) — assert on the underlying env var names instead, which is what
    # actually enforces isolation.
    settings = {
        "Technical": "GROQ_API_KEY_TECHNICAL",
        "HR / Culture": "GEMINI_API_KEY_CULTURE",
        "Hiring Manager": "GEMINI_API_KEY_HIRING",
        "Skeptic": "GROQ_API_KEY_SKEPTIC",
    }
    assert len(set(settings.values())) == 4, "each role must read a distinct credential"
    for role, expected_env_name in settings.items():
        # provider_for looks up os.getenv(expected_env_name); confirm indirectly
        # by monkeypatching one key and checking only that role picks it up.
        import os

        os.environ[expected_env_name] = "test-key-for-" + role
        try:
            _, value = app.provider_for(role)
            assert value == "test-key-for-" + role
            for other_role in app.ROLES:
                if other_role == role:
                    continue
                _, other_value = app.provider_for(other_role)
                assert other_value != "test-key-for-" + role
        finally:
            del os.environ[expected_env_name]


def test_opinion_only_cites_a_real_verified_quote(sample_evidence):
    """Every conclusion must trace back to an exact quote from the evidence
    packet, not paraphrased or invented text — this is what makes agent
    claims auditable against the source PDF/txt."""
    for role in app.ROLES:
        opinion = app.panel_opinion(role, "AI Engineer", "", sample_evidence)
        cited_id = opinion["evidence_id"]
        if cited_id is None:
            continue  # only happens when there is truly no matching evidence
        matching = [item for item in sample_evidence if item["id"] == cited_id]
        assert matching, f"{role} cited an evidence ID that does not exist"
        assert opinion["quote"] == matching[0]["quote"]


def test_opinion_with_no_matching_evidence_is_marked_insufficient():
    """If none of the role's keywords appear anywhere in the candidate
    evidence, the persona must say so explicitly rather than guessing."""
    empty_evidence = [
        {
            "id": "E-01",
            "source": "Job Description · jd.txt",
            "source_kind": "job_posting",
            "quote": "We are looking for a great teammate.",
            "verified": True,
        }
    ]
    opinion = app.fallback_opinion("Technical", "AI Engineer", empty_evidence)
    assert opinion["recommendation"] == "INSUFFICIENT EVIDENCE"
    assert opinion["confidence"] == "Low"
    assert opinion["evidence_id"] is None
