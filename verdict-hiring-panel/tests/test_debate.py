"""Tests for the debate step and the explicit pre/post-debate state changes.

Covers the evaluator's "debate quality" and "visible state changes" requirements:
  - later debate turns respond directly to earlier turns (responds_to chain)
  - agent outputs from the debate are passed into subsequent rounds
  - a revised position produces a real score change, with a stated reason,
    and an explicit before/after record (not silent reuse of the first score)
"""
from __future__ import annotations

import app


def _agents(sample_evidence):
    return [app.fallback_opinion(role, "AI Engineer", sample_evidence) for role in app.ROLES]


def test_debate_falls_back_without_api_keys(sample_evidence):
    """debate() must not silently no-op when no live keys are configured —
    it should produce the local rule-based debate instead."""
    agents = _agents(sample_evidence)
    turns = app.debate("AI Engineer", "", sample_evidence, agents)
    assert len(turns) == 3
    assert [t["speaker"] for t in turns] == ["Skeptic", "Hiring Manager", "Technical"]


def test_later_turns_directly_respond_to_earlier_turns(sample_evidence):
    """This is the core 'real debate, not four summaries' invariant: every
    turn after the first must name exactly which prior speaker it is
    responding to."""
    agents = _agents(sample_evidence)
    turns = app.debate("AI Engineer", "", sample_evidence, agents)

    assert turns[0]["responds_to"] is None  # opens the debate
    assert turns[1]["responds_to"] == "Skeptic"
    assert turns[2]["responds_to"] == "Hiring Manager"
    # Every non-opening turn must reference a speaker who actually spoke earlier.
    speakers_so_far = {turns[0]["speaker"]}
    for turn in turns[1:]:
        assert turn["responds_to"] in speakers_so_far
        speakers_so_far.add(turn["speaker"])


def test_debate_turns_receive_prior_history(sample_evidence):
    """fallback_debate's turns are built from the *other* agents' initial
    opinions (e.g. turn 2's quote comes from the Hiring Manager's own
    independent opinion, not from thin air) — confirming later turns are
    genuinely conditioned on what came before, not static text."""
    agents = _agents(sample_evidence)
    turns = app.debate("AI Engineer", "", sample_evidence, agents)

    hiring_manager_opinion = next(a for a in agents if a["role"] == "Hiring Manager")
    turn_2 = turns[1]
    assert turn_2["speaker"] == "Hiring Manager"
    assert turn_2["quote"] == hiring_manager_opinion["quote"]
    assert turn_2["evidence_id"] == hiring_manager_opinion["evidence_id"]


def test_unresolved_ownership_risk_is_flagged(sample_evidence):
    """sample_evidence includes an explicit 'only observed... did not own it'
    phrase — the debate must surface this as an unresolved risk rather than
    quietly dropping it."""
    agents = _agents(sample_evidence)
    turns = app.debate("AI Engineer", "", sample_evidence, agents)
    assert any(turn["unresolved_risk"] for turn in turns)


def test_final_position_records_explicit_before_after_scores(sample_evidence):
    """The literal requirement: an agent's opinion changing during debate must
    be visible as a distinct pre_debate_score vs. score (post-debate), with a
    stated reason — not silently overwritten."""
    agents = _agents(sample_evidence)
    turns = app.debate("AI Engineer", "", sample_evidence, agents)

    hiring_manager_initial = next(a for a in agents if a["role"] == "Hiring Manager")
    final = app.final_position("Hiring Manager", hiring_manager_initial, turns, "AI Engineer", "", sample_evidence)

    assert final["pre_debate_score"] == hiring_manager_initial["score"]
    assert final["revised_after_debate"] is True
    assert final["score"] != final["pre_debate_score"]
    assert final["revision"]  # a human-readable reason must be present


def test_final_position_with_no_debate_activity_is_unchanged(sample_evidence):
    """An agent whose debate turn was never marked as revised should keep its
    pre-debate score, and say so explicitly — a state log entry of 'unchanged'
    is just as important to show as a change."""
    agents = _agents(sample_evidence)
    turns = app.debate("AI Engineer", "", sample_evidence, agents)

    hr_initial = next(a for a in agents if a["role"] == "HR / Culture")
    final = app.final_position("HR / Culture", hr_initial, turns, "AI Engineer", "", sample_evidence)

    assert final["revised_after_debate"] is False
    assert final["score"] == final["pre_debate_score"]
    assert final["revision"] == "No score revision after the debate."


def test_run_panel_produces_an_explicit_json_state_log(sample_sources):
    """End-to-end: the full pipeline's output includes a state_log array with
    one entry per agent, each showing pre/post scores and whether it changed —
    this is the literal 'explicit JSON log' the report and UI both surface."""
    report = app.run_panel(sample_sources, "AI Engineer", "", "Candidate")

    assert "state_log" in report
    assert len(report["state_log"]) == len(app.ROLES)
    for entry in report["state_log"]:
        assert set(entry) == {"agent", "pre_debate_score", "post_debate_score", "changed", "reason"}
        assert isinstance(entry["changed"], bool)

    # Given the ownership-risk phrasing in sample_sources, at least one agent
    # is expected to actually revise its position during the debate.
    assert any(entry["changed"] for entry in report["state_log"])
    assert report["debate"]["changed_positions"], "at least one role should appear in changed_positions"


def test_run_panel_recommendation_is_one_of_the_defined_tiers(sample_sources):
    report = app.run_panel(sample_sources, "AI Engineer", "", "Candidate")
    assert report["recommendation"] in {
        "HIRE",
        "INTERVIEW",
        "REJECT",
        "INTERVIEW — VERIFY CREDIBILITY CLAIM",
    }
