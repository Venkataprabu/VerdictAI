from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

BASE = Path(__file__).resolve().parent
ALLOWED = {"txt", "pdf", "docx"}
ROLES = ["Technical", "HR / Culture", "Hiring Manager", "Skeptic"]
MAX_POSITION_CHARS = 300
MAX_REQUIREMENTS_CHARS = 4000
API_FAILURES: dict[str, str] = {}
API_FAILURES_LOCK = threading.Lock()
ENV_FILES_LOADED: list[str] = []


def is_candidate_source(kind: str) -> bool:
    """Candidate evidence has separate resume/transcript labels but one trust level."""
    return kind.startswith("candidate")


def load_local_env() -> None:
    """Small dependency-free .env loader for local hackathon use."""
    # The active file may be copied out of attached_assets/, while the .env
    # remains in the folder from which Flask is launched. Support both paths.
    candidates = [BASE / ".env", Path.cwd() / ".env", BASE.parent / ".env"]
    seen: set[Path] = set()
    for env_file in candidates:
        env_file = env_file.resolve()
        if env_file in seen or not env_file.exists():
            continue
        seen.add(env_file)
        try:
            lines = env_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            # Real environment variables take precedence, except an empty
            # injected value should not prevent a useful .env value.
            if name and value and not os.getenv(name, "").strip():
                os.environ[name] = value
        ENV_FILES_LOADED.append(str(env_file))


load_local_env()
print(
    "[config] .env files loaded: "
    + (", ".join(ENV_FILES_LOADED) if ENV_FILES_LOADED else "none")
)
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

app = Flask(__name__, static_folder="public", static_url_path="")
# Vercel serves anything under public/** straight from its CDN at the same root path
# (public/demo-debate.mp3 -> /demo-debate.mp3), so pointing Flask's own static handler
# at "public" with an empty url prefix keeps local `python app.py` and a Vercel
# deployment resolving assets at the exact same URLs.
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # total request size, all files combined


@app.after_request
def set_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify(error="Uploaded files are too large (32MB combined limit)."), 413


@app.errorhandler(500)
def internal_error(_error):
    return jsonify(error="Something went wrong while analyzing the candidates. Please try again."), 500


def extract_text(file_path: Path) -> str:
    extension = file_path.suffix.lower().lstrip(".")
    if extension == "txt":
        return file_path.read_text(encoding="utf-8", errors="replace")
    if extension == "pdf":
        import pypdf
        return "\n".join(page.extract_text() or "" for page in pypdf.PdfReader(str(file_path)).pages)
    if extension == "docx":
        from docx import Document
        return "\n".join(p.text for p in Document(str(file_path)).paragraphs)
    raise ValueError("Unsupported document type")


def sentences(text: str) -> list[str]:
    chunks: list[str] = []
    for line in text.splitlines():
        normalized = re.sub(r"\s+", " ", line).strip(" •-\t")
        if not normalized:
            continue
        chunks.extend(re.split(r"(?<=[.!?])\s+", normalized))
    # PDF extractors often return short, punctuation-free resume lines.
    return [chunk.strip(" •-\t") for chunk in chunks if len(chunk.strip()) >= 10]


def build_evidence(sources: dict[str, dict[str, str]]) -> list[dict]:
    """Create the only evidence packet any persona is allowed to see."""
    keywords = re.compile(
        r"\b(built|shipped|led|designed|reduced|increased|improved|"
        r"python|react|api|team|collaborat|conflict|partnered|aws|sql|"
        r"database|performance|ownership|production|rag|retrieval|"
        r"communication|stakeholder|incident|migration)\b",
        re.I,
    )
    records: list[dict] = []
    for source, details in sources.items():
        text = details["text"]
        selected = [sentence for sentence in sentences(text) if keywords.search(sentence)]
        if not selected:
            selected = sentences(text)[:4]
        for sentence in selected:
            records.append(
                {
                    "source": source,
                    "source_kind": details.get("kind", "candidate"),
                    "quote": sentence[:420],
                    "verified": sentence in text,
                }
            )
    # Keep enough context for a real debate without allowing a prompt to grow forever.
    for number, record in enumerate(records[:15], 1):
        record["id"] = f"E-{number:02d}"
    return records[:15]


def evidence_packet(evidence: list[dict]) -> str:
    return "\n".join(
        f"{item['id']} | {item['source_kind']} | {item['source']} | {item['quote']}"
        for item in evidence
    )


def provider_env_names(role: str, phase: str = "initial") -> tuple[str, ...]:
    """Role-specific keys are preferred, with generic-key compatibility."""
    if phase == "adjudication":
        return ("GEMINI_API_KEY_ADJUDICATOR", "GEMINI_API_KEY")
    settings = {
        "Technical": ("OPENROUTER_API_KEY_TECHNICAL", "OPENROUTER_API_KEY"),
        "HR / Culture": ("GEMINI_API_KEY_CULTURE", "GEMINI_API_KEY"),
        "Hiring Manager": ("GEMINI_API_KEY_HIRING", "GEMINI_API_KEY"),
        "Skeptic": ("OPENROUTER_API_KEY_SKEPTIC", "OPENROUTER_API_KEY"),
    }
    return settings[role]


def provider_for(role: str, phase: str = "initial") -> tuple[str, str]:
    """Different keys preserve isolation; generic keys keep local setup compatible."""
    provider = "gemini" if phase == "adjudication" or role in {"HR / Culture", "Hiring Manager"} else "openrouter"
    api_key = next((os.getenv(name, "").strip() for name in provider_env_names(role, phase) if os.getenv(name, "").strip()), "")
    return provider, api_key


def public_api_config() -> dict:
    """Return configuration status without ever returning key values."""
    roles = {}
    for role in ROLES:
        names = provider_env_names(role)
        selected_name = next((name for name in names if os.getenv(name, "").strip()), None)
        roles[role] = {
            "provider": "gemini" if role in {"HR / Culture", "Hiring Manager"} else "groq",
            "configured": selected_name is not None,
            "source": selected_name,
        }
    return {
        "env_files_loaded": ENV_FILES_LOADED,
        "roles": roles,
    }


def record_api_failure(role: str, reason: str) -> None:
    with API_FAILURES_LOCK:
        API_FAILURES[role] = reason[:240]


def parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def call_llm(provider: str, api_key: str, system: str, user: str) -> dict:
    if provider == "openrouter":
        body = json.dumps(
            {
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.2,
                "max_tokens": 300,
            }
        ).encode()
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://verdict-hiring-panel.vercel.app",
            "X-Title": "Verdict Hiring Panel",
        }
    else:
        body = json.dumps(
            {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"parts": [{"text": user}]}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2, "maxOutputTokens": 1024},
            }
        ).encode()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    req = urllib.request.Request(url, data=body, method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=45) as response:
        data = json.loads(response.read())
    raw = (
        data["choices"][0]["message"]["content"]
        if provider == "openrouter"
        else data["candidates"][0]["content"]["parts"][0]["text"]
    )
    return parse_json(raw)


def validate_evidence_result(result: dict, evidence: list[dict]) -> dict:
    valid_ids = {item["id"] for item in evidence}
    evidence_id = result.get("evidence_id")
    if evidence_id not in valid_ids:
        raise ValueError("The model returned an unverified evidence ID")
    quote = next(item["quote"] for item in evidence if item["id"] == evidence_id)
    result["quote"] = quote
    result["score"] = max(0.0, min(10.0, float(result.get("score", 0))))
    result["confidence"] = result.get("confidence", "Low")
    result["fit"] = result.get("fit", "Unknown")
    result["recommendation"] = result.get("recommendation", "INSUFFICIENT EVIDENCE")
    return result


def llm_opinion(role: str, position: str, requirements: str, evidence: list[dict]) -> dict | None:
    provider, api_key = provider_for(role)
    if not api_key:
        record_api_failure(role, f"Missing API key: set {', '.join(provider_env_names(role))}.")
        return None
    system = f"""You are the {role} on a hiring panel evaluating fit for {position}.
Context: {requirements or 'Not supplied; state what cannot be assessed.'}
Pre-debate independent call — you cannot see other opinions. Use only verified candidate records.
Return strict JSON: score (0-10), recommendation (ADVANCE, HOLD, or INSUFFICIENT EVIDENCE),
confidence (Low, Medium, or High), fit (Strong, Promising, Needs validation, or Unknown),
reasoning (1-2 sentences: top strength and main concern), evidence_id (one packet ID), and quote.
Every conclusion must cite one exact packet quote."""
    try:
        result = validate_evidence_result(call_llm(provider, api_key, system, evidence_packet(evidence)), evidence)
        result.update({"role": role, "mode": f"Live {provider.title()} API", "phase": "independent"})
        return result
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        if isinstance(exc, urllib.error.HTTPError):
            reason = f"HTTP {exc.code} from {provider.title()} API"
        else:
            reason = f"{type(exc).__name__}: {exc}"
        record_api_failure(role, reason)
        return None


def cited(evidence: list[dict], words: Iterable[str]) -> dict | None:
    for item in evidence:
        if any(word.lower() in item["quote"].lower() for word in words):
            return item
    return evidence[0] if evidence else None


def fallback_opinion(role: str, position: str, evidence: list[dict]) -> dict:
    rules = {
        "Technical": (["python", "react", "api", "sql", "database", "aws", "performance", "designed", "built", "rag"], "technical depth"),
        "HR / Culture": (["team", "collaborat", "conflict", "partnered", "communication", "stakeholder"], "collaboration"),
        "Hiring Manager": (["shipped", "led", "reduced", "increased", "improved", "impact", "production"], "role impact"),
        "Skeptic": (["led", "owner", "partnered", "team", "observed", "only", "actually"], "ownership scope"),
    }
    terms, focus = rules[role]
    candidate_evidence = [item for item in evidence if is_candidate_source(item["source_kind"])]
    item = cited(candidate_evidence, terms)
    if not item:
        return {
            "role": role, "focus": focus, "recommendation": "INSUFFICIENT EVIDENCE",
            "confidence": "Low", "score": 0, "quote": "No verified source fact supports a conclusion.",
            "evidence_id": None, "fit": "Unknown",
            "reasoning": "The submitted record does not contain enough verified evidence for this assessment.",
            "mode": "Local fallback", "phase": "independent",
        }
    confidence = "High" if len(candidate_evidence) >= 5 else "Medium" if candidate_evidence else "Low"
    ownership_risk = any(
        re.search(r"\b(observed|only|didn't|did not|partnered|team)\b", e["quote"], re.I)
        for e in evidence
    )
    score = {"Technical": 7.4, "HR / Culture": 7.0, "Hiring Manager": 7.6, "Skeptic": 4.0 if ownership_risk else 6.2}[role]
    return {
        "role": role,
        "focus": focus,
        "recommendation": "FLAG FOR DEBATE" if role == "Skeptic" else "ADVANCE",
        "confidence": confidence,
        "score": score,
        "quote": item["quote"],
        "evidence_id": item["id"],
        "fit": "Needs validation" if role == "Skeptic" else "Promising",
        "reasoning": (
            f"This verified {focus} signal is relevant to the {position} role."
            if role != "Skeptic"
            else f"For {position}, this evidence does not independently prove the full scope of ownership."
        ),
        "mode": "Local fallback",
        "phase": "independent",
    }


def panel_opinion(role: str, position: str, requirements: str, evidence: list[dict]) -> dict:
    result = llm_opinion(role, position, requirements, evidence)
    if result is not None:
        return result
    fallback = fallback_opinion(role, position, evidence)
    fallback["api_error"] = API_FAILURES.get(role, "Live API call failed; local fallback used.")
    return fallback


def fallback_debate(position: str, evidence: list[dict], agents: list[dict]) -> list[dict]:
    skeptic = next(agent for agent in agents if agent["role"] == "Skeptic")
    hiring = next(agent for agent in agents if agent["role"] == "Hiring Manager")
    technical = next(agent for agent in agents if agent["role"] == "Technical")
    unresolved = any(
        re.search(r"\b(observed|only|didn't|did not|partnered|team|collaborat)\b", item["quote"], re.I)
        for item in evidence if is_candidate_source(item["source_kind"])
    )
    return [
        {
            "turn": 1, "speaker": "Skeptic", "responds_to": None,
            "stance": "challenge", "evidence_id": skeptic.get("evidence_id"),
            "quote": skeptic["quote"],
            "message": f"The record supports a signal, but it does not independently prove the full ownership scope for {position}.",
            "revised": False, "unresolved_risk": unresolved, "resolved": False,
        },
        {
            "turn": 2, "speaker": "Hiring Manager", "responds_to": "Skeptic",
            "stance": "counter", "evidence_id": hiring.get("evidence_id"),
            "quote": hiring["quote"],
            "message": "I agree the ownership claim needs validation, but the verified role impact is still relevant and should not be diluted into a rejection.",
            "revised": True, "unresolved_risk": unresolved, "resolved": not unresolved,
        },
        {
            "turn": 3, "speaker": "Technical", "responds_to": "Hiring Manager",
            "stance": "refine", "evidence_id": technical.get("evidence_id"),
            "quote": technical["quote"],
            "message": "The technical signal is promising; I am revising from an unqualified advance to advance only with a direct production-ownership check.",
            "revised": True, "unresolved_risk": unresolved, "resolved": False,
        },
        {
            "turn": 4, "speaker": "HR / Culture", "responds_to": "Technical",
            "stance": "moderate", "evidence_id": technical.get("evidence_id"),
            "quote": technical["quote"],
            "message": "I agree the evidence is promising, but the final report should distinguish collaboration from sole ownership and carry that question into the interview.",
            "revised": True, "unresolved_risk": unresolved, "resolved": False,
        },
    ]


def live_debate_turn(role: str, position: str, requirements: str, evidence: list[dict], agents: list[dict], history: list[dict]) -> dict | None:
    provider, api_key = provider_for(role)
    if not api_key:
        record_api_failure(f"Debate · {role}", f"Missing API key: set {', '.join(provider_env_names(role))}.")
        return None
    system = f"""You are the {role} in a live hiring-panel debate about {position}.
Respond directly to the latest speaker, cite exact evidence, revise your view if warranted.
Only candidate records prove capability; job-posting records define the role.
Return strict JSON: speaker, responds_to, stance, message (2 sentences max), evidence_id, revised (boolean),
unresolved_risk (boolean), resolved (boolean), final_position (Strong, Promising, Needs validation, or Unknown).
Only cite an ID from the evidence packet."""
    user = json.dumps(
        {
            "requirements": requirements,
            "evidence": evidence_packet(evidence),
            "independent_opinions": agents,
            "debate_so_far": history,
            "instruction": f"Make a substantive {role} turn that directly answers the previous turn.",
        }
    )
    try:
        result = validate_evidence_result(call_llm(provider, api_key, system, user), evidence)
        result.update({"turn": len(history) + 1, "speaker": role, "quote": result["quote"], "mode": f"Live {provider.title()} API"})
        return result
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        record_api_failure(f"Debate · {role}", f"{type(exc).__name__}: {exc}")
        return None


def debate(position: str, requirements: str, evidence: list[dict], agents: list[dict]) -> list[dict]:
    history: list[dict] = []
    fallback_turns = {turn["speaker"]: turn for turn in fallback_debate(position, evidence, agents)}
    for role in ROLES:
        turn = live_debate_turn(role, position, requirements, evidence, agents, history)
        if turn is None:
            turn = fallback_turns[role]
        history.append(turn)
    # At least one direct response is a hard invariant, even if a model omitted the field.
    for index, turn in enumerate(history):
        if index and not turn.get("responds_to"):
            turn["responds_to"] = history[index - 1]["speaker"]
    return history


def llm_final_position(
    role: str,
    position: str,
    requirements: str,
    evidence: list[dict],
    initial: dict,
    debate_turns: list[dict],
) -> dict | None:
    provider, api_key = provider_for(role)
    if not api_key:
        record_api_failure(f"Post-debate · {role}", f"Missing API key: set {', '.join(provider_env_names(role))}.")
        return None
    system = f"""You are the {role} issuing your final post-debate position for {position}.
Revise your score only if the debate surfaced new evidence. Use only verified candidate records.
Return strict JSON: score (0-10), recommendation (ADVANCE, HOLD, or INSUFFICIENT EVIDENCE),
confidence (Low, Medium, or High), fit (Strong, Promising, Needs validation, or Unknown),
reasoning (1-2 sentences), evidence_id, quote, revised (boolean), revision_reason."""
    user = json.dumps(
        {
            "requirements": requirements,
            "initial_opinion": initial,
            "full_debate": debate_turns,
            "evidence": evidence_packet(evidence),
        }
    )
    try:
        result = validate_evidence_result(call_llm(provider, api_key, system, user), evidence)
        result.update({"role": role, "mode": f"Live {provider.title()} API", "phase": "post-debate"})
        return result
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        record_api_failure(f"Post-debate · {role}", f"{type(exc).__name__}: {exc}")
        return None


def final_position(role: str, initial: dict, debate_turns: list[dict], position: str, requirements: str, evidence: list[dict]) -> dict:
    """Use an explicit post-debate position for adjudication; never silently reuse the first score."""
    live_result = llm_final_position(role, position, requirements, evidence, initial, debate_turns)
    if live_result is not None:
        live_result["pre_debate_score"] = round(float(initial.get("score", 0)), 1)
        live_result["revised_after_debate"] = bool(
            live_result.get("revised") or live_result.get("score") != initial.get("score")
        )
        live_result["revision"] = live_result.get("revision_reason") or (
            "Position revised after the debate." if live_result["revised_after_debate"]
            else "No score revision after the debate."
        )
        return live_result

    relevant = [turn for turn in debate_turns if turn.get("speaker") == role]
    changed = any(turn.get("revised") for turn in relevant)
    revised_score = initial.get("score", 0)
    if changed:
        if role == "Skeptic":
            revised_score = min(revised_score, 4.5)
        else:
            revised_score = max(0, revised_score - 0.6)
    result = dict(initial)
    result["pre_debate_score"] = round(float(initial.get("score", 0)), 1)
    result["score"] = round(float(revised_score), 1)
    result["phase"] = "post-debate"
    result["revised_after_debate"] = changed
    result["revision"] = (
        "Position revised after direct challenge; ownership must be validated."
        if changed else "No score revision after the debate."
    )
    return result


def derive_weights(role_context: str) -> tuple[dict[str, float], str]:
    context = role_context.lower()
    weights = {"Technical": 35.0, "Hiring Manager": 30.0, "HR / Culture": 20.0, "Skeptic": 15.0}
    signals: list[str] = []
    if re.search(r"\b(python|api|sql|cloud|rag|machine learning|architecture|backend|technical)\b", context):
        weights["Technical"] += 5
        signals.append("technical requirements")
    if re.search(r"\b(lead|ownership|impact|delivery|production|manager|senior)\b", context):
        weights["Hiring Manager"] += 5
        signals.append("ownership/impact requirements")
    if re.search(r"\b(team|collaborat|communication|stakeholder|culture)\b", context):
        weights["HR / Culture"] += 5
        signals.append("collaboration requirements")
    total = sum(weights.values())
    weights = {key: round(value * 100 / total, 1) for key, value in weights.items()}
    basis = "Default role weights"
    if signals:
        basis += " adjusted from attached role context for " + ", ".join(signals)
    return weights, basis


def confidence_score(final_agents: list[dict], evidence: list[dict], debate_turns: list[dict], override: bool) -> int:
    scores = [float(agent.get("score", 0)) for agent in final_agents]
    mean = sum(scores) / len(scores) if scores else 0
    agreement = max(0.0, 1.0 - (sum(abs(score - mean) for score in scores) / len(scores)) / 5)
    cited_count = sum(bool(agent.get("evidence_id")) for agent in final_agents)
    evidence_coverage = cited_count / len(final_agents) if final_agents else 0
    unresolved = any(turn.get("unresolved_risk") and not turn.get("resolved") for turn in debate_turns)
    debate_resolution = 0.45 if unresolved else 0.95
    value = 100 * (agreement * 0.4 + evidence_coverage * 0.25 + debate_resolution * 0.2 + (0.0 if override else 0.15))
    if override:
        value -= 8
    return max(0, min(99, round(value)))


def fallback_narrative(adjudication: dict, debate_turns: list[dict]) -> str:
    if adjudication["override_applied"]:
        return (
            f"The post-debate weighted score is {adjudication['weighted_score']}/10, but the Skeptic surfaced "
            "a specific evidence-backed ownership contradiction that remained unresolved. The panel therefore "
            "caps the recommendation at an interview focused on verifying the claim."
        )
    return (
        f"The panel uses post-debate positions and redistributed confidence-weighted dimensions, producing "
        f"{adjudication['weighted_score']}/10. The evidence supports the displayed tier without a Skeptic veto."
    )


def llm_adjudication_narrative(position: str, requirements: str, evidence: list[dict], debate_turns: list[dict], adjudication: dict) -> str | None:
    provider, api_key = provider_for("Hiring Manager", "adjudication")
    if not api_key:
        return None
    system = """You are the final adjudicator for a hiring decision-support report.
The numeric result and any Skeptic cap are computed by code. Write exactly 2-3 sentences: state the verdict tier,
cite the single strongest evidence signal that drove it, and name the one key risk to verify in an interview.
Be direct and professional. Do not change the computed tier. Return JSON with one key: narrative."""
    user = json.dumps(
        {
            "position": position,
            "requirements": requirements,
            "evidence": evidence_packet(evidence),
            "full_debate": debate_turns,
            "computed_adjudication": adjudication,
        }
    )
    try:
        result = call_llm(provider, api_key, system, user)
        return str(result.get("narrative", "")).strip() or None
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def run_panel(sources: dict[str, dict[str, str]], position: str, requirements: str, candidate_label: str) -> dict:
    with API_FAILURES_LOCK:
        API_FAILURES.clear()
    evidence = build_evidence(sources)
    # These are four separate calls. No opinion is passed to another before debate().
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(panel_opinion, role, position, requirements, evidence) for role in ROLES]
        agents = [future.result() for future in futures]
    debate_turns = debate(position, requirements, evidence, agents)
    with ThreadPoolExecutor(max_workers=4) as pool:
        final_futures = [
            pool.submit(final_position, role, initial, debate_turns, position, requirements, evidence)
            for role, initial in zip(ROLES, agents)
        ]
        final_agents = [future.result() for future in final_futures]

    role_context = " ".join([requirements] + [item["text"] for item in sources.values() if item["kind"] in {"job_posting", "required_skills"}])
    weights, weight_basis = derive_weights(role_context)
    confidence_factors = {"High": 1.0, "Medium": 0.7, "Low": 0.4}
    effective_raw = {
        agent["role"]: weights[agent["role"]] * confidence_factors.get(agent.get("confidence"), 0.4)
        for agent in final_agents
    }
    effective_total = sum(effective_raw.values()) or 1
    redistributed = {role: round(value * 100 / effective_total, 1) for role, value in effective_raw.items()}
    weighted_score = round(
        sum(agent["score"] * redistributed[agent["role"]] / 100 for agent in final_agents), 2
    )
    base_tier = "HIRE" if weighted_score >= 8 else "INTERVIEW" if weighted_score >= 5 else "REJECT"
    unresolved_turns = [
        turn for turn in debate_turns if turn.get("unresolved_risk") and not turn.get("resolved")
    ]
    skeptic = next(agent for agent in final_agents if agent["role"] == "Skeptic")
    veto = bool(unresolved_turns and skeptic.get("evidence_id"))
    override_applied = veto and base_tier == "HIRE"
    recommendation = (
        "INTERVIEW — VERIFY CREDIBILITY CLAIM" if override_applied
        else base_tier
    )
    adjudication = {
        "weighted_score": weighted_score,
        "base_tier": base_tier,
        "recommendation": recommendation,
        "thresholds": {"HIRE": ">= 8.0", "INTERVIEW": "5.0–7.99", "REJECT": "< 5.0"},
        "weights": weights,
        "effective_weights_after_confidence_redistribution": redistributed,
        "weight_basis": weight_basis,
        "override_applied": override_applied,
        "override": (
            "Skeptic veto: specific evidence-backed contradiction remained unresolved; tier capped at INTERVIEW."
            if veto else "No Skeptic veto applied."
        ),
    }
    narrative = llm_adjudication_narrative(position, requirements, evidence, debate_turns, adjudication)
    adjudication["narrative"] = narrative or fallback_narrative(adjudication, debate_turns)
    strengths = [
        f"{agent.get('focus', agent['role']).title()} is supported by {agent.get('evidence_id')}."
        for agent in final_agents[:3] if agent.get("evidence_id")
    ]
    concerns = [
        "Individual ownership must be validated: collaboration evidence does not prove sole authorship."
        if veto else "The panel should still validate the strongest claims directly in the next interview."
    ]
    if not evidence:
        concerns.append("No verified evidence was extractable from the submitted records.")
    # Explicit, machine-checkable record of every agent's pre- vs. post-debate position.
    # This is what proves an opinion actually moved during the debate, not just that a
    # different number appears in the final report.
    state_log = [
        {
            "agent": agent["role"],
            "pre_debate_score": agent.get("pre_debate_score"),
            "post_debate_score": agent.get("score"),
            "changed": bool(agent.get("revised_after_debate")),
            "reason": agent.get("revision"),
        }
        for agent in final_agents
    ]
    return {
        "candidate": candidate_label,
        "position": position,
        "requirements": requirements,
        "recommendation": recommendation,
        "confidence": confidence_score(final_agents, evidence, debate_turns, override_applied),
        "evidence": evidence,
        "agents": final_agents,
        "debate": {
            "topic": f"{position} fit: ownership scope versus demonstrated impact",
            "turns": debate_turns,
            "has_direct_response": any(turn.get("responds_to") for turn in debate_turns),
            "changed_positions": [agent["role"] for agent in final_agents if agent.get("revised_after_debate")],
        },
        "state_log": state_log,
        "adjudication": adjudication,
        "strengths": strengths,
        "concerns": concerns,
        "sources": list(sources),
        "live_agents": sum(agent["mode"].startswith("Live") for agent in agents),
        "api_config": public_api_config(),
        "api_failures": dict(API_FAILURES),
        "audio": {
            "url": "/demo-debate.mp3",
            "transcript": "/demo-debate-transcript.txt",
            "note": "Included demo recording: Skeptic, Hiring Manager, and adjudicator voices.",
        },
    }


def ingest(files, kind: str, sources: dict[str, dict[str, str]]) -> None:
    for file in files:
        if not file or not file.filename:
            continue
        name = secure_filename(file.filename)
        extension = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if not name or extension not in ALLOWED:
            raise ValueError(f"{name or 'File'}: only TXT, PDF, and DOCX are supported.")
        # Every upload is written to a private OS temp file, read, and deleted
        # immediately after text extraction: nothing from a resume or transcript
        # is retained on disk once the request finishes. This also keeps the app
        # compatible with read-only serverless filesystems (e.g. Vercel), which
        # only allow writes under the system temp directory.
        with tempfile.NamedTemporaryFile(suffix=f".{extension}", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            file.save(tmp_path)
            text = extract_text(tmp_path)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"{name}: could not read this file as {extension.upper()}.") from exc
        finally:
            tmp_path.unlink(missing_ok=True)
        if text.strip():
            sources[f"{kind.replace('_', ' ').title()} · {name}"] = {"text": text, "kind": kind}


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/analyze")
def analyze():
    position = request.form.get("target_position", "").strip()[:MAX_POSITION_CHARS]
    requirements = request.form.get("requirements", "").strip()[:MAX_REQUIREMENTS_CHARS]
    if not position:
        return jsonify(error="Enter the target position before analyzing the candidates."), 400

    sources_common: dict[str, dict[str, str]] = {}
    try:
        ingest(request.files.getlist("required_skills"), "required_skills", sources_common)
        candidate_a_sources: dict[str, dict[str, str]] = {}
        ingest(request.files.getlist("candidate_resume"), "candidate_resume", candidate_a_sources)
        ingest(request.files.getlist("candidate_transcript"), "candidate_transcript", candidate_a_sources)
        if not candidate_a_sources:
            raise ValueError("Upload a readable candidate resume or interview transcript.")

        candidate_b_sources: dict[str, dict[str, str]] = {}
        ingest(request.files.getlist("candidate_b_resume"), "candidate_b_resume", candidate_b_sources)
        ingest(request.files.getlist("candidate_b_transcript"), "candidate_b_transcript", candidate_b_sources)
    except ValueError as exc:
        # Expected, user-facing validation failures (bad extension, unreadable file, etc.)
        return jsonify(error=str(exc)), 400
    except OSError:
        # Unexpected server-side storage failure — log details server-side, tell the
        # client nothing that could leak filesystem internals.
        app.logger.exception("Filesystem error while processing an upload")
        return jsonify(error="Could not process the submitted files due to a server-side storage error."), 500

    reports = {
        "Candidate": run_panel(
            {**sources_common, **candidate_a_sources}, position, requirements, "Candidate"
        )
    }
    if candidate_b_sources:
        reports["Candidate B"] = run_panel(
            {**sources_common, **candidate_b_sources}, position, requirements, "Candidate B"
        )
    return jsonify(
        {
            "position": position,
            "reports": reports,
            "candidate_count": len(reports),
            "audio": {"url": "/demo-debate.mp3", "transcript": "/demo-debate-transcript.txt"},
        }
    )


if __name__ == "__main__":
    # Debug mode (the interactive debugger + verbose tracebacks) is opt-in only, via
    # FLASK_DEBUG=true in your local .env. Never enable it in a deployed environment —
    # Flask's debugger allows arbitrary code execution if it's ever reachable.
    debug_mode = os.getenv("FLASK_DEBUG", "false").strip().lower() == "true"
    app.run(debug=debug_mode)
