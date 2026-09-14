# Verdict Hiring Panel

A runnable Flask web application for evidence-grounded, multi-agent candidate review.

> **If you're picking this project back up:** rotate any API keys that were ever
> in a `.env` file you shared outside your own machine (chat, a zip, a repo).
> Treat any key that left your machine as compromised, regardless of where it went.

## Run in VS Code

1. Open the `verdict-hiring-panel` folder in VS Code.
2. In its terminal, create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install packages and start the app:

   ```powershell
   pip install -r requirements.txt
   python app.py
   ```

4. Open `http://127.0.0.1:5000` in your browser.

## Run the test suite

```powershell
pip install -r requirements-dev.txt
pytest
```

The suite (`tests/test_agents.py`, `tests/test_debate.py`, `tests/test_upload.py`)
runs entirely offline against the app's local fallback logic — no API keys or
network access required, so it's safe to run in CI. It checks:

- **Agent independence** — each of the 4 personas (Technical, HR/Culture, Hiring
  Manager, Skeptic) is invoked separately, reads a distinct API key, and cannot
  see another agent's conclusion before forming its own (verified structurally
  via the function signature, not just by convention).
- **Debate quality** — later turns explicitly respond to earlier ones
  (`responds_to`), and an agent that revises its position produces a real,
  recorded score change with a stated reason.
- **Upload handling** — disallowed extensions are rejected, path-traversal
  filenames are sanitized, empty files are skipped without error, and a
  malformed PDF is turned into a clean validation error instead of crashing
  the request.

## API keys (live AI personas)

The app works in local evidence mode without keys. To enable live independent
persona calls, copy `.env.example` to `.env`, add the provider keys, then run
`python app.py`. The app reads `.env` automatically. **Never commit `.env` or
share it anywhere outside your own machine** — if a key ever leaves your
machine (chat, screenshot, public repo), rotate it immediately.

Key assignment deliberately preserves independence:

- `GROQ_API_KEY_TECHNICAL` - Groq, Technical Agent
- `GEMINI_API_KEY_CULTURE` - Gemini, HR / Culture Agent
- `GEMINI_API_KEY_HIRING` - Gemini, Hiring Manager Agent
- `GROQ_API_KEY_SKEPTIC` - Groq, Skeptic Agent
- `GEMINI_API_KEY_ADJUDICATOR` - optional Gemini narrative writer after the code-level adjudication
- `FLASK_DEBUG` - leave `false` (default). Only set to `true` for local debugging;
  never enable it in a deployed environment — Flask's interactive debugger
  allows arbitrary code execution if it's ever reachable.

## What works now

- Enter the target position; no job-posting file is required.
- Optionally attach a required-skills file and/or type role requirements.
- Upload a candidate resume, interview transcript, or both as separate inputs.
- The form starts with one candidate. Use `＋ Add Candidate B` only when you want a comparison; its resume and transcript inputs then appear.
- Extract text from each file and create exact, source-linked evidence facts (`E-01`, `E-02`, ...).
- Resume and transcript evidence share the candidate trust level, so separate upload labels do not hide valid evidence from the agents.
- Run four separately invoked panel personas with no access to another persona's pre-debate conclusion.
- Run a real, sequential debate where later turns directly respond to earlier turns and can revise a position.
- Use post-debate scores in code-level adjudication. Dimension weights are confidence-adjusted and redistributed when evidence is weak.
- Apply explicit thresholds and a Skeptic veto only when a specific, evidence-backed contradiction remains unresolved.
- Compute confidence from agreement, evidence coverage, debate resolution, and override penalty; the LLM does not invent the percentage.
- Show the full debate, an explicit before/after JSON state-change log per agent, changed positions, weighting math, strengths, concerns, and evidence ledger.
- Include `public/demo-debate.mp3`, a multi-voice local-speech recording of the debate format, plus its transcript.

## AI and independence

When keys are configured, `app.py` calls Groq or Gemini separately for each persona. Each independent call receives only the job context and verified evidence packet, never another agent's conclusion. The debate calls are separate from the independent calls and receive the full prior debate transcript. If a key/API call is absent or fails, the app falls back to a transparent local rule set and labels this clearly.

## Decision mechanics

1. Independent opinions are gathered in parallel and retain their own evidence citations.
2. Skeptic, Hiring Manager, and Technical agents take sequential debate turns; each turn has a `responds_to` field.
3. Final agent positions are derived after the debate, with visible pre/post scores and revision reasons, and collected into an explicit `state_log` array (`pre_debate_score`, `post_debate_score`, `changed`, `reason` per agent) so a position change is auditable as data, not just prose.
4. Default dimensions are Technical 35%, Hiring Manager 30%, HR/Culture 20%, and Skeptic 15%. Attached skills context can nudge the first three dimensions.
5. Low-confidence dimensions receive less weight; the released weight is redistributed across the other dimensions.
6. Scores map to Hire (>=8), Interview (5-7.99), or Reject (<5). A specific unresolved Skeptic contradiction caps a Hire at `INTERVIEW — VERIFY CREDIBILITY CLAIM`.
7. The optional adjudicator LLM writes the narrative only; it cannot silently change the computed recommendation.

## Security notes

- Uploaded files are processed through a private OS temp file that is deleted
  immediately after text extraction — nothing from a resume or transcript is
  retained on disk after the request finishes. This also keeps the app
  compatible with read-only serverless filesystems.
- Uploads are restricted to `.txt`, `.pdf`, and `.docx`, filenames are passed
  through `secure_filename()`, and the combined request size is capped
  (`MAX_CONTENT_LENGTH`, 32MB).
- A malformed file (e.g. a `.pdf` that isn't valid PDF structure) is caught
  and turned into a clean 400 response instead of an unhandled server error.
- Flask's debug mode is off unless you explicitly set `FLASK_DEBUG=true`
  locally — never enable it in a deployed environment.
- Basic security response headers (`X-Content-Type-Options`,
  `X-Frame-Options`, `Referrer-Policy`) are set on every response.
- `.env` is git-ignored. `.env.example` ships with blank placeholders only.

## Accessibility notes

- Every form control has a programmatically associated `<label>` (via
  matching `id`/`for`, not just visual proximity).
- File inputs are visually hidden but stay keyboard-reachable (no
  `display:none` on interactive controls).
- Focus states are visible (`:focus-visible` outlines were not suppressed).
- Status/error messages use `role="status"`/`aria-live="polite"` so screen
  readers announce them without needing to be re-read manually.
- The candidate comparison tabs follow the ARIA `tablist`/`tab` pattern with
  arrow-key navigation.
- Decorative icons/glyphs are marked `aria-hidden="true"` so they aren't
  announced as noise.
- A "skip to results" link is available for keyboard users.

## Deploying to Vercel

This app deploys to Vercel's Python runtime with effectively zero
configuration: Vercel auto-detects `app.py`'s Flask `app` instance as the
entrypoint. The included `vercel.json` only raises the function's max
duration to 60 seconds, since a full panel run can involve several
sequential LLM calls (independent opinions run in parallel, but the three
debate turns and the final positions are calls that can add up).

1. Push this project to a Git repository (or run `vercel deploy` from this folder).
2. Import the repository in the Vercel dashboard (or accept the CLI prompts).
3. In the project's Environment Variables settings, add the same keys listed
   above (`GROQ_API_KEY_TECHNICAL`, etc.) — never commit them in the repo.
4. Deploy.

Static assets (`public/demo-debate.mp3`, `public/demo-debate-transcript.txt`)
are served by Vercel's CDN directly from the `public/` directory rather than
through Flask, per Vercel's guidance for Flask apps. Locally, Flask serves
the same `public/` folder at the same root-level URLs
(`/demo-debate.mp3`), so no path differs between the two environments.

If you're on Vercel's Hobby plan and see timeouts on requests that involve
several live LLM calls, either raise `maxDuration` further (Pro plan) or
reduce per-provider call latency (smaller `GROQ_MODEL`/`GEMINI_MODEL`, or a
shorter `call_llm` timeout in `app.py`).

## Files

- `app.py` — Flask API, document parsing, evidence verification, panel pipeline
- `templates/index.html` — browser interface
- `public/` — static assets served at the same root URLs by both Flask locally and Vercel's CDN
- `tests/` — offline pytest suite covering agent independence, debate state changes, and upload validation
- `requirements.txt` — runtime Python dependencies
- `requirements-dev.txt` — adds `pytest` for running the test suite
- `vercel.json` — raises the deployed function's max duration for multi-call LLM requests
- `.env.example` — copy to `.env` and fill in your own keys; never commit `.env`

## Safety / scope

This is decision-support software, not an autonomous hiring decision-maker. Retain original source documents and evidence links for review outside this app if you need a long-term audit trail — this app itself does not persist uploaded documents. Do not let an agent cite a quote that was not verified in the source.
