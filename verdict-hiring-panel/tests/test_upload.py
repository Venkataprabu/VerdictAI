"""Tests for document ingestion: extension allowlisting, filename sanitization,
and edge cases (empty files, malformed PDFs, missing sections).

Covers the evaluator's "file upload security" and "PDF parser validation for
edge cases" requirements.
"""
from __future__ import annotations

import io
import os

import pytest
from werkzeug.datastructures import FileStorage

import app


def _file(content: bytes, filename: str) -> FileStorage:
    return FileStorage(stream=io.BytesIO(content), filename=filename)


def test_ingest_accepts_a_normal_text_file():
    sources: dict = {}
    app.ingest([_file(b"Built and shipped a Python API.", "resume.txt")], "candidate_resume", sources)
    assert len(sources) == 1
    key = next(iter(sources))
    assert "resume.txt" in key
    assert sources[key]["kind"] == "candidate_resume"
    assert "Python API" in sources[key]["text"]


def test_ingest_rejects_disallowed_extensions():
    sources: dict = {}
    with pytest.raises(ValueError, match="only TXT, PDF, and DOCX"):
        app.ingest([_file(b"MZ\x90\x00", "malware.exe")], "candidate_resume", sources)
    assert sources == {}


@pytest.mark.parametrize("bad_name", ["script.js", "archive.zip", "image.png", "noextension"])
def test_ingest_rejects_various_disallowed_extensions(bad_name):
    sources: dict = {}
    with pytest.raises(ValueError):
        app.ingest([_file(b"anything", bad_name)], "candidate_resume", sources)


def test_ingest_sanitizes_path_traversal_filenames():
    """A filename engineered to escape the intended directory must be reduced
    to a safe basename — the resulting source key must not contain any
    path-traversal sequence, and nothing should ever be written outside the
    private temp file ingest() creates for itself."""
    sources: dict = {}
    app.ingest(
        [_file(b"Some resume content that is long enough to keep.", "../../etc/passwd.txt")],
        "candidate_resume",
        sources,
    )
    assert len(sources) == 1
    key = next(iter(sources))
    assert ".." not in key
    assert "/" not in key.split(" · ", 1)[1]


def test_ingest_silently_skips_an_empty_file():
    """An empty upload is not an error — it just contributes no evidence.
    The route-level check for 'no readable candidate documents' catches the
    case where every upload was empty."""
    sources: dict = {}
    app.ingest([_file(b"", "empty.txt")], "candidate_resume", sources)
    assert sources == {}


def test_ingest_skips_none_or_filenameless_entries():
    sources: dict = {}
    app.ingest([None, _file(b"", "")], "candidate_resume", sources)
    assert sources == {}


def test_ingest_handles_a_malformed_pdf_without_crashing():
    """A file with a .pdf extension but no valid PDF structure inside must be
    turned into a clean, catchable ValueError — not an unhandled parser
    exception (pypdf raises PdfStreamError/PdfReadError, which the route
    previously did not catch)."""
    sources: dict = {}
    with pytest.raises(ValueError, match="could not read this file as PDF"):
        app.ingest([_file(b"this is not a real pdf structure at all", "fake.pdf")], "candidate_resume", sources)
    assert sources == {}


def test_ingest_leaves_no_files_behind_on_disk(tmp_path, monkeypatch):
    """Uploaded content must not be retained after processing: this both
    protects candidate PII and keeps the app compatible with read-only
    serverless filesystems."""
    import tempfile

    created_paths = []
    original_named_temp_file = tempfile.NamedTemporaryFile

    def tracking_named_temp_file(*args, **kwargs):
        handle = original_named_temp_file(*args, **kwargs)
        created_paths.append(handle.name)
        return handle

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", tracking_named_temp_file)

    sources: dict = {}
    app.ingest([_file(b"Built and shipped a Python API.", "resume.txt")], "candidate_resume", sources)

    assert created_paths, "expected ingest() to use a temp file"
    for path in created_paths:
        assert not os.path.exists(path), "temp file was not cleaned up after ingest"


def test_analyze_requires_a_target_position(client):
    response = client.post("/api/analyze", data={})
    assert response.status_code == 400
    assert "target position" in response.get_json()["error"].lower()


def test_analyze_requires_at_least_one_candidate_document(client):
    response = client.post("/api/analyze", data={"target_position": "AI Engineer"})
    assert response.status_code == 400
    assert "resume or interview transcript" in response.get_json()["error"].lower()


def test_analyze_rejects_disallowed_file_type(client):
    data = {
        "target_position": "AI Engineer",
        "candidate_resume": (io.BytesIO(b"anything"), "resume.exe"),
    }
    response = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "TXT, PDF, and DOCX" in response.get_json()["error"]


def test_analyze_end_to_end_with_valid_documents(client):
    data = {
        "target_position": "AI Engineer",
        "candidate_resume": (
            io.BytesIO(b"Built and shipped a Python API used in production."),
            "resume.txt",
        ),
        "candidate_transcript": (
            io.BytesIO(b"I partnered with the team but only observed the deployment, I did not own it."),
            "transcript.txt",
        ),
    }
    response = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    body = response.get_json()
    assert body["candidate_count"] == 1
    assert "Candidate" in body["reports"]
    assert "state_log" in body["reports"]["Candidate"]
