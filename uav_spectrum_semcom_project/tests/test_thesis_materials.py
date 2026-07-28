from pathlib import Path

from spectrum_semcom.thesis_materials import local_markdown_links, validate_local_links, validate_reference_records, validate_required_markers


def test_required_markers_and_local_links(tmp_path: Path) -> None:
    target = tmp_path / "figure.png"
    target.write_bytes(b"x")
    document = tmp_path / "doc.md"
    document.write_text("required ![figure](figure.png) [web](https://example.com)", encoding="utf-8")
    assert validate_required_markers(document.read_text(), ("required",)) == []
    assert local_markdown_links(document.read_text()) == ["figure.png"]
    assert validate_local_links(document) == []


def test_broken_local_link_is_reported(tmp_path: Path) -> None:
    document = tmp_path / "doc.md"
    document.write_text("[missing](none.md)", encoding="utf-8")
    assert validate_local_links(document) == ["broken local link: none.md"]


def test_verified_reference_registry_rejects_bad_metadata() -> None:
    good = [
        {"id": "R1", "title": "Paper", "year": 2025, "identifier": "doi:1", "url": "https://example.org/paper", "status": "verified"}
    ]
    assert validate_reference_records(good, minimum=1) == []
    bad = good + [
        {"id": "R1", "title": "", "year": None, "identifier": "", "url": "http://example.org", "status": "candidate"}
    ]
    errors = validate_reference_records(bad, minimum=2)
    assert "reference ids must be non-empty and unique" in errors
    assert "unverified reference: R1" in errors
    assert "invalid primary-source url: R1" in errors
    assert "incomplete reference metadata: R1" in errors
