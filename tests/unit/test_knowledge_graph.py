# SPDX-License-Identifier: BSD-3-Clause
"""Knowledge graph: the real notes are consistent, and the parser handles the link syntax."""

from lunarsim.tools.knowledge_graph import (
    build_graph,
    check_knowledge_graph,
    extract_links,
    read_notes,
    render_overview,
)


def write_note(directory, identifier, body, note_type="component"):
    """Write one note with a valid header into ``directory``."""
    header = (
        f"---\nid: {identifier}\ntitle: Title of {identifier}\ntype: {note_type}\nstatus: current\n"
        f"updated: 2026-11-14\nsummary: Summary of {identifier}.\n---\n"
    )
    (directory / f"{identifier}.md").write_text(header + body, encoding="utf-8")


def test_repository_knowledge_graph_has_no_problems():
    """The checked-in notes have valid headers, no broken links and up-to-date generated files."""
    problems = check_knowledge_graph()
    assert problems == [], "\n".join(problems)


def test_extract_links_reads_relations_and_skips_code_blocks():
    """A list line names the relation; plain links are mentions; code is ignored."""
    body = "- depends on: [[alpha]]\nSee [[beta|the beta note]] and `[[delta]]`.\n```\n[[gamma]]\n```\n"
    assert extract_links(body) == [("depends on", "alpha"), ("mentions", "beta")]


def test_check_reports_broken_and_unlinked_notes(tmp_path):
    """A link to a missing note and a note nobody links to are both reported."""
    write_note(tmp_path, "first-note", "- uses: [[second-note]]\n- uses: [[missing-note]]\n")
    write_note(tmp_path, "second-note", "Nothing here.\n")
    problems = check_knowledge_graph(tmp_path)
    assert any("[[missing-note]] has no note" in problem for problem in problems)
    assert any(problem.startswith("first-note: no other note links to it") for problem in problems)


def test_header_problems_are_reported(tmp_path):
    """A wrong id, an unknown type and a missing header field are reported."""
    write_note(tmp_path, "good-note", "Text.\n", note_type="unknown-type")
    (tmp_path / "bad-note.md").write_text("---\nid: other-name\n---\nText\n", encoding="utf-8")
    _, problems = read_notes(tmp_path)
    assert any("unknown type" in problem for problem in problems)
    assert any("bad-note.md: header misses" in problem for problem in problems)


def test_graph_and_overview_contain_named_relations(tmp_path):
    """Edges keep their relation names, duplicates are dropped, and the diagram draws named edges."""
    write_note(tmp_path, "first-note", "- uses: [[second-note]]\n- uses: [[second-note]]\n[[second-note]]\n")
    write_note(tmp_path, "second-note", "- used by: [[first-note]]\n")
    notes, _ = read_notes(tmp_path)
    graph = build_graph(notes)
    assert {"source": "first-note", "target": "second-note", "relation": "uses"} in graph["edges"]
    assert len([edge for edge in graph["edges"] if edge["source"] == "first-note"]) == 2
    overview = render_overview(graph)
    assert 'first_note -- "uses" --> second_note' in overview
    assert "mentions" not in overview.split("```mermaid")[1].split("```")[0]
