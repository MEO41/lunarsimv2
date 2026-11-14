# SPDX-License-Identifier: BSD-3-Clause
"""Project knowledge graph: linked Markdown notes in ``docs/knowledge``.

Each note is one Markdown file with a YAML header::

    ---
    id: slot-loop-architecture        # equals the file name without .md
    title: Slot loop architecture
    type: component                   # one of NOTE_TYPES
    status: current                   # one of NOTE_STATUSES
    updated: 2026-11-14
    summary: One line that says what the note holds.
    ---

Links between notes are written as ``[[note-id]]``. A list line of the form
``- relation name: [[note-id]]`` gives the link a relation name; every other link is a
"mentions" link. ``lunarsim knowledge-graph`` writes ``graph.json`` (nodes and edges) and
``graph_overview.md`` (a Mermaid diagram); ``lunarsim knowledge-graph --check`` fails on
broken links, bad headers, or generated files that are out of date.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

KNOWLEDGE_DIRECTORY = Path(__file__).resolve().parents[3] / "docs" / "knowledge"
GRAPH_JSON_NAME = "graph.json"
GRAPH_OVERVIEW_NAME = "graph_overview.md"
# Files in the directory that are not notes.
NON_NOTE_FILES = {"README.md", GRAPH_OVERVIEW_NAME}

NOTE_TYPES = (
    "hub",  # entry point, reading order
    "rule",  # standing instruction from the user
    "design",  # design documents, goals, scenarios
    "milestone",  # roadmap and gate status
    "component",  # a part of the code
    "decision",  # decisions already taken
    "result",  # measured numbers
    "issue",  # known weak points
    "plan",  # work planned, not started
    "research",  # facts gathered from public sources
    "environment",  # machines, installation, tools
    "history",  # what happened when
)
NOTE_STATUSES = ("current", "planned", "waiting-for-user", "open", "superseded")
REQUIRED_HEADER_FIELDS = ("id", "title", "type", "status", "updated", "summary")

LINK_PATTERN = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")
INLINE_CODE_PATTERN = re.compile(r"`[^`]*`")
RELATION_LINE_PATTERN = re.compile(r"^\s*[-*]\s+([A-Za-z][A-Za-z0-9 _-]*?)\s*:\s*\[\[([^\]|#]+)")


@dataclass
class KnowledgeNote:
    """One note of the knowledge graph."""

    identifier: str
    title: str
    note_type: str
    status: str
    updated: str
    summary: str
    path: Path
    # (relation name, target identifier) in the order they appear in the note.
    links: list[tuple[str, str]] = field(default_factory=list)


def split_header(text: str) -> tuple[dict, str]:
    """Split a note into its YAML header (as a dict) and its Markdown body."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end < 0:
        return {}, text
    header = yaml.safe_load(text[4:end]) or {}
    return header, text[end + 4 :]


def extract_links(body: str) -> list[tuple[str, str]]:
    """Return (relation, target) for every ``[[target]]`` link in a note body.

    Fenced code blocks and inline code spans are skipped, so examples of the link syntax are
    not counted.
    """
    links: list[tuple[str, str]] = []
    inside_code_block = False
    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            inside_code_block = not inside_code_block
            continue
        if inside_code_block:
            continue
        line = INLINE_CODE_PATTERN.sub("", line)
        relation_match = RELATION_LINE_PATTERN.match(line)
        relation = relation_match.group(1).strip().lower() if relation_match else "mentions"
        for target in LINK_PATTERN.findall(line):
            links.append((relation, target.strip()))
    return links


def read_notes(directory: Path = KNOWLEDGE_DIRECTORY) -> tuple[list[KnowledgeNote], list[str]]:
    """Read every note in ``directory``. Return the notes and a list of header problems."""
    notes: list[KnowledgeNote] = []
    problems: list[str] = []
    for path in sorted(directory.glob("*.md")):
        if path.name in NON_NOTE_FILES:
            continue
        header, body = split_header(path.read_text(encoding="utf-8"))
        missing = [name for name in REQUIRED_HEADER_FIELDS if name not in header]
        if missing:
            problems.append(f"{path.name}: header misses {', '.join(missing)}")
            continue
        if header["id"] != path.stem:
            problems.append(f"{path.name}: id '{header['id']}' differs from the file name")
        if header["type"] not in NOTE_TYPES:
            problems.append(f"{path.name}: unknown type '{header['type']}'")
        if header["status"] not in NOTE_STATUSES:
            problems.append(f"{path.name}: unknown status '{header['status']}'")
        notes.append(
            KnowledgeNote(
                identifier=str(header["id"]),
                title=str(header["title"]),
                note_type=str(header["type"]),
                status=str(header["status"]),
                updated=str(header["updated"]),
                summary=str(header["summary"]),
                path=path,
                links=extract_links(body),
            )
        )
    return notes, problems


def find_broken_links(notes: list[KnowledgeNote], directory: Path = KNOWLEDGE_DIRECTORY) -> list[str]:
    """Return a message for every link whose target note does not exist (README.md links included)."""
    known = {note.identifier for note in notes}
    broken = [
        f"{note.path.name}: [[{target}]] has no note"
        for note in notes
        for _, target in note.links
        if target not in known
    ]
    readme = directory / "README.md"
    if readme.is_file():
        broken += [
            f"README.md: [[{target}]] has no note"
            for _, target in extract_links(readme.read_text(encoding="utf-8"))
            if target not in known
        ]
    return broken


def find_unlinked_notes(notes: list[KnowledgeNote], directory: Path = KNOWLEDGE_DIRECTORY) -> list[str]:
    """Return the identifiers of notes that no other note and not the README link to."""
    linked = {target for note in notes for _, target in note.links if target != note.identifier}
    readme = directory / "README.md"
    if readme.is_file():
        linked |= {target for _, target in extract_links(readme.read_text(encoding="utf-8"))}
    return sorted(note.identifier for note in notes if note.identifier not in linked)


def build_graph(notes: list[KnowledgeNote]) -> dict:
    """Return the graph as a JSON-ready dict with ``nodes`` and ``edges`` (duplicates removed)."""
    nodes = [
        {
            "id": note.identifier,
            "title": note.title,
            "type": note.note_type,
            "status": note.status,
            "updated": note.updated,
            "summary": note.summary,
            "path": note.path.name,
        }
        for note in notes
    ]
    edges: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for note in notes:
        for relation, target in note.links:
            key = (note.identifier, target, relation)
            if target == note.identifier or key in seen:
                continue
            seen.add(key)
            edges.append({"source": note.identifier, "target": target, "relation": relation})
    return {"nodes": nodes, "edges": edges}


def mermaid_identifier(note_identifier: str) -> str:
    """Return a Mermaid-safe node name for a note identifier."""
    return re.sub(r"[^A-Za-z0-9_]", "_", note_identifier)


def render_overview(graph: dict) -> str:
    """Return a Markdown page with a Mermaid diagram of the graph, grouped by note type.

    Only named relations are drawn, so the picture stays readable; ``graph.json`` has all links.
    """
    lines = [
        "# Knowledge graph overview",
        "",
        "Generated by `lunarsim knowledge-graph`. Do not edit by hand.",
        "Only named relations are drawn; `graph.json` also holds the plain mentions.",
        "",
        "```mermaid",
        "flowchart LR",
    ]
    for note_type in NOTE_TYPES:
        members = [node for node in graph["nodes"] if node["type"] == note_type]
        if not members:
            continue
        lines.append(f"  subgraph {note_type}")
        for node in members:
            label = node["title"].replace('"', "'")
            lines.append(f'    {mermaid_identifier(node["id"])}["{label}"]')
        lines.append("  end")
    for edge in graph["edges"]:
        if edge["relation"] == "mentions":
            continue
        lines.append(
            f'  {mermaid_identifier(edge["source"])} -- "{edge["relation"]}" --> {mermaid_identifier(edge["target"])}'
        )
    lines += ["```", "", "## Notes", "", "| Note | Type | Status | Summary |", "| --- | --- | --- | --- |"]
    for node in sorted(graph["nodes"], key=lambda node: (NOTE_TYPES.index(node["type"]), node["id"])):
        lines.append(f"| [{node['title']}]({node['path']}) | {node['type']} | {node['status']} | {node['summary']} |")
    return "\n".join(lines) + "\n"


def generated_files(directory: Path = KNOWLEDGE_DIRECTORY) -> dict[Path, str]:
    """Return the content that ``graph.json`` and ``graph_overview.md`` should have now."""
    notes, _ = read_notes(directory)
    graph = build_graph(notes)
    return {
        directory / GRAPH_JSON_NAME: json.dumps(graph, indent=2, ensure_ascii=False) + "\n",
        directory / GRAPH_OVERVIEW_NAME: render_overview(graph),
    }


def check_knowledge_graph(directory: Path = KNOWLEDGE_DIRECTORY) -> list[str]:
    """Return every problem: bad headers, broken links, unlinked notes, out-of-date generated files."""
    notes, problems = read_notes(directory)
    problems += find_broken_links(notes, directory)
    problems += [f"{identifier}: no other note links to it" for identifier in find_unlinked_notes(notes, directory)]
    for path, content in generated_files(directory).items():
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            problems.append(f"{path.name} is out of date: run 'lunarsim knowledge-graph'")
    return problems


def write_knowledge_graph(directory: Path = KNOWLEDGE_DIRECTORY) -> list[Path]:
    """Write ``graph.json`` and ``graph_overview.md`` and return their paths."""
    written = []
    for path, content in generated_files(directory).items():
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written
