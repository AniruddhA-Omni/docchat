"""Source-code parser: one element per top-level symbol (function / class / method) with line ranges.

Python uses the ``ast`` module; notebooks are split per cell; other languages use tree-sitter
when the optional ``code`` extra is installed, else a regex symbol scanner. Every element carries
``line_start``/``line_end`` so answers cite ``file.py:L10-42``.
"""

from __future__ import annotations

import ast
import contextlib
import json
import re
from pathlib import Path

from docchat.ingest.context import ParseContext
from docchat.ingest.detect import CODE_LANGUAGES
from docchat.ingest.parsers.text import read_text
from docchat.schemas import Element, Kind, ParsedDocument

MAX_SYMBOL_LINES = 120

# Start-of-symbol patterns for common languages (regex fallback).
_SYMBOL_PATTERNS: dict[str, re.Pattern[str]] = {
    "javascript": re.compile(
        r"^\s*(export\s+)?(default\s+)?(async\s+)?(function\*?\s+(?P<a>\w+)|class\s+(?P<b>\w+)|"
        r"(const|let|var)\s+(?P<c>\w+)\s*=\s*(async\s*)?(\([^)]*\)|\w+)\s*=>)"
    ),
    "java": re.compile(
        r"^\s*(public|private|protected|static|final|abstract|\s)*\s*(class|interface|enum|record)"
        r"\s+(?P<a>\w+)|^\s*(public|private|protected)[\w<>\[\],\s]*\s(?P<b>\w+)\s*\([^;]*$"
    ),
    "go": re.compile(r"^func\s+(\([^)]*\)\s*)?(?P<a>\w+)|^type\s+(?P<b>\w+)\s+(struct|interface)"),
    "rust": re.compile(
        r"^\s*(pub\s+)?(async\s+)?(fn\s+(?P<a>\w+)|struct\s+(?P<b>\w+)|"
        r"enum\s+(?P<c>\w+)|impl(<[^>]*>)?\s+(?P<d>[\w:]+)|trait\s+(?P<e>\w+))"
    ),  # fmt: skip
    "csharp": re.compile(
        r"^\s*(public|private|protected|internal|static|sealed|abstract|partial|\s)*\s*"
        r"(class|interface|enum|record|struct)\s+(?P<a>\w+)|"
        r"^\s*(public|private|protected|internal)[\w<>\[\],\s]*\s(?P<b>\w+)\s*\([^;]*$"
    ),
}
_SYMBOL_PATTERNS["typescript"] = _SYMBOL_PATTERNS["javascript"]
_SYMBOL_PATTERNS["kotlin"] = re.compile(r"^\s*(fun\s+(?P<a>\w+)|(data\s+)?class\s+(?P<b>\w+))")
_SYMBOL_PATTERNS["ruby"] = re.compile(
    r"^\s*(def\s+(?P<a>[\w.?!]+)|class\s+(?P<b>\w+)|module\s+(?P<c>\w+))"
)  # noqa: E501
_SYMBOL_PATTERNS["php"] = re.compile(r"^\s*(public\s+|private\s+|protected\s+|static\s+)*"
                                     r"(function\s+(?P<a>\w+)|class\s+(?P<b>\w+))")  # fmt: skip
_SYMBOL_PATTERNS["c"] = re.compile(r"^[A-Za-z_][\w\s\*]*\s\**(?P<a>\w+)\s*\([^;]*$")
_SYMBOL_PATTERNS["cpp"] = re.compile(
    r"^(class|struct)\s+(?P<a>\w+)|^[A-Za-z_][\w\s\*:&<>]*\s[\*&]*(?P<b>[\w:~]+)\s*\([^;]*$"
)  # noqa: E501


def parse_code(path: Path, ctx: ParseContext) -> ParsedDocument:
    language = CODE_LANGUAGES.get(path.suffix.lower(), "text")
    parsed = ParsedDocument(doc_id=ctx.doc_id, file_name=path.name, path=path, format="code",
                            parser="code", language=language)  # fmt: skip
    if path.suffix.lower() == ".ipynb":
        _parse_notebook(path, parsed)
        return parsed
    source = read_text(path).replace("\r\n", "\n")
    lines = source.split("\n")
    symbols: list[tuple[str, str, int, int]] = []  # (name, kind, start, end)
    if language == "python":
        try:
            symbols = _python_symbols(source)
            parsed.parser = "python-ast"
        except SyntaxError as exc:
            parsed.warnings.append(f"python parse failed ({exc}); using line splitter")
    if not symbols and language != "python":
        symbols = _tree_sitter_symbols(source, language) or _regex_symbols(lines, language)
        parsed.parser = "tree-sitter" if symbols and parsed.parser == "code" else parsed.parser

    parsed.elements.append(_outline(path, language, lines, symbols))
    if language == "python" and parsed.parser == "python-ast":
        parsed.symbols = _python_symbol_table(source)
    else:
        parsed.symbols = [{"symbol": n, "kind": k, "line_start": s, "line_end": e}
                          for n, k, s, e in symbols]  # fmt: skip
    covered = set()
    for name, kind, start, end in symbols:
        for s, e in _split_range(start, end):
            body = "\n".join(lines[s - 1 : e])
            parsed.elements.append(Element(
                kind=Kind.CODE, text=f"```{language}\n{body}\n```", line_start=s, line_end=e,
                meta={"symbol": name, "symbol_kind": kind, "language": language},
            ))  # fmt: skip
        covered.update(range(start, end + 1))
    # Module-level code not inside any symbol (imports, constants, scripts).
    for s, e in _uncovered_ranges(lines, covered):
        body = "\n".join(lines[s - 1 : e])
        if body.strip():
            parsed.elements.append(Element(kind=Kind.CODE, text=f"```{language}\n{body}\n```",
                                           line_start=s, line_end=e,
                                           meta={"symbol": "<module>", "language": language}))  # fmt: skip
    parsed.elements.sort(key=lambda el: (el.line_start or 0) if el.meta.get("symbol") else -1)
    return parsed


def _outline(path: Path, language: str, lines: list[str], symbols) -> Element:
    doc = ""
    if language == "python":
        with contextlib.suppress(SyntaxError):
            doc = ast.get_docstring(ast.parse("\n".join(lines))) or ""
    listing = "\n".join(f"- {kind} `{name}` (lines {s}-{e})" for name, kind, s, e in symbols)
    text = f"File {path.name} ({language}, {len(lines)} lines)."
    if doc:
        text += f"\nModule docstring: {doc}"
    if listing:
        text += f"\nDefines:\n{listing}"
    return Element(kind=Kind.PARAGRAPH, text=text, line_start=1, line_end=len(lines),
                   meta={"outline": True, "language": language})  # fmt: skip


def _python_symbols(source: str) -> list[tuple[str, str, int, int]]:
    tree = ast.parse(source)
    out = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            out.append((node.name, "function", _start(node), node.end_lineno))
        elif isinstance(node, ast.ClassDef):
            methods = [
                n for n in node.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
            ]
            span = node.end_lineno - _start(node) + 1
            if span <= MAX_SYMBOL_LINES or not methods:
                out.append((node.name, "class", _start(node), node.end_lineno))
                continue
            # Large class: header (up to first method) + one element per method.
            out.append((node.name, "class", _start(node), _start(methods[0]) - 1))
            for m in methods:
                out.append((f"{node.name}.{m.name}", "method", _start(m), m.end_lineno))
    return out


def _python_symbol_table(source: str) -> list[dict]:
    """Every function, class and method (qualified as Class.method) with its line range."""
    out = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            out.append({"symbol": node.name, "kind": "function",
                        "line_start": _start(node), "line_end": node.end_lineno})  # fmt: skip
        elif isinstance(node, ast.ClassDef):
            out.append({"symbol": node.name, "kind": "class",
                        "line_start": _start(node), "line_end": node.end_lineno})  # fmt: skip
            for m in node.body:
                if isinstance(m, ast.FunctionDef | ast.AsyncFunctionDef):
                    out.append({"symbol": f"{node.name}.{m.name}", "kind": "method",
                                "line_start": _start(m), "line_end": m.end_lineno})  # fmt: skip
    return out


def _start(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno, *(d.lineno for d in decorators)])


def _tree_sitter_symbols(source: str, language: str) -> list[tuple[str, str, int, int]]:
    try:
        from tree_sitter_language_pack import get_parser
    except ImportError:
        return []
    try:
        parser = get_parser(language)
    except Exception:
        return []
    tree = parser.parse(source.encode("utf-8"))
    wanted = ("function", "method", "class", "interface", "struct", "enum", "impl", "trait")
    out = []
    for child in tree.root_node.children:
        node = child
        if node.type in ("export_statement", "decorated_definition") and node.named_children:
            node = node.named_children[-1]
        if not any(w in node.type for w in wanted) and node.type != "lexical_declaration":
            continue
        name_node = node.child_by_field_name("name")
        if name_node is None and node.type == "lexical_declaration" and node.named_children:
            name_node = node.named_children[0].child_by_field_name("name")
        if name_node is None:
            continue
        kind = (
            "class" if any(w in node.type for w in ("class", "interface", "struct")) else "function"
        )
        out.append(
            (name_node.text.decode(), kind, child.start_point[0] + 1, child.end_point[0] + 1)
        )
    return out


def _regex_symbols(lines: list[str], language: str) -> list[tuple[str, str, int, int]]:
    pattern = _SYMBOL_PATTERNS.get(language)
    if pattern is None:
        return []
    starts = []
    for i, line in enumerate(lines, 1):
        m = pattern.match(line)
        if m:
            name = next((v for v in m.groupdict().values() if v), None)
            if name and name not in {"if", "for", "while", "switch", "return", "catch"}:
                kind = "class" if re.search(r"\b(class|struct|interface|enum|trait)\b", line) \
                    else "function"  # fmt: skip
                starts.append((i, name, kind))
    out = []
    for k, (start, name, kind) in enumerate(starts):
        end = starts[k + 1][0] - 1 if k + 1 < len(starts) else len(lines)
        while end > start and not lines[end - 1].strip():
            end -= 1
        out.append((name, kind, start, end))
    return out


def _split_range(start: int, end: int) -> list[tuple[int, int]]:
    if end - start + 1 <= MAX_SYMBOL_LINES:
        return [(start, end)]
    step, overlap = MAX_SYMBOL_LINES, 10
    return [(s, min(end, s + step - 1)) for s in range(start, end + 1, step - overlap)
            if s <= end]  # fmt: skip


def _uncovered_ranges(lines: list[str], covered: set[int]) -> list[tuple[int, int]]:
    ranges, start = [], None
    for i in range(1, len(lines) + 1):
        if i not in covered:
            start = start or i
        elif start:
            ranges.append((start, i - 1))
            start = None
    if start:
        ranges.append((start, len(lines)))
    out = []
    for s, e in ranges:
        out.extend(_split_range(s, e))
    return out


def _parse_notebook(path: Path, parsed: ParsedDocument) -> None:
    nb = json.loads(read_text(path))
    for k, cell in enumerate(nb.get("cells", []), 1):
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        if cell.get("cell_type") == "markdown":
            parsed.elements.append(Element(kind=Kind.PARAGRAPH, text=src.strip(), page=k,
                                           meta={"cell": k}))  # fmt: skip
        else:
            parsed.elements.append(Element(kind=Kind.CODE, text=f"```python\n{src}\n```", page=k,
                                           meta={"cell": k, "symbol": f"cell {k}",
                                                 "language": "python"}))  # fmt: skip
    parsed.parser = "notebook"
