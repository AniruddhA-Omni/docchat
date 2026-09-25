"""Turn raw cell grids (spreadsheets, document tables) into clean, typed, SQL-ready DataFrames.

Handles report-style sheets: title rows above the table, multi-row / merged headers, several
tables on one sheet separated by blank rows, totals rows (flagged so aggregations can skip them)
and numbers stored as text ("$1,234", "12%", "(500)"). Every row keeps ``_row``, its 1-based row
number in the source, so answers can cite exact spreadsheet rows.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

import numpy as np
import pandas as pd

from docchat.schemas import TableData

TOTAL_RE = re.compile(r"^\s*(grand\s+|sub-?)?totals?\b", re.I)
_NUM_RE = re.compile(r"^\(?-?[$€£¥₹]?\s*-?[\d,]*\.?\d+\s*%?\)?$")
_DATE_RE = re.compile(
    r"^(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
    r"|\d{1,2}\s+[a-z]{3,9}\.?\s+\d{4}|[a-z]{3,9}\.?\s+\d{1,2},?\s+\d{4})"
    r"([ T]\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?)?\s*(z|[+-]\d{2}:?\d{2})?$",
    re.I,
)

Cell = Any
Grid = list[list[Cell]]


def is_empty(value: Cell) -> bool:
    return (
        value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == ""
    )


def parse_number(value: Cell) -> float | None:
    """Parse numbers stored as text. Returns None if the value is not numeric."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float | np.number):
        return float(value)
    s = str(value).strip().replace("\u00a0", " ")
    if not s or not _NUM_RE.match(s.replace(" ", "")):
        return None
    negative = s.startswith("(") and s.endswith(")")
    s = re.sub(r"[()$€£¥₹,%\s]", "", s)
    try:
        num = float(s)
    except ValueError:
        return None
    return -num if negative else num


def sql_identifier(text: str, fallback: str = "col") -> str:
    ident = re.sub(r"[^0-9a-zA-Z]+", "_", str(text)).strip("_").lower()
    if not ident:
        ident = fallback
    if ident[0].isdigit():
        ident = f"{fallback}_{ident}"
    return ident[:60]


def _dedupe(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for name in names:
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        out.append(name)
    return out


# ------------------------------------------------------------------------------------------------
# grid -> blocks -> tables
# ------------------------------------------------------------------------------------------------
def split_blocks(grid: Grid) -> list[tuple[int, Grid]]:
    """Split a grid on fully empty rows. Returns (0-based start row, rows) pairs."""
    blocks, current, start = [], [], 0
    for i, row in enumerate(grid):
        if all(is_empty(c) for c in row):
            if current:
                blocks.append((start, current))
            current = []
            continue
        if not current:
            start = i
        current.append(row)
    if current:
        blocks.append((start, current))
    return blocks


def _trim_columns(rows: Grid) -> Grid:
    width = max(len(r) for r in rows)
    rows = [list(r) + [None] * (width - len(r)) for r in rows]
    keep = [j for j in range(width) if any(not is_empty(r[j]) for r in rows)]
    return [[r[j] for j in keep] for r in rows]


def _is_texty(row: list[Cell]) -> bool:
    cells = [c for c in row if not is_empty(c)]
    return bool(cells) and all(
        isinstance(c, str) and parse_number(c) is None and not isinstance(c, dt.date) for c in cells
    )


def _n_filled(row: list[Cell]) -> int:
    return sum(not is_empty(c) for c in row)


def grid_to_tables(grid: Grid, label: str, sheet: str | None = None) -> list[TableData]:
    """Detect every table in a sheet-like grid (title rows become the table title)."""
    tables: list[TableData] = []
    pending_titles: list[str] = []
    for start, rows in split_blocks(grid):
        rows = _trim_columns(rows)
        width = len(rows[0])
        # Leading single-cell rows are titles / notes, not data.
        while rows and _n_filled(rows[0]) == 1 and (width > 1 or len(rows) == 1):
            pending_titles.append(str(next(c for c in rows[0] if not is_empty(c))).strip())
            rows, start = rows[1:], start + 1
        if len(rows) < 2 or width < 2:
            if rows:
                pending_titles.extend(" ".join(str(c) for c in r if not is_empty(c)) for r in rows)
            continue
        table = rows_to_table(rows, label=label, sheet=sheet, first_row=start + 1)
        if pending_titles:
            table.title = pending_titles[0]
            table.notes = pending_titles[1:]
            pending_titles = []
        tables.append(table)
    return tables


def rows_to_table(
    rows: Grid, label: str, sheet: str | None = None, first_row: int = 1, page: int | None = None
) -> TableData:
    """Build a typed DataFrame from rows whose leading row(s) are headers."""
    rows = _trim_columns(rows)
    n_header = 1
    # Multi-row headers only follow a "grouping" first row (blank or repeated, i.e. merged cells).
    first = [None if is_empty(c) else str(c).strip() for c in rows[0]]
    grouped = any(c is None for c in first) or any(
        a == b for a, b in zip(first, first[1:], strict=False)
    )
    while grouped and n_header < min(3, len(rows) - 1) and _is_texty(rows[n_header]):
        n_header += 1
    if n_header > 1 and all(_is_texty(r) for r in rows):
        n_header = 1  # an all-text table (e.g. a requirements list) has one header row

    header_rows = rows[:n_header]
    # Forward-fill horizontally merged header cells ("H1" spanning two columns).
    filled = []
    for hr in header_rows[:-1]:
        last, out = None, []
        for c in hr:
            last = c if not is_empty(c) else last
            out.append(last)
        filled.append(out)
    filled.append(header_rows[-1])

    labels = []
    for j in range(len(rows[0])):
        parts: list[str] = []
        for hr in filled:
            v = hr[j]
            if not is_empty(v) and str(v).strip() not in parts:
                parts.append(str(v).strip())
        labels.append(" ".join(parts) or f"column {j + 1}")
    sql_cols = _dedupe([sql_identifier(lbl) for lbl in labels])

    data = rows[n_header:]
    df = pd.DataFrame(data, columns=sql_cols)
    df = df.apply(_coerce_column)
    df["_is_total"] = [bool(TOTAL_RE.match(str(r[0] or ""))) for r in data]
    df["_row"] = list(range(first_row + n_header, first_row + n_header + len(data)))
    return TableData(
        df=df,
        columns=dict(zip(sql_cols, labels, strict=True)),
        label=label,
        sheet=sheet,
        page=page,
        header_row=first_row + n_header - 1,
    )


def _coerce_column(col: pd.Series) -> pd.Series:
    values = [v for v in col if not is_empty(v)]
    if not values:
        return col
    if all(isinstance(v, dt.datetime | dt.date | pd.Timestamp) for v in values):
        return pd.to_datetime(col, errors="coerce")
    nums = [parse_number(v) for v in values]
    if sum(n is not None for n in nums) >= 0.9 * len(values):
        parsed = col.map(lambda v: np.nan if is_empty(v) else parse_number(v))
        parsed = parsed.astype(float)
        if parsed.dropna().apply(float.is_integer).all():
            return parsed.astype("Int64")
        return parsed
    if all(isinstance(v, str) for v in values) and all(_DATE_RE.match(v.strip()) for v in values):
        dates = pd.to_datetime(col, errors="coerce", format="mixed")
        if dates.notna().sum() >= 0.9 * len(values):
            return dates
    return col.map(lambda v: None if is_empty(v) else str(v).strip())


def frame_to_table(df: pd.DataFrame, label: str, first_row: int = 2) -> TableData:
    """Clean an already-rectangular DataFrame (e.g. from CSV) with its header row."""
    labels = [str(c) for c in df.columns]
    sql_cols = _dedupe([sql_identifier(lbl) for lbl in labels])
    df = df.copy()
    df.columns = sql_cols
    df = df.apply(_coerce_column)
    df["_is_total"] = df[sql_cols[0]].map(lambda v: bool(TOTAL_RE.match(str(v or ""))))
    df["_row"] = range(first_row, first_row + len(df))
    return TableData(df=df, columns=dict(zip(sql_cols, labels, strict=True)), label=label,
                     header_row=first_row - 1)  # fmt: skip


# ------------------------------------------------------------------------------------------------
# profiling -> "table card" text indexed for retrieval and used in SQL prompts
# ------------------------------------------------------------------------------------------------
def profile_table(table: TableData, max_values: int = 6) -> list[dict[str, Any]]:
    df = table.df
    body = df[~df["_is_total"]] if "_is_total" in df else df
    profile = []
    for col in df.columns:
        if col in ("_row", "_is_total"):
            continue
        s = body[col]
        entry: dict[str, Any] = {
            "name": col,
            "label": table.columns.get(col, col),
            "nulls": int(s.isna().sum()),
            "distinct": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            entry["type"] = "integer" if pd.api.types.is_integer_dtype(s) else "number"
            if s.notna().any():
                entry.update(min=_py(s.min()), max=_py(s.max()), mean=round(float(s.mean()), 2))
        elif pd.api.types.is_datetime64_any_dtype(s):
            entry["type"] = "date"
            if s.notna().any():
                entry.update(min=str(s.min().date()), max=str(s.max().date()))
        else:
            entry["type"] = "text"
            top = s.value_counts().head(max_values)
            entry["values" if entry["distinct"] <= max_values else "examples"] = list(top.index)
        profile.append(entry)
    return profile


def _py(v: Any) -> Any:
    return v.item() if hasattr(v, "item") else v


def table_card(table: TableData, file_name: str, sample_rows: int = 3) -> str:
    """Compact description of a table: what it is, its columns and a few rows."""
    df = table.df
    n_total = int(df["_is_total"].sum()) if "_is_total" in df else 0
    where = f'sheet "{table.sheet}" in {file_name}' if table.sheet else file_name
    if table.page:
        where += f", page {table.page}"
    lines = [f"Table `{table.name}` ({where}): {len(df) - n_total} data rows."]
    if table.title:
        lines.append(f"Title: {table.title}")
    lines.extend(f"Note: {n}" for n in table.notes)
    if n_total:
        lines.append(f"{n_total} total row(s) flagged with _is_total = true.")
    lines.append("Columns:")
    for p in profile_table(table):
        desc = f"- {p['name']} ({p['type']}"
        if p["label"] != p["name"]:
            desc += f', header "{p["label"]}"'
        if "min" in p:
            desc += f"; {p['min']} to {p['max']}"
        if "values" in p:
            desc += "; values: " + ", ".join(map(str, p["values"]))
        elif "examples" in p:
            desc += "; e.g. " + ", ".join(map(str, p["examples"][:3]))
        lines.append(desc + ")")
    visible = df.drop(columns=[c for c in ("_is_total",) if c in df]).head(sample_rows)
    lines.append("Sample rows:")
    lines.append(dataframe_markdown(visible))
    return "\n".join(lines)


def dataframe_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    df = df.head(max_rows)
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        out.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(out)


def rows_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [[_fmt(c) for c in r] + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA or v is pd.NaT:
        return ""
    if isinstance(v, pd.Timestamp):
        return str(v.date()) if v == v.normalize() else str(v)
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).replace("|", "\\|").replace("\n", " ")
