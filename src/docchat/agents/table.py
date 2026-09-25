"""Table / Data agent: text-to-SQL over the user's spreadsheets and document tables.

Flow: schema cards for up to 3 candidate tables -> LLM writes DuckDB SQL -> sqlglot guard ->
sandboxed execution with timeout -> up to 2 repair rounds on errors, 1 on empty results (with the
real distinct values as hints) -> pandas op-spec fallback. Results are exact numbers, cited to the
file / sheet / page and, for row-level answers, the source row numbers.
"""

from __future__ import annotations

import hashlib
import re

import pandas as pd

from docchat.agents.types import AgentContext, Evidence
from docchat.data.pandas_ops import OpSpec, run_spec
from docchat.data.sandbox import Sandbox
from docchat.data.sql_guard import UnsafeSQLError, extract_sql, referenced_columns
from docchat.index.catalog import TableRecord
from docchat.ingest.tables import dataframe_markdown
from docchat.llm import LLMError
from docchat.llm.prompts import PANDAS_SYSTEM, SQL_EMPTY, SQL_FIX, SQL_SYSTEM

MAX_TABLES = 3
MAX_ERROR_RETRIES = 2
PREVIEW_ROWS = 20


def _schema_block(tables: list[TableRecord]) -> str:
    return "\n\n".join(t.card for t in tables)


def _cite_location(t: TableRecord) -> str:
    if t.sheet:
        return f"sheet {t.sheet}"
    if t.page:
        return f"p.{t.page}"
    return "all rows"


def run(ctx: AgentContext, question: str, table_names: list[str],
        last_sql: dict | None = None) -> tuple[list[Evidence], dict]:  # fmt: skip
    catalog = ctx.services.catalog
    tables = [t for name in table_names[:MAX_TABLES] if (t := catalog.get_table(name))]
    trace: dict = {"agent": "table", "tables": [t.name for t in tables], "attempts": []}
    if not tables:
        trace["error"] = "no candidate tables"
        return [], trace

    llm = ctx.services.chat_llm(ctx.settings)
    prompt = f"Tables:\n{_schema_block(tables)}\n\nQuestion: {question}"
    if last_sql and set(last_sql.get("tables", [])) & {t.name for t in tables}:
        prompt += (f"\n\nThe previous question was answered with this query (modify it if the "
                   f"question is a follow-up):\n```sql\n{last_sql['sql']}\n```")  # fmt: skip
    messages = [{"role": "user", "content": prompt}]
    result_df: pd.DataFrame | None = None
    final_sql, method, retries = None, "sql", 0

    with Sandbox(tables) as sandbox:
        empty_retry_used = False
        for attempt in range(1 + MAX_ERROR_RETRIES + 1):
            try:
                reply = llm.chat(messages, system=SQL_SYSTEM, tag="text-to-sql", think=False,
                                 temperature=0.0, num_predict=400)  # fmt: skip
            except LLMError as exc:
                trace["error"] = str(exc)
                break
            try:
                sql = extract_sql(reply.content)
                res = sandbox.run(sql)
            except Exception as exc:  # guard rejections, timeouts and DuckDB errors
                err = str(exc).split("\n")[0][:300]
                trace["attempts"].append({"sql": _safe_sql(reply.content), "error": err})
                if attempt >= MAX_ERROR_RETRIES:
                    break
                retries += 1
                messages += [{"role": "assistant", "content": reply.content},
                             {"role": "user", "content": SQL_FIX.format(error=err)}]  # fmt: skip
                continue
            trace["attempts"].append({"sql": res.sql, "rows": len(res.df), "seconds": res.seconds})
            if res.df.empty and not empty_retry_used:
                empty_retry_used = True
                retries += 1
                hints = _value_hints(sandbox, res.sql, tables)
                messages += [{"role": "assistant", "content": reply.content},
                             {"role": "user", "content": SQL_EMPTY.format(values=hints)}]  # fmt: skip
                result_df, final_sql = res.df, res.sql
                continue
            result_df, final_sql = res.df, res.sql
            break

    if result_df is None:
        result_df, spec = _pandas_fallback(ctx, question, tables, trace)
        if result_df is None:
            return [], trace
        method, final_sql = "pandas", spec

    evidence = _to_evidence(question, tables, result_df, final_sql, method, retries)
    trace.update(method=method, sql=final_sql, rows=len(result_df), retries=retries)
    return [evidence], trace


def _safe_sql(reply: str) -> str:
    try:
        return extract_sql(reply)
    except UnsafeSQLError:
        return reply[:300]


def _value_hints(sandbox: Sandbox, sql: str, tables: list[TableRecord]) -> str:
    cols = referenced_columns(sql)
    lines = []
    for t in tables:
        for col in sandbox.text_columns(t.name):
            if col in cols:
                values = sandbox.distinct_values(t.name, col)
                lines.append(f'{t.name}."{col}": {", ".join(values)}')
    return "\n".join(lines) or "(no text columns were filtered)"


def _pandas_fallback(ctx: AgentContext, question: str, tables: list[TableRecord],
                     trace: dict) -> tuple[pd.DataFrame | None, str | None]:  # fmt: skip
    llm = ctx.services.chat_llm(ctx.settings)
    prompt = f"Tables:\n{_schema_block(tables)}\n\nQuestion: {question}"
    try:
        spec = llm.json([{"role": "user", "content": prompt}], OpSpec, system=PANDAS_SYSTEM,
                        tag="pandas-spec", num_predict=400)  # fmt: skip
        table = next(t for t in tables if t.name == spec.table)
        df = run_spec(spec, ctx.services.catalog.load_frame(table))
    except (LLMError, StopIteration, ValueError, KeyError, TypeError) as exc:
        trace["attempts"].append({"pandas": True, "error": str(exc)[:300]})
        return None, None
    trace["attempts"].append({"pandas": spec.describe(), "rows": len(df)})
    return df, f"pandas: {spec.describe()}"


def _to_evidence(question: str, tables: list[TableRecord], df: pd.DataFrame, sql: str,
                 method: str, retries: int) -> Evidence:  # fmt: skip
    used = [t for t in tables if re.search(rf'\b"?{re.escape(t.name)}"?\b', sql)] or tables[:1]
    main = used[0]
    row_ids = [int(r) for r in df["_row"].tolist()[:50]] if "_row" in df.columns else []
    shown = df.drop(columns=[c for c in ("_is_total",) if c in df.columns])
    files = ", ".join(dict.fromkeys(f"{t.file_name} ({_cite_location(t)})" for t in used))
    content = [f"Query result computed from {files}.", f"Method: {method.upper()}"]
    content.append(f"```sql\n{sql}\n```" if method == "sql" else sql)
    if df.shape == (1, 1):
        content.append(f"RESULT: {df.columns[0]} = {_fmt_value(df.iat[0, 0])}")
    content.append(f"Result ({len(df)} row{'s' if len(df) != 1 else ''}):")
    content.append(dataframe_markdown(shown, PREVIEW_ROWS) if len(df) else "(no rows)")
    if len(df) > PREVIEW_ROWS:
        content.append(f"... {len(df) - PREVIEW_ROWS} more rows not shown")
    digest = hashlib.sha1(f"{sql}|{question}".encode()).hexdigest()[:10]
    score = 1.0 if retries == 0 and method == "sql" else 0.85 if method == "sql" else 0.65
    if df.empty:
        score = 0.3
    return Evidence(
        id=f"sql:{digest}", agent="table", kind="table_result", content="\n".join(content),
        score=score, doc_id=main.doc_id, file_name=main.file_name,
        file_type="tabular" if main.sheet or not main.page else "",
        location=(f"{_cite_location(main)}, rows {_row_span(row_ids)}" if row_ids and main.sheet
                  else f"rows {_row_span(row_ids)}" if row_ids else _cite_location(main)),
        page=main.page, sheet=main.sheet, table_name=main.name, sql=sql, row_ids=row_ids,
        section=main.title or "",
    )  # fmt: skip


def _row_span(rows: list[int]) -> str:
    if len(rows) <= 6:
        return ", ".join(map(str, rows))
    return f"{', '.join(map(str, rows[:5]))} … ({len(rows)} rows)"


def _fmt_value(v) -> str:
    if isinstance(v, float) and v.is_integer():
        return f"{int(v)}"
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1e6 else f"{v:,.2f}"
    return str(v)
