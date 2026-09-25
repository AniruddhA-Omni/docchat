import datetime as dt

import pandas as pd
import pytest

from docchat.ingest.tables import (
    frame_to_table,
    grid_to_tables,
    parse_number,
    profile_table,
    rows_to_table,
    table_card,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("$1,234.50", 1234.5), ("(500)", -500.0), ("12%", 12.0), ("€ 3", 3.0), ("7", 7.0),
     ("abc", None), ("SO-0001", None), ("", None), (True, None), (4, 4.0)],
)  # fmt: skip
def test_parse_number(raw, expected):
    assert parse_number(raw) == expected


def test_report_style_sheet_with_title_merged_header_and_totals():
    grid = [
        ["Acme - Regional Summary", None, None, None, None],
        ["Source: export", None, None, None, None],
        [None, None, None, None, None],
        ["Region", "H1", None, "H2", None],
        ["Region", "Revenue (USD)", "Orders", "Revenue (USD)", "Orders"],
        ["North", "$1,000", 3, "2,000", 4],
        ["South", "500", 1, "(100)", 2],
        ["Total", 1500, 4, 1900, 6],
    ]
    [table] = grid_to_tables(grid, label="x / Summary", sheet="Summary")
    assert table.title == "Acme - Regional Summary"
    assert table.notes == ["Source: export"]
    assert list(table.columns.values()) == [
        "Region", "H1 Revenue (USD)", "H1 Orders", "H2 Revenue (USD)", "H2 Orders",
    ]  # fmt: skip
    df = table.df
    assert df["h1_revenue_usd"].tolist() == [1000, 500, 1500]
    assert df["h2_revenue_usd"].tolist() == [2000, -100, 1900]
    assert df["_is_total"].tolist() == [False, False, True]
    assert df["_row"].tolist() == [6, 7, 8]  # 1-based sheet rows for citations


def test_two_tables_on_one_sheet():
    grid = [["a", "b"], [1, 2], [3, 4], [None, None], ["Second table"], ["c", "d"], ["x", 1]]
    tables = grid_to_tables(grid, label="s")
    assert len(tables) == 2
    assert tables[1].title == "Second table"
    assert tables[1].df["_row"].tolist() == [7]


def test_ids_are_not_dates_but_dates_are():
    df = pd.DataFrame({"ticket_id": ["T-0001", "T-0002"], "created": ["2025-01-02", "2025-03-04T10:00:00"],
                       "n": ["1", "2"]})  # fmt: skip
    t = frame_to_table(df, label="t.csv")
    assert t.df["ticket_id"].tolist() == ["T-0001", "T-0002"]
    assert pd.api.types.is_datetime64_any_dtype(t.df["created"])
    assert t.df["n"].tolist() == [1, 2]


def test_all_text_table_keeps_single_header():
    rows = [
        ["ID", "Requirement", "Priority"],
        ["R-01", "Map a site", "Must"],
        ["R-02", "Dock", "Should"],
    ]
    t = rows_to_table(rows, label="prd")
    assert list(t.df.columns[:3]) == ["id", "requirement", "priority"]
    assert len(t.df) == 2


def test_profile_and_card():
    rows = [
        ["region", "revenue", "day"],
        ["EU", 10, dt.date(2025, 1, 1)],
        ["US", 30, dt.date(2025, 2, 1)],
    ]
    t = rows_to_table(rows, label="r")
    t.name = "sales"
    prof = {p["name"]: p for p in profile_table(t)}
    assert prof["revenue"]["type"] == "integer" and prof["revenue"]["max"] == 30
    assert prof["region"]["values"] == ["EU", "US"]
    assert prof["day"]["type"] == "date"
    card = table_card(t, "r.xlsx")
    assert "Table `sales`" in card and "revenue (integer" in card
