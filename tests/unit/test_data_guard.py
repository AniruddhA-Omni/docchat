import pandas as pd
import pytest

from docchat.data.pandas_ops import OpSpec, run_spec
from docchat.data.sandbox import QueryTimeout, Sandbox
from docchat.data.sql_guard import UnsafeSQLError, extract_sql, validate_sql
from docchat.index.catalog import TableRecord

ALLOWED = {"sales"}


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE sales",
        "DELETE FROM sales",
        "UPDATE sales SET revenue = 0",
        "INSERT INTO sales VALUES (1)",
        "CREATE TABLE x AS SELECT 1",
        "COPY sales TO 'out.csv'",
        "ATTACH 'other.db'",
        "PRAGMA database_list",
        "INSTALL httpfs",
        "LOAD httpfs",
        "SET enable_external_access = true",
        "SELECT 1; DROP TABLE sales",
        "SELECT * FROM read_csv('C:/secrets.csv')",
        "SELECT * FROM read_parquet('x.parquet')",
        "SELECT * FROM 'C:/data/secret.parquet'",
        "SELECT * FROM glob('*')",
        "SELECT getenv('PATH')",
        "SELECT * FROM read_text('C:/Windows/win.ini')",
        "SELECT * FROM duckdb_settings()",
        "SELECT * FROM other_table",
        "SELECT * FROM main.sales",
    ],
)
def test_guard_blocks_unsafe_sql(sql):
    with pytest.raises(UnsafeSQLError):
        validate_sql(sql, ALLOWED)


def test_guard_allows_queries_and_adds_limit():
    assert validate_sql("SELECT region FROM sales", ALLOWED).endswith("LIMIT 200")
    assert "LIMIT 1000" in validate_sql("SELECT * FROM sales LIMIT 99999", ALLOWED)
    cte = validate_sql("WITH t AS (SELECT * FROM sales) SELECT COUNT(*) FROM t", ALLOWED)
    assert cte.startswith("WITH")


def test_extract_sql_from_fence():
    assert extract_sql("Sure:\n```sql\nSELECT 1;\n```\nDone") == "SELECT 1"


@pytest.fixture
def sales_table(tmp_path):
    df = pd.DataFrame({
        "region": ["EU", "US", "EU", "Total"], "revenue": [10, 20, 30, 60],
        "_is_total": [False, False, False, True], "_row": [2, 3, 4, 5],
    })  # fmt: skip
    path = tmp_path / "sales.parquet"
    df.to_parquet(path)
    return TableRecord("sales", "doc", "sales.xlsx / S", None, "S", None, str(path), {}, "", 4)


def test_sandbox_runs_and_is_locked(sales_table):
    with Sandbox([sales_table]) as sb:
        res = sb.run(
            'SELECT region, SUM(revenue) AS r FROM sales WHERE NOT "_is_total" GROUP BY 1 ORDER BY 1'
        )
        assert res.df.to_dict("records") == [{"region": "EU", "r": 40}, {"region": "US", "r": 20}]
        with pytest.raises(Exception, match="(?i)permission|disabled|external"):
            sb.con.execute(f"SELECT * FROM read_parquet('{sales_table.parquet_path}')")
        with pytest.raises(Exception, match="(?i)lock|configuration"):
            sb.con.execute("SET enable_external_access = true")


def test_sandbox_timeout(tmp_path):
    path = tmp_path / "big.parquet"
    pd.DataFrame({"x": range(3000)}).to_parquet(path)
    table = TableRecord("big", "doc", "big", None, None, None, str(path), {}, "", 3000)
    with Sandbox([table], timeout_s=0.5) as sb, pytest.raises(QueryTimeout):
        sb.run("SELECT COUNT(*) FROM big a, big b, big c, big d WHERE a.x + b.x + c.x + d.x = -1")


def test_pandas_spec_excludes_totals_and_groups():
    df = pd.DataFrame({"region": ["EU", "US", "EU", "Total"], "revenue": [10, 20, 30, 60],
                       "_is_total": [False, False, False, True]})  # fmt: skip
    spec = OpSpec(table="sales", group_by=["region"],
                  aggregations=[{"column": "revenue", "func": "sum", "alias": "total"}],
                  sort=[{"column": "total", "descending": True}])  # fmt: skip
    out = run_spec(spec, df)
    assert out.to_dict("records") == [{"region": "EU", "total": 40}, {"region": "US", "total": 20}]
    single = run_spec(OpSpec(table="sales", filters=[{"column": "region", "op": "==", "value": "eu"}],
                             aggregations=[{"column": "revenue", "func": "mean"}]), df)  # fmt: skip
    assert single.iloc[0, 0] == 20
    with pytest.raises(ValueError):
        run_spec(OpSpec(table="sales", filters=[{"column": "nope", "op": "==", "value": 1}]), df)
