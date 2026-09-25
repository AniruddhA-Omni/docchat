from pathlib import Path

from docchat.index.sparse import BM25Encoder, tokenize
from docchat.ingest.pipeline import delete_document, ingest_file


def test_tokenizer_keeps_identifiers_and_numbers():
    toks = tokenize("Invoice INV-2025-0917 totals $18,450.00; see list_items() and camelCase")
    assert "inv-2025-0917" in toks and "0917" in toks
    assert "18450.00" in toks and "18450" in toks
    assert "list_items" in toks and "item" in toks  # split + light stemming
    assert "camel" in toks and "the" not in toks


def test_bm25_saturates_term_frequency():
    enc = BM25Encoder()
    one = enc.encode_document("robot")
    many = enc.encode_document("robot " * 50)
    assert many.values[0] > one.values[0]
    assert many.values[0] < (enc.k1 + 1)  # bounded by k1 + 1


def _ingest(services, settings, corpus: Path, names):
    return {n: ingest_file(services, settings, corpus / n) for n in names}


def test_hybrid_search_filters_and_dedup(services, settings, corpus):
    results = _ingest(services, settings, corpus,
                      ["engineering_handbook.md", "meeting_notes_2025-11-03.txt", "inventory_service.py"])  # fmt: skip
    assert all(r.status == "ready" for r in results.values())
    again = ingest_file(services, settings, corpus / "engineering_handbook.md")
    assert again.status == "skipped"

    q = "When are production deploys frozen?"
    hits = services.store.search(
        services.embedder.embed_query(q), services.sparse.encode_query(q), limit=5
    )
    assert hits[0].chunk.file_name == "engineering_handbook.md"
    assert hits[0].dense_rank is not None and hits[0].sparse_rank is not None

    notes_id = results["meeting_notes_2025-11-03.txt"].doc_id
    scoped = services.store.search(services.embedder.embed_query(q), services.sparse.encode_query(q),
                                   limit=5, doc_ids=[notes_id])  # fmt: skip
    assert {h.chunk.doc_id for h in scoped} == {notes_id}

    sparse_only = services.store.search(
        None, services.sparse.encode_query("list_items"), mode="sparse"
    )
    assert sparse_only[0].chunk.file_name == "inventory_service.py"


def test_tables_registered_and_delete_cleans_up(services, settings, corpus):
    r = ingest_file(services, settings, corpus / "sales_2025.xlsx")
    tables = services.catalog.tables_for(r.doc_id)
    assert {t.name for t in tables} == {"sales_2025_orders", "sales_2025_regional_summary",
                                        "sales_2025_targets"}  # fmt: skip
    assert all(Path(t.parquet_path).exists() for t in tables)
    cards = services.store.document_chunks(r.doc_id)
    assert all(c.kind == "table_card" and c.table_name for c in cards)

    delete_document(services, settings, r.doc_id)
    assert services.store.count(r.doc_id) == 0
    assert services.catalog.tables_for(r.doc_id) == []
    assert not any(Path(t.parquet_path).exists() for t in tables)
    assert services.catalog.get_document(r.doc_id) is None


def test_code_symbols_in_catalog(services, settings, corpus):
    ingest_file(services, settings, corpus / "inventory_service.py")
    found = services.catalog.find_symbols(["list_items"])
    assert found and found[0]["symbol"] == "InventoryService.list_items"
