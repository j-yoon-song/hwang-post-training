from synth_parallel.caches import SQLiteKVCache


def test_sqlite_kv_cache_set_many(tmp_path):
    cache = SQLiteKVCache(tmp_path / "cache.sqlite", table_name="kv")
    try:
        items = [
            ("a", {"x": 1}),
            ("b", {"x": 2}),
            ("c", {"x": 3}),
        ]
        cache.set_many(items)
        assert cache.get("a") == {"x": 1}
        assert cache.get("b") == {"x": 2}
        assert cache.get("c") == {"x": 3}
    finally:
        cache.close()
