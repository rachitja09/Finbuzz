from utils.watchlist import prefetch_quotes


def test_prefetch_empty():
    res = prefetch_quotes([])
    assert isinstance(res, dict)

# Note: we don't call external API in tests; just validate function runs with sample inputs
def test_prefetch_sample():
    res = prefetch_quotes(["XXXX", "YYYY"], max_workers=2)
    assert isinstance(res, dict)
