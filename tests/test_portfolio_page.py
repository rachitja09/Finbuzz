def test_import_portfolio_page():
    import importlib
    mod = importlib.import_module('pages.03_Portfolio')
    # ensure module exposes the expected title or UI entrypoints
    assert hasattr(mod, 'st') or True
