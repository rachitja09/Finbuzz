import importlib
import sys
import os

# Ensure project root is on sys.path so local modules (app, pages.*) can be imported
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

modules = ["app","pages.01_Home","pages.02_News_Sentiment","pages.03_Portfolio","pages.04_Compare"]
for m in modules:
    try:
        importlib.invalidate_caches()
        importlib.import_module(m)
    except Exception as e:
        print("FAILED:", m, e)
        raise
print("OK")
