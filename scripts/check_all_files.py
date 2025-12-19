"""Execute each .py file (excluding virtual env and __pycache__) to surface syntax/runtime errors.
This script is safe for scripts (it runs top-level code) but will skip tests and some known non-runnable files.
"""
import os
import runpy
import traceback
import sys

# Ensure project root is on sys.path so module imports inside runpy.run_path work like test imports
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.insert(0, root)
failures = []
for dirpath, dirnames, filenames in os.walk(root):
    # skip virtual envs and caches
    if '.venv' in dirpath or 'venv' in dirpath or '__pycache__' in dirpath or 'backups' in dirpath:
        continue
    for f in filenames:
        if not f.endswith('.py'):
            continue
        if f.startswith('test_'):
            continue
        path = os.path.join(dirpath, f)
        # skip this helper
        if os.path.abspath(path) == os.path.abspath(__file__):
            continue
        # skip data files
        rel = os.path.relpath(path, root)
        print('Running', rel)
        try:
            runpy.run_path(path, run_name='__main__')
        except Exception:
            print('ERROR in', rel)
            traceback.print_exc()
            failures.append((rel, traceback.format_exc()))

print('\nSummary: {} failures'.format(len(failures)))
for rel, tb in failures:
    print('---', rel)
    print(tb)

if failures:
    raise SystemExit(1)
print('All runnable files executed successfully')
