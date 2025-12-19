"""
Safe cleanup script for the repository.
- Deletes: __pycache__ directories, .pyc files, *.bak files, and the backups/ folder
- Skips common virtualenv folders (.venv-2, .venv, venv)

Run from the workspace root with: python ./scripts/cleanup_repo.py
"""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {'.venv-2', '.venv', 'venv', '.git'}

removed = {
    'pyc_files': [],
    'pycache_dirs': [],
    'bak_files': [],
    'backups_dirs': [],
}

for dirpath, dirnames, filenames in os.walk(ROOT, topdown=True):
    # Skip virtualenvs and .git
    dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

    # Remove .pyc files
    for fn in filenames:
        if fn.endswith('.pyc'):
            p = Path(dirpath) / fn
            try:
                p.unlink()
                removed['pyc_files'].append(str(p.relative_to(ROOT)))
            except Exception as e:
                print(f"Failed to remove {p}: {e}")

    # Remove .bak files
    for fn in filenames:
        if fn.endswith('.bak'):
            p = Path(dirpath) / fn
            try:
                p.unlink()
                removed['bak_files'].append(str(p.relative_to(ROOT)))
            except Exception as e:
                print(f"Failed to remove {p}: {e}")

# Remove __pycache__ directories and backups/ directories
for path in ROOT.rglob('*'):
    try:
        if path.is_dir() and path.name == '__pycache__':
            shutil.rmtree(path)
            removed['pycache_dirs'].append(str(path.relative_to(ROOT)))
        if path.is_dir() and path.name == 'backups':
            # Do a recursive delete of the backups dir
            shutil.rmtree(path)
            removed['backups_dirs'].append(str(path.relative_to(ROOT)))
    except Exception as e:
        print(f"Failed to remove {path}: {e}")

# Print summary
print('\nCleanup summary:')
print(f"  .pyc files removed: {len(removed['pyc_files'])}")
print(f"  __pycache__ dirs removed: {len(removed['pycache_dirs'])}")
print(f"  .bak files removed: {len(removed['bak_files'])}")
print(f"  backups/ dirs removed: {len(removed['backups_dirs'])}")

if removed['pyc_files']:
    print('\nSample .pyc removed:')
    print('\n'.join(removed['pyc_files'][:20]))

if removed['pycache_dirs']:
    print('\nSample __pycache__ removed:')
    print('\n'.join(removed['pycache_dirs'][:20]))

if removed['bak_files']:
    print('\nSample .bak removed:')
    print('\n'.join(removed['bak_files'][:20]))

if removed['backups_dirs']:
    print('\nbackups/ removed:')
    print('\n'.join(removed['backups_dirs']))

print('\nDone.')
