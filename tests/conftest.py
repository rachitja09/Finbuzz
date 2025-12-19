import os
import sys

# Ensure tests run with the project root on sys.path so tests can import project packages
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)
