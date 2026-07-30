"""Syntax check: compile all .py files without importing (no heavy deps needed).

Usage: python scripts/check_syntax.py
Exit code 0 = all OK, 1 = syntax errors found.
"""
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".venv", "__pycache__", ".git", "models", "node_modules"}

errors = []
count = 0

for py_file in sorted(ROOT.rglob("*.py")):
    # Skip excluded directories
    if any(part in SKIP_DIRS for part in py_file.relative_to(ROOT).parts):
        continue
    count += 1
    try:
        py_compile.compile(str(py_file), doraise=True)
    except py_compile.PyCompileError as e:
        errors.append(str(e))

if errors:
    print(f"SYNTAX ERRORS ({len(errors)}/{count} files):")
    for err in errors:
        print(f"  {err}")
    sys.exit(1)
else:
    print(f"OK: {count} files compiled without syntax errors")
    sys.exit(0)
