"""Refuse to commit anything that could carry non-public material into this repo.

Run by ``.git/hooks/pre-commit`` over the staged files. A hit is a hard failure:
this repository is public and must contain public-benchmark work only.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Each pattern names a class of non-public material. Keep the list literal and readable:
# a regex nobody can explain is a regex nobody maintains.
FORBIDDEN: list[tuple[str, str]] = [
    (r"quantit", "employer name / internal domain"),
    (r"finter", "employer platform"),
    (r"arkraft|njtransit", "employer project"),
    (r"/home/[a-z]+/", "absolute home path"),
    (r"/locdisk\d", "internal disk mount"),
    (r"\b(?:10|172|192)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "private IP address"),
    (r"\bksadlq|\bksaplq", "internal hostname"),
    (r"xox[bpsa]-|ghp_[A-Za-z0-9]{20,}|glpat-|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}", "credential"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private key"),
    (r"\b(?:am|pm|ffd|mf|fm)_[a-z0-9]+_[a-z0-9_]+_v\d", "internal model identifier"),
    (r"\b(?:72659962|43129462|71149806|44858811)\b", "personal brokerage account number"),
]

ALLOW_FILES = {"tools/leak_guard.py", "DATA_POLICY.md"}


def staged_files() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        capture_output=True, text=True, check=True,
    )
    return [f for f in out.stdout.split("\n") if f]


def scan(paths: list[str]) -> list[str]:
    problems: list[str] = []
    for rel in paths:
        if rel in ALLOW_FILES:
            continue
        path = Path(rel)
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary fixture; nothing to read
        for pattern, label in FORBIDDEN:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                line = text[: m.start()].count("\n") + 1
                problems.append(f"{rel}:{line}: {label} -> {m.group(0)[:40]!r}")
    return problems


def main() -> int:
    paths = sys.argv[1:] or staged_files()
    problems = scan(paths)
    if problems:
        print("leak_guard: refusing to commit non-public material", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        return 1
    print(f"leak_guard: {len(paths)} file(s) clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
