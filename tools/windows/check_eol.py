#!/usr/bin/env python
"""Fail if a tracked text file has CRLF line endings.

The tree is LF. A Windows editor (or `git config core.autocrlf=true`) that rewrites a file to CRLF
breaks the POSIX shell scripts, the sed-based tooling and the HIP sources' line continuations, and it
shows up as a whole-file diff rather than as the line that changed. Run it from the repository root:

    python tools/windows/check_eol.py            # exit 1 and list offenders
    python tools/windows/check_eol.py --fix      # rewrite them to LF in place

The check reads the working tree, so it also catches files that are about to be committed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

TEXT_EXT = (
    ".py", ".sh", ".ps1", ".md", ".cu", ".cpp", ".c", ".h", ".hip", ".yml", ".yaml",
    ".json", ".txt", ".jinja", ".tmpl", ".nix",
)


def tracked_text_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], capture_output = True, check = True).stdout
    names = [n.decode("utf-8", "surrogateescape") for n in out.split(b"\0") if n]
    return [n for n in names if n.endswith(TEXT_EXT)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action = "store_true", help = "rewrite the offending files to LF")
    a = ap.parse_args()

    files = tracked_text_files()
    bad = []
    for f in files:
        try:
            with open(f, "rb") as fh:
                if b"\r\n" in fh.read():
                    bad.append(f)
        except OSError:
            pass                                    # deleted in the working tree: not our business

    if not bad:
        print(f"check_eol: {len(files)} tracked text files, all LF")
        return 0

    if a.fix:
        for f in bad:
            with open(f, "rb") as fh:
                data = fh.read()
            with open(f, "wb") as fh:
                fh.write(data.replace(b"\r\n", b"\n"))
        print(f"check_eol: rewrote {len(bad)} file(s) to LF")
        return 0

    print(f"check_eol: {len(bad)} of {len(files)} tracked text files have CRLF line endings:", file = sys.stderr)
    for f in bad[:40]:
        print(f"  {f}", file = sys.stderr)
    if len(bad) > 40:
        print(f"  ... and {len(bad) - 40} more", file = sys.stderr)
    print("fix: python tools/windows/check_eol.py --fix", file = sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
