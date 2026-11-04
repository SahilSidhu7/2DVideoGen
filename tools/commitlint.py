"""Commit message linter: hold new commits to the style this repo already uses.

The convention is written down in docs/COMMITS.md. In short:

    Imperative, sentence-case subject, 72 chars max, no period

    Body wrapped at 72 columns, explaining why rather than what.

    Co-Authored-By: Name <email>

Errors (exit 1):
    subject-empty    no subject line
    subject-length   subject over 72 characters
    subject-period   subject ends with a period
    subject-case     subject starts lowercase without an area prefix
    blank-line       line 2 is not blank
Warnings (reported, never fail):
    subject-long     subject over 60 characters
    body-length      body line over 72 columns, unless it is a trailer
                     (`Key: value` in the final paragraph) or holds a URL

An area prefix is a lowercase word and a colon (`paper: add ...`), or
`Attempt N:` for attempt-log commits. Comment lines, everything below the
scissors line, and messages git writes itself (merge, revert, fixup!,
squash!, amend!) are skipped.

Usage:
    python3 tools/commitlint.py .git/COMMIT_EDITMSG     # what the hook runs
    python3 tools/commitlint.py - < message.txt         # read stdin
    python3 tools/commitlint.py --range origin/master..HEAD
    python3 tools/commitlint.py --range A..B --quiet    # errors only
"""

import argparse
import re
import subprocess
import sys

SUBJECT_MAX = 72           # hard limit, matches the body wrap width
SUBJECT_PREFERRED = 60     # past this, warn
BODY_MAX = 72              # wrap width; warn only, history runs to ~78
SCISSORS = "# ------------------------ >8 ------------------------"

AREA_RE = re.compile(r"^([a-z][a-z0-9/_-]*|Attempt \d+):(?: |$)")
TRAILER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9-]*: \S")
SKIP_RE = re.compile(r'^(Merge |Revert "|fixup! |squash! |amend! )')


def clean(text):
    """Drop comment lines and anything below the scissors line."""
    lines = []
    for line in text.splitlines():
        if line.startswith(SCISSORS):
            break
        if not line.startswith("#"):
            lines.append(line.rstrip())
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return lines


def trailer_start(lines):
    """Index of the first line of the trailer block, or len(lines)."""
    start = len(lines)
    for i in range(len(lines) - 1, 1, -1):
        if not lines[i]:
            break
        if not TRAILER_RE.match(lines[i]):
            return len(lines)
        start = i
    return start


def lint(lines):
    """Return (errors, warnings), each a list of (line_number, rule, message)."""
    errors, warnings = [], []
    if not lines:
        return [(1, "subject-empty", "no subject line")], []
    subject = lines[0]
    if SKIP_RE.match(subject):
        return [], []
    m = AREA_RE.match(subject)
    rest = subject[m.end():] if m else subject
    if not rest.strip():
        errors.append((1, "subject-empty", "nothing after the prefix"))
    if len(subject) > SUBJECT_MAX:
        errors.append((1, "subject-length",
                       f"{len(subject)} chars, max {SUBJECT_MAX}"))
    elif len(subject) > SUBJECT_PREFERRED:
        warnings.append((1, "subject-long",
                         f"{len(subject)} chars, aim for {SUBJECT_PREFERRED}"))
    if subject.endswith("."):
        errors.append((1, "subject-period", "no trailing period"))
    if not m and subject[0].islower():
        errors.append((1, "subject-case",
                       "capitalize, or use a lowercase `area:` prefix"))
    if len(lines) > 1 and lines[1]:
        errors.append((2, "blank-line", "leave line 2 blank"))
    trailers = trailer_start(lines)
    for n, line in enumerate(lines[1:], start=2):
        if len(line) <= BODY_MAX or n - 1 >= trailers or "://" in line:
            continue
        warnings.append((n, "body-length", f"{len(line)} cols, wrap at {BODY_MAX}"))
    return errors, warnings


def report(label, errors, warnings, quiet=False):
    for n, rule, msg in errors:
        print(f"{label}:{n}: error: {rule}: {msg}")
    if not quiet:
        for n, rule, msg in warnings:
            print(f"{label}:{n}: warning: {rule}: {msg}")
    return bool(errors)


def lint_range(rev_range, quiet=False):
    out = subprocess.run(["git", "log", "--format=%H%x00%B%x00", rev_range],
                         capture_output=True, text=True, check=True).stdout
    parts = out.split("\x00")
    bad = False
    for sha, body in zip(parts[0::2], parts[1::2]):
        bad |= report(sha.strip()[:7], *lint(clean(body)), quiet=quiet)
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("file", nargs="?", help="message file, or - for stdin")
    ap.add_argument("--range", help="lint every commit in a rev range, e.g. A..B")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="print errors only")
    args = ap.parse_args()
    if args.range:
        return 1 if lint_range(args.range, args.quiet) else 0
    if not args.file:
        ap.error("give a message file, - for stdin, or --range")
    if args.file == "-":
        text = sys.stdin.read()
    else:
        with open(args.file, encoding="utf-8") as fh:
            text = fh.read()
    bad = report(args.file, *lint(clean(text)), quiet=args.quiet)
    if bad:
        print("See docs/COMMITS.md for the convention.", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
