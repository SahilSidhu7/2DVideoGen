"""Commit message linter: enforce the convention this repo already follows.

Rules (subject = first line):
  subject-length   subject is at most 72 characters
  subject-case     subject does not start with a lowercase letter
  subject-period   subject has no trailing period
  subject-prefix   no `type:` / `type(scope):` prefix
  blank-line       line 2 is blank when there is a body
  body-length      body lines are at most 80 characters (wrap at ~72),
                   except URLs, trailers (`Key: value`) and indented or
                   fenced code

Comment lines (`#`), everything after the git scissors line, and merge,
revert and fixup!/squash! subjects are skipped.

Usage:
    python3 tools/commit_msg_lint.py .git/COMMIT_EDITMSG    # commit-msg hook
    python3 tools/commit_msg_lint.py --fix MSGFILE          # mechanical fixes
    python3 tools/commit_msg_lint.py --range origin/master..HEAD

`--fix` only capitalizes, strips a trailing period or a `type(scope):`
prefix, and inserts the missing blank line; it never re-wraps prose.
Exits 1 if any rule is violated.
"""

import argparse
import re
import subprocess
import sys

SUBJECT_MAX = 72
BODY_MAX = 80              # wrap at ~72; existing history runs up to 78
SCISSORS = "# ------------------------ >8 ------------------------"

PREFIX_RE = re.compile(r"^[a-z]+(\([^)]*\))?!?:\s+")
TRAILER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9-]*: \S")
SKIP_RE = re.compile(r"^(Merge |Revert |fixup! |squash! )")


def clean(text):
    """Drop comment lines and anything after the scissors line."""
    lines = []
    for line in text.splitlines():
        if line.startswith(SCISSORS):
            break
        if line.startswith("#"):
            continue
        lines.append(line)
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def lint(lines):
    """Return a list of (line_number, rule, message)."""
    if not lines or SKIP_RE.match(lines[0]):
        return []
    out = []
    subject = lines[0]
    if len(subject) > SUBJECT_MAX:
        out.append((1, "subject-length",
                    f"{len(subject)} chars, max {SUBJECT_MAX}"))
    if PREFIX_RE.match(subject):
        out.append((1, "subject-prefix", "drop the `type:` prefix"))
    if subject[0].islower():
        out.append((1, "subject-case", "start with an uppercase letter"))
    if subject.endswith("."):
        out.append((1, "subject-period", "no trailing period"))
    if len(lines) > 1 and lines[1].strip():
        out.append((2, "blank-line", "separate subject and body"))
    fenced = False
    for n, line in enumerate(lines[1:], start=2):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if (fenced or len(line) <= BODY_MAX or line[:1] in " \t"
                or "://" in line or TRAILER_RE.match(line)):
            continue
        out.append((n, "body-length", f"{len(line)} chars, max {BODY_MAX}"))
    return out


def fix(text):
    """Apply the safe mechanical fixes to a raw message; keep other lines."""
    lines = text.splitlines()
    i = next((k for k, l in enumerate(lines)
              if l.strip() and not l.startswith("#")), None)
    if i is None or SKIP_RE.match(lines[i]):
        return text
    subj = PREFIX_RE.sub("", lines[i], count=1).rstrip()
    subj = subj.rstrip(".")
    lines[i] = subj[:1].upper() + subj[1:]
    nxt = lines[i + 1] if i + 1 < len(lines) else ""
    if nxt.strip() and not nxt.startswith("#"):
        lines.insert(i + 1, "")
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def report(label, problems):
    for n, rule, msg in problems:
        print(f"{label}:{n}: {rule}: {msg}")
    return bool(problems)


def lint_range(rev_range):
    out = subprocess.run(["git", "log", "--format=%H%x00%B%x00", rev_range],
                         capture_output=True, text=True, check=True).stdout
    parts = out.split("\x00")
    bad = False
    for sha, body in zip(parts[0::2], parts[1::2]):
        bad |= report(sha.strip()[:7], lint(clean(body)))
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("file", nargs="?", help="commit message file")
    ap.add_argument("--range", help="lint every commit in a rev range")
    ap.add_argument("--fix", action="store_true",
                    help="apply safe mechanical fixes in place")
    args = ap.parse_args()
    if args.range:
        return 1 if lint_range(args.range) else 0
    if not args.file:
        ap.error("give a message file or --range")
    with open(args.file, encoding="utf-8") as fh:
        text = fh.read()
    if args.fix:
        fixed = fix(text)
        if fixed != text:
            with open(args.file, "w", encoding="utf-8") as fh:
                fh.write(fixed)
            text = fixed
    return 1 if report(args.file, lint(clean(text))) else 0


if __name__ == "__main__":
    sys.exit(main())
