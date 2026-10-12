#!/usr/bin/env python3
"""Generate a Keep-a-Changelog style CHANGELOG.md from git history.

Commits are grouped by tag (or by commit date when the repo has no tags) and
sorted into Added / Changed / Fixed / Docs / Other. Conventional-commit
prefixes win; otherwise the subject's leading keywords decide. Output is
deterministic: no generation timestamp.

    python tools/changelog.py -o CHANGELOG.md
    python tools/changelog.py --since 2026-09-01
"""
import argparse
import re
import subprocess
import sys
from collections import OrderedDict

SECTIONS = ["Added", "Changed", "Fixed", "Docs", "Other"]
CONVENTIONAL = {
    "feat": "Added", "add": "Added",
    "fix": "Fixed",
    "docs": "Docs", "doc": "Docs", "paper": "Docs",
    "refactor": "Changed", "perf": "Changed", "chore": "Changed",
    "style": "Changed", "build": "Changed", "ci": "Changed",
}
# (pattern on the lowercased subject, section), checked in order
KEYWORDS = [
    (r"^(fix|correct|repair)\b", "Fixed"),
    (r"\b(correct|fix)\b", "Fixed"),
    (r"^(add|complete|introduce|create)\b", "Added"),
    (r"^(ignore|reframe|make|split|stitch|attribute|name|change|update|"
     r"rename|remove|state)\b", "Changed"),
    (r"\battempts?\b", "Added"),
    (r"\b(post|posts|readme|paper|bibliography|write-up)\b", "Docs"),
]
SEP = "\x1f"
END = "\x1e"


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True,
                          text=True).stdout


def classify(subject):
    m = re.match(r"^(\w+)(\([^)]*\))?!?:\s", subject)
    if m and m.group(1).lower() in CONVENTIONAL:
        return CONVENTIONAL[m.group(1).lower()]
    low = subject.lower()
    for pat, section in KEYWORDS:
        if re.search(pat, low):
            return section
    return "Other"


def clean(subject):
    return re.sub(r"^(\w+)(\([^)]*\))?!?:\s", lambda m: "" if m.group(1).lower()
                  in CONVENTIONAL and m.group(1).lower() != "paper"
                  else m.group(0), subject).strip()


def tag_map():
    """Map full commit hash -> tag name for tags pointing at commits."""
    out = git("tag", "--list", "--format=%(refname:short)%09%(*objectname)%09%(objectname)")
    tags = {}
    for line in out.splitlines():
        name, peeled, direct = line.split("\t")
        tags[peeled or direct] = name
    return tags


def commits(since):
    args = ["log", "--no-merges", f"--format=%H{SEP}%h{SEP}%ad{SEP}%s{END}",
            "--date=short"]
    if since:
        args.append(f"--since={since}")
    for rec in git(*args).split(END):
        rec = rec.strip()
        if rec:
            full, short, date, subject = rec.split(SEP, 3)
            yield full, short, date, subject


def build(since):
    tags = tag_map()
    releases = OrderedDict()  # heading -> {section: [lines]}
    current = None
    for full, short, date, subject in commits(since):
        if full in tags:
            current = f"[{tags[full]}] - {date}"
        if current is None:
            current = date if not tags else "Unreleased"
        if not tags:
            current = date
        sec = releases.setdefault(current, OrderedDict((s, []) for s in SECTIONS))
        sec[classify(subject)].append(f"- {clean(subject)} (`{short}`)")
    lines = ["# Changelog", "",
             "Generated from git history by `tools/changelog.py`; do not edit by hand.",
             ""]
    for heading, sections in releases.items():
        lines += [f"## {heading}", ""]
        for name in SECTIONS:
            if sections[name]:
                lines += [f"### {name}", "", *sections[name], ""]
    return "\n".join(lines).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--output", help="write here instead of stdout")
    ap.add_argument("--since", help="only commits after this date (git --since)")
    args = ap.parse_args()
    text = build(args.since)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
