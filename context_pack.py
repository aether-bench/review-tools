#!/usr/bin/env python3
"""Context pack for a pull-request review.

Writes one markdown file that gives every reviewer the same bounded view of a change, so that reviewers read the pack
instead of each exploring the repository. For each changed file it prints every hunk inside its enclosing function,
before and after the change; the file's signature skeleton; the callers of every changed or removed function; the
definitions of the functions the added lines call; and every literal the change adds or removes together with its other
occurrences in the repository. Generated and lock files (lockfiles, minified bundles, snapshots, vendored trees) are
listed but not shown. The pack is purely structural: it applies no language-specific bug rules and judges nothing. Its
output is deterministic for a given base, head and budget.

usage: context_pack.py --base <ref> --head <ref> [--title TITLE] [--body-file FILE] [--max-tokens N] [--out FILE]
Run inside the repository; --base is the commit the change is diffed from (pass the merge base). Hunks are numbered per
file exactly as `git diff --unified=3 <base> <head>` prints them. Exits non-zero on any git failure.
"""
import argparse
import re
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

NOISE_PATHSPECS = [
    ":!**/node_modules/**", ":!**/vendor/**", ":!**/dist/**", ":!**/build/**", ":!**/__snapshots__/**",
    ":!*.lock", ":!*lock.json", ":!*lock.yaml", ":!*.min.js", ":!*.min.css", ":!*.map", ":!*.snap", ":!*.svg", ":!*.sum",
    ":!*.po", ":!*.pot", ":!*.pysnap", ":!**/snapshots/**",
]

# git's builtin diff drivers give language-aware hunk headers and whole-function context (-W); JavaScript and TypeScript
# have no builtin driver, so one is defined below. Real tabs and newlines are required: git compiles these as POSIX ERE.
ATTRIBUTES = """\
*.py diff=python
*.rb diff=ruby
*.rake diff=ruby
*.go diff=golang
*.java diff=java
*.scala diff=java
*.kt diff=kotlin
*.kts diff=kotlin
*.rs diff=rust
*.php diff=php
*.c diff=cpp
*.h diff=cpp
*.cc diff=cpp
*.cpp diff=cpp
*.hpp diff=cpp
*.cs diff=csharp
*.pl diff=perl
*.pm diff=perl
*.ex diff=elixir
*.exs diff=elixir
*.ts diff=tsjs
*.tsx diff=tsjs
*.js diff=tsjs
*.jsx diff=tsjs
*.mjs diff=tsjs
*.cjs diff=tsjs
*.vue diff=tsjs
*.svelte diff=tsjs
"""
TSJS_XFUNCNAME = (
    "!^[ \t]*(if|for|while|switch|catch|else|return|try|do|await)[ \t(]"
    "\n"
    "^[ \t]*(export[ \t]+)?(default[ \t]+)?(async[ \t]+)?("
    "function[ \t*]+[A-Za-z_$][A-Za-z0-9_$]*"
    "|class[ \t]+[A-Za-z_$][A-Za-z0-9_$]*"
    "|(const|let|var)[ \t]+[A-Za-z_$][A-Za-z0-9_$]*[^=]*=[ \t]*(async[ \t]+)?(function[ \t(]|\\(|[A-Za-z_$][A-Za-z0-9_$]*[ \t]*=>)"
    "|(public|private|protected|static|readonly|get|set|async|override|[ \t])*[A-Za-z_$][A-Za-z0-9_$]*[ \t]*(<[^>]*>)?\\([^;]*\\{[ \t]*$"
    ")"
)

# language family by extension: function context, skeleton regexes and definition matching are per family
FAMILY = {
    "py": "py", "pyi": "py", "rb": "rb", "rake": "rb", "go": "go", "java": "java", "scala": "java", "kt": "kt", "kts": "kt",
    "rs": "rs", "php": "php", "c": "c", "h": "c", "cc": "c", "cpp": "c", "hpp": "c", "cs": "cs", "pl": "perl", "pm": "perl",
    "ex": "ex", "exs": "ex", "ts": "tsjs", "tsx": "tsjs", "js": "tsjs", "jsx": "tsjs", "mjs": "tsjs", "cjs": "tsjs",
    "vue": "tsjs", "svelte": "tsjs", "es6": "tsjs", "swift": "swift",
}
SKELETON_RES = {
    "py": re.compile(r"^\s*(class|def|async def)\s"),
    "rb": re.compile(r"^\s*(class|module|def)\s"),
    "go": re.compile(r"^(func|type)\s"),
    "rs": re.compile(r"^\s*(pub(\([^)]*\))?\s+)?(fn|struct|enum|trait|impl|type|mod)\s"),
    "java": re.compile(r"^\s*((public|private|protected|static|final|abstract|default|synchronized|native)\s+)*([\w<>\[\],.?]+\s+)+\w+\s*\([^;]*$|^\s*(public\s+|private\s+|protected\s+)?(abstract\s+|final\s+|static\s+)*(class|interface|enum|record)\s"),
    "kt": re.compile(r"^\s*((public|private|protected|internal|override|open|abstract|suspend|inline)\s+)*(fun|class|object|interface|val|var)\s"),
    "cs": re.compile(r"^\s*((public|private|protected|internal|static|override|virtual|abstract|async)\s+)+[\w<>\[\],.?]+\s+\w+\s*\(|^\s*(class|interface|enum|struct|record)\s"),
    "tsjs": re.compile(r"^\s*(export\s+)?(default\s+)?(async\s+)?(function|class|interface|type|enum|const|let|var)\s+[A-Za-z_$]|^\s{2,}(public|private|protected|static|async|get|set|readonly|override|\s)*[A-Za-z_$][\w$]*\s*(<[^>]*>)?\([^)]*\)?\s*(:\s*[^{]+)?\{\s*$|^\s{2,}[A-Za-z_$][\w$]*\s*:\s*(async\s*)?(function\b|\(|[A-Za-z_$][\w$]*\s*=>)"),
    "php": re.compile(r"^\s*((public|private|protected|static|abstract|final)\s+)*(function|class|interface|trait)\s"),
    "c": re.compile(r"^[A-Za-z_].*\)\s*\{?\s*$|^\s*(struct|enum|typedef|#define)\s"),
    "swift": re.compile(r"^\s*((public|private|internal|open|static|override|final)\s+)*(func|class|struct|enum|protocol|extension|var|let)\s"),
}

KEYWORDS = {
    "if", "for", "while", "switch", "catch", "return", "function", "def", "new", "typeof", "super", "this", "self", "else",
    "elif", "elsif", "unless", "until", "when", "case", "except", "finally", "loop", "begin", "rescue", "ensure", "select",
    "defer", "go", "match", "try", "do", "await", "async", "yield", "class", "import", "from", "require", "export",
    "default", "throw", "raise", "with", "as", "in", "is", "not", "and", "or", "lambda", "proc", "print", "len", "str",
    "int", "float", "bool", "list", "dict", "set", "tuple", "range", "enumerate", "zip", "isinstance", "hasattr",
    "getattr", "setattr", "sorted", "min", "max", "sum", "abs", "any", "all", "map", "filter", "reduce", "forEach", "find",
    "includes", "indexOf", "slice", "splice", "concat", "keys", "values", "entries", "assign", "freeze", "parse",
    "stringify", "toString", "String", "Number", "Boolean", "Array", "Object", "Promise", "resolve", "reject", "then",
    "expect", "describe", "it", "test", "beforeEach", "afterEach", "beforeAll", "afterAll", "assert", "console", "log",
    "logger", "debug", "info", "warn", "warning", "error", "format", "join", "split", "strip", "append", "push", "pop",
    "get", "set", "add", "remove", "delete", "has", "make", "cap", "panic", "recover", "puts", "each", "times", "fetch",
    "Sprintf", "Errorf", "Printf", "Println", "Fprintf", "Error", "Exception", "RuntimeError", "ValueError", "TypeError",
    "require_relative", "attr_reader", "attr_accessor", "private", "public", "protected", "static", "void", "sizeof",
    "defined", "mock", "patch", "jest", "vi", "mocker", "cy", "sleep", "timing", "timer", "incr", "gauge", "Partition",
}
MAX_HITS_PER_NAME = 300
MAX_CALLER_LINES = 20
MAX_CALLEES = 40
MAX_DEFINITIONS_PER_CALLEE = 2
DEFINITION_LINES = 15
MAX_LITERALS = 80
MAX_LITERAL_LOCATIONS = 10
MAX_SKELETON_LINES = 80
WHOLE_FUNCTION_MAX = 200     # a -W hunk up to this many lines is printed whole
WINDOW = 60                  # otherwise this many lines around each change


def git(*args: str, ok_codes=(0,)) -> str:
    """git's stdout decoded as UTF-8 with no newline translation, so a lone carriage return stays inside its line and
    hunk payload counts stay right; a byte that is not UTF-8 becomes U+FFFD."""
    r = subprocess.run(["git", *args], capture_output=True)
    if r.returncode not in ok_codes:
        raise SystemExit(f"git {' '.join(args[:6])} ... failed ({r.returncode}): {r.stderr.decode('utf-8', errors='replace').strip()[:500]}")
    return r.stdout.decode("utf-8", errors="replace")


def family(path: str) -> str:
    ext = path.rsplit(".", 1)[-1].lower() if "." in path.rsplit("/", 1)[-1] else ""
    return FAMILY.get(ext, "")


@dataclass
class Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    header: str
    lines: list = field(default_factory=list)  # (tag, text, old_line or None, new_line or None)

    @property
    def new_end(self) -> int:
        return self.new_start + max(self.new_len - 1, 0)


@dataclass
class FileDiff:
    path: str
    old_path: str
    status: str  # A, M, D, or T for a file that became a symlink or the reverse
    binary: bool = False  # a half of a type change may be binary while the other half has text hunks
    hunks: list = field(default_factory=list)


HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@ ?(.*)$")


def name_status(cfg: list, base: str, head: str, pathspecs: list) -> list:
    """(status, old_path, new_path) per changed file, in diff order, from NUL-delimited output: the exact paths, however
    git would quote or escape them in a text header. A type change (`T`: a file became a symlink, or the reverse) is
    two entries, D then A, because the patch prints it as a deletion block followed by an addition block."""
    fields = git(*cfg, "diff", "--name-status", "-z", base, head, "--", ".", *pathspecs).split("\0")
    entries, i = [], 0
    while i < len(fields) and fields[i]:
        status = fields[i]
        if status[0] in ("R", "C"):
            entries.append((status[0], fields[i + 1], fields[i + 2]))
            i += 3
        elif status[0] == "T":
            entries += [("D", fields[i + 1], fields[i + 1]), ("A", fields[i + 1], fields[i + 1])]
            i += 2
        else:
            entries.append((status[0], fields[i + 1], fields[i + 1]))
            i += 2
    return entries


def parse_diff(text: str, entries: list) -> "OrderedDict[str, FileDiff]":
    """Hunks of a unified diff. File identities come from `entries` (name_status, in the order of the diff blocks), so a
    quoted or escaped path in a text header is never parsed; hunk payload is consumed by the counts the hunk header
    declares, so a payload line that starts with `--- ` or `+++ ` is content and the trailing empty split item is not."""
    files: "OrderedDict[str, FileDiff]" = OrderedDict()
    cur = None
    hunk = None
    old_line = new_line = old_left = new_left = 0
    block = 0
    for raw in text.split("\n"):
        if raw.startswith("diff --git "):
            if block >= len(entries):
                raise SystemExit(f"diff has more file blocks than `git diff --name-status` listed: {raw}")
            status, old_path, path = entries[block]
            block += 1
            if status == "A" and path in files and files[path].status == "D":
                cur = files[path]  # the addition half of a type change joins its deletion half
                cur.status = "T"
            else:
                cur = FileDiff(path=path, old_path=old_path, status=status if status in ("A", "D") else "M")
                files[cur.path] = cur
            hunk = None
            continue
        if cur is None:
            continue
        if hunk is not None and (old_left or new_left):
            tag = raw[:1] if raw else " "
            if tag == "\\":  # "\ No newline at end of file"
                continue
            text_ = raw[1:]
            if tag == " ":
                hunk.lines.append((" ", text_, old_line, new_line))
                old_line, new_line, old_left, new_left = old_line + 1, new_line + 1, old_left - 1, new_left - 1
            elif tag == "-":
                hunk.lines.append(("-", text_, old_line, None))
                old_line, old_left = old_line + 1, old_left - 1
            elif tag == "+":
                hunk.lines.append(("+", text_, None, new_line))
                new_line, new_left = new_line + 1, new_left - 1
            else:
                raise SystemExit(f"unparsed hunk line in {cur.path}: {raw!r}")
            if old_left < 0 or new_left < 0:
                raise SystemExit(f"hunk in {cur.path} is longer than its header declares")
            continue
        if raw.startswith("Binary files"):
            cur.binary = True
        elif raw.startswith("@@"):
            m = HUNK_RE.match(raw)
            if not m:
                raise SystemExit(f"unparsed hunk header: {raw}")
            hunk = Hunk(int(m.group(1)), int(m.group(2) or 1), int(m.group(3)), int(m.group(4) or 1), m.group(5))
            cur.hunks.append(hunk)
            old_line, new_line = hunk.old_start, hunk.new_start
            old_left, new_left = hunk.old_len, hunk.new_len
    if block != len(entries):
        raise SystemExit(f"diff has {block} file blocks but `git diff --name-status` listed {len(entries)}")
    return files


def symbol_from_funcline(text: str):
    t = text.strip()
    for pat in (
        r"(?:def|func(?:tion)?|class|module|fn|interface|struct|type|enum|record)\s+(?:\([^)]*\)\s*)?\*?([A-Za-z_$][\w$]*)",
        r"(?:const|let|var|val)\s+([A-Za-z_$][\w$]*)",
        r"([A-Za-z_$][\w$]*)\s*(?:<[^>]*>)?\s*\(",
    ):
        m = re.search(pat, t)
        if m and m.group(1) not in KEYWORDS and len(m.group(1)) >= 3:
            return m.group(1)
    return None


# Declaration syntax is per language family. A typed declaration (`int name(`, `public static Foo name(`) exists only
# in the families below; elsewhere every declaration is keyword-led, so Go's `defer name()` and `go name()` are calls.
TYPED_DECLARATION_FAMILIES = {"c", "java", "cs", "kt", "php", "swift", "tsjs", ""}
# A line led by one of these is a statement that calls or uses `name`, never a typed declaration of it.
STATEMENT_KEYWORDS = {"return", "throw", "throws", "new", "yield", "await", "else", "case", "if", "while", "for", "switch",
                      "catch", "try", "do", "goto", "break", "continue", "not", "and", "or", "in", "as", "is", "typeof",
                      "delete", "elif", "elsif", "unless", "until", "when", "raise", "assert", "print", "puts", "defer",
                      "go", "sizeof", "echo"}
# `void name()` is an expression in JavaScript and TypeScript and a declaration in C, Java and C#.
EXPRESSION_KEYWORDS = {"tsjs": {"void"}, "": {"void"}}
_definition_regexes: dict = {}


def definition_regex(name: str, fam: str) -> re.Pattern:
    """A line that declares `name` in a file of language family `fam`: a keyword-led declaration (def, func, class, ...)
    or a const/let/var/val binding in every family; a typed declaration, led by modifier or type tokens that are not
    statement keywords of that family, where the language writes declarations that way; a property bound to a function;
    a method whose body opens on the same line, outside Python and Ruby where `name(args) {` is a call with a block;
    and a Ruby singleton method. A bare call `name(...)`, `return name(...)` or `defer name(...)` does not match."""
    key = (name, fam)
    if key not in _definition_regexes:
        n = re.escape(name)
        alternatives = [
            rf"(?:^|[^\w$.])(?:def|func|function|fn|fun|sub|class|module|interface|type|struct|enum|record|trait|impl)\s+(?:\([^)]*\)\s*)?\*?{n}\s*[(<:{{\[=]",
            rf"(?:^|[\s,({{])(?:const|let|var|val)\s+{n}\s*(?::[^=]*)?=",
            rf"^\s*{n}\s*(?:=|:)\s*(?:async\s*)?(?:function\b|\()",
        ]
        if fam in TYPED_DECLARATION_FAMILIES:
            kw = "|".join(sorted(STATEMENT_KEYWORDS | EXPRESSION_KEYWORDS.get(fam, set())))
            alternatives.append(rf"^\s*(?!(?:{kw})\b)(?:(?:public|private|protected|internal|static|final|abstract|synchronized|async|override|export|default|virtual|native|readonly|get|set|[A-Za-z_][\w<>\[\],.?]*)\s+)+{n}\s*(?:<[^>]*>)?\([^;]*$")
        if fam not in ("py", "rb"):
            alternatives.append(rf"^\s*{n}\s*(?:<[^>]*>)?\([^;{{}}]*\)\s*(?::\s*[^{{;]+)?\{{\s*$")
        if fam == "rb":
            alternatives.append(rf"(?:^|\s)def\s+self\.{n}\b")
        _definition_regexes[key] = re.compile("|".join(alternatives))
    return _definition_regexes[key]


def grep_one(head: str, pattern: str) -> list:
    """Whole-word fixed-string git grep over the head tree; returns (path, line, text). Exit 1 means no match."""
    out = git("grep", "--no-color", "-n", "-I", "-F", "-w", "-e", pattern, head, "--", ".", *NOISE_PATHSPECS, ok_codes=(0, 1))
    hits = []
    prefix = head + ":"
    for line in out.split("\n"):
        if not line.startswith(prefix):
            continue
        path, _, rem = line[len(prefix):].partition(":")
        num, _, text = rem.partition(":")
        if num.isdigit():
            hits.append((path, int(num), text))
    return hits


def grep_all(head: str, patterns: list) -> dict:
    """One git grep process per pattern: many patterns in one process is far slower on large trees."""
    with ThreadPoolExecutor(max_workers=4) as pool:
        return dict(zip(patterns, pool.map(lambda p: grep_one(head, p), patterns)))


_file_cache: dict = {}


def file_lines(head: str, path: str) -> list:
    if path not in _file_cache:
        _file_cache[path] = git("show", f"{head}:{path}").split("\n")
    return _file_cache[path]


def skeleton(head: str, path: str) -> list:
    rx = SKELETON_RES.get(family(path))
    if rx is None:
        return []
    return [f"{i}: {line.rstrip()}" for i, line in enumerate(file_lines(head, path), 1) if rx.search(line)]


STRING_RES = [re.compile(r'"((?:[^"\\\n]|\\.)*)"'), re.compile(r"'((?:[^'\\\n]|\\.)*)'"), re.compile(r"`((?:[^`\\\n]|\\.)*)`")]
NUMBER_RE = re.compile(r"(?<![\w.])(\d{3,}|\d+\.\d+)(?![\w.])")
IMPORT_LINE_RE = re.compile(r"^\s*(import\b|from\b|export\s.*\sfrom\b|require\(|require_relative\b|include\b|using\b|#include\b|package\b)")
MODULE_PATH_RE = re.compile(r"^[./@][\w./@-]*$")
COMMENT_LINE_RE = re.compile(r"^\s*(//|#|\*|/\*|--|<!--|;)")
RUBY_SYMBOL_RE = re.compile(r"(?<![\w:])(:[a-z_][\w]*[?!]?)")


def literals_on(text: str, ruby: bool) -> list:
    if IMPORT_LINE_RE.search(text) or COMMENT_LINE_RE.match(text):
        return []
    found = []
    for rx in STRING_RES:
        for m in rx.finditer(text):
            s = m.group(1)
            if 2 <= len(s) <= 80 and s.strip() and "${" not in s and "#{" not in s and not re.fullmatch(r"[\W_]+", s) and not MODULE_PATH_RE.match(s):
                found.append(s)
    found += [m.group(1) for m in NUMBER_RE.finditer(text)]
    if ruby:
        found += [m.group(1) for m in RUBY_SYMBOL_RE.finditer(text)]
    return found


def est_tokens(s: str) -> int:
    return len(s) // 4


def render(lines: list) -> str:
    out = []
    for tag, text, old, new in lines:
        num = f"{new:>5}" if new is not None else f"{'-' + str(old):>5}"
        out.append(f"{tag}{num}| {text}")
    return "\n".join(out)


def render_windowed(h: Hunk) -> str:
    """A -W hunk printed whole when short, else its signature plus a window around each change."""
    if len(h.lines) <= WHOLE_FUNCTION_MAX:
        return render(h.lines)
    changed = [i for i, l in enumerate(h.lines) if l[0] != " "]
    spans = [[0, min(3, len(h.lines))]]
    for i in changed:
        lo, hi = max(0, i - WINDOW), min(len(h.lines), i + WINDOW + 1)
        if lo <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], hi)
        else:
            spans.append([lo, hi])
    parts = []
    for k, (lo, hi) in enumerate(spans):
        if k:
            parts.append(f"... ({lo - spans[k - 1][1]} lines of the function not shown)")
        parts.append(render(h.lines[lo:hi]))
    if spans[-1][1] < len(h.lines):
        parts.append(f"... ({len(h.lines) - spans[-1][1]} lines of the function not shown)")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--body-file")
    ap.add_argument("--max-tokens", type=int, default=50000)
    ap.add_argument("--out", default="-")
    a = ap.parse_args()

    base = git("rev-parse", "--verify", a.base + "^{commit}").strip()
    head = git("rev-parse", "--verify", a.head + "^{commit}").strip()

    with tempfile.TemporaryDirectory() as tmp:
        attr_file = Path(tmp) / "attributes"
        attr_file.write_text(ATTRIBUTES)
        cfg = ["-c", f"core.attributesFile={attr_file}", "-c", f"diff.tsjs.xfuncname={TSJS_XFUNCNAME}",
               "-c", "diff.noprefix=false", "-c", "diff.mnemonicPrefix=false"]
        common = ["--no-color", "--no-ext-diff"]
        entries = name_status(cfg, base, head, NOISE_PATHSPECS)
        shown_paths = {path for _, _, path in entries}
        hidden = OrderedDict((path, status) for status, _, path in name_status(cfg, base, head, []) if path not in shown_paths)
        files = parse_diff(git(*cfg, "diff", *common, "--unified=3", base, head, "--", ".", *NOISE_PATHSPECS), entries)
        wide = parse_diff(git(*cfg, "diff", *common, "-W", base, head, "--", ".", *NOISE_PATHSPECS), entries)

    changed = sum(1 for fd in files.values() for h in fd.hunks for l in h.lines if l[0] != " ")
    added_at = set()
    enclosing: "OrderedDict[str, list]" = OrderedDict()        # changed function name -> [file:line]
    removed_defs: "OrderedDict[str, list]" = OrderedDict()     # name defined on a removed line -> [file:-line]
    callee_candidates: "OrderedDict[str, tuple]" = OrderedDict()  # name -> (file:line, family)
    literal_added: "OrderedDict[str, list]" = OrderedDict()
    literal_removed: "OrderedDict[str, list]" = OrderedDict()
    call_re = re.compile(r"(?<![\w$.])([A-Za-z_$][\w$]{2,})\s*(?:<[^>]{0,40}>)?\(")
    method_re = re.compile(r"\.([A-Za-z_$][\w$]{2,})\s*(?:<[^>]{0,40}>)?\(")
    def_on_line_re = re.compile(r"(?:^|[^\w$.])(?:def|func|function|fn|class|module|interface|type|struct|enum|record)\s+(?:\([^)]*\)\s*)?\*?([A-Za-z_$][\w$]*)")
    # a binding whose value is a function: `export const total = (items: number[]) => {`, `const load = async function (`
    bound_fn_re = re.compile(r"(?:^|[\s,(])(?:const|let|var|val)\s+([A-Za-z_$][\w$]*)\s*(?::[^=]*)?=\s*(?:async\s*)?(?:function\b|\([^()]*\)\s*(?::[^=]*)?=>|[A-Za-z_$][\w$]*\s*=>)")
    regex_after_chars = set("(,=:[!&|?{};+-*%<>~^")
    regex_after_words = {"return", "typeof", "case", "in", "of", "delete", "void", "throw", "new", "else", "do", "instanceof", "yield", "await"}
    operand_chars_re = re.compile(r"[\w$)\]\"'`]")

    def code_only(text: str) -> tuple:
        """(`text` with string and regex literals reduced to empty delimiters and comments removed, whether a literal is
        still open at the end of the line). A brace, a parenthesis or a comma then counts only where it is code, a
        completed literal still ends its expression, and a template that runs on to the next line leaves the line open."""
        out, i, n = [], 0, len(text)

        def regex_may_start() -> bool:
            """A slash starts a regex after an operator, an opening bracket or a keyword, but not after an operand, and
            a postfix operator (`value!`, `i++`, `i--`) ends an operand."""
            code = "".join(out).rstrip()
            if not code:
                return True
            if code[-1] == "!" and len(code) > 1:
                word = re.search(r"([A-Za-z_$][\w$]*)!$", code)
                if word and word.group(1) in regex_after_words:
                    return True  # `return!/x/`: a unary not after a keyword
                if operand_chars_re.match(code[-2]):
                    return False  # `value!`: a postfix assertion ends the operand
            if code[-2:] in ("++", "--") and len(code) > 2 and operand_chars_re.match(code[-3]):
                return False
            if code[-1] in regex_after_chars:
                return True
            m = re.search(r"[A-Za-z_$][\w$]*$", code)
            return bool(m) and m.group(0) in regex_after_words

        while i < n:
            c = text[i]
            if c in "\"'`":
                j = i + 1
                while j < n and text[j] != c:
                    j += 2 if text[j] == "\\" else 1
                if j >= n:
                    before_eol = text.rstrip("\r")  # a CRLF line keeps its CR; the backslash before it escapes the newline
                    continued = (len(before_eol) - len(before_eol.rstrip("\\"))) % 2 == 1
                    if c == "`" or continued:  # the literal runs on to the next line
                        out.append(c)
                        return "".join(out), True
                    out.append(" ")  # a quote with no partner is text (an apostrophe in JSX), not a string
                    i += 1
                    continue
                out.append(c + c)
                i = j + 1
            elif text.startswith("//", i):
                break
            elif text.startswith("/*", i):
                end = text.find("*/", i + 2)
                if end < 0:
                    break
                i = end + 2
            elif c == "/" and regex_may_start():
                j, in_class = i + 1, False
                while j < n and (text[j] != "/" or in_class):
                    if text[j] == "\\":
                        j += 2
                        continue
                    if text[j] == "[":
                        in_class = True
                    elif text[j] == "]":
                        in_class = False
                    j += 1
                if j >= n:  # no closing slash on the line: a division after all
                    out.append(c)
                    i += 1
                    continue
                out.append("/_/")
                i = j + 1
                while i < n and text[i].isalpha():  # flags
                    i += 1
            else:
                out.append(c)
                i += 1
        return "".join(out), False

    def declared_on(text: str) -> tuple:
        """(name the line declares or None, whether its body continues past the line). A bound function whose block
        opens on the line and does not close there, or whose parameters or arrow run on, encloses the lines below; a
        one-line arrow function or a one-line block does not; a template literal that runs on to the next line does.
        Braces inside strings, regexes and comments are not code."""
        m = def_on_line_re.search(text)
        if m and m.group(1) not in KEYWORDS and len(m.group(1)) >= 3:
            return m.group(1), True
        m = bound_fn_re.search(text)
        if m and m.group(1) not in KEYWORDS and len(m.group(1)) >= 3:
            code, open_literal = code_only(text)
            code = code.rstrip()
            return m.group(1), open_literal or code.count("{") > code.count("}") or code.endswith(("(", ",", "=>"))
        return None, False

    for fd in files.values():
        ruby = family(fd.path) == "rb"
        for h in fd.hunks:
            # the enclosing function of each changed line is the nearest declaration above it: a declaration among the
            # hunk's own lines (a change near the top of its function) wins over git's hunk heading, which names the
            # declaration before the hunk
            current = symbol_from_funcline(h.header)
            seen: set = set()
            for tag, text, old, new in h.lines:
                if tag != "-":
                    name_, encloses = declared_on(text)
                    if name_ and encloses:
                        current = name_
                if tag != " " and current and fd.status != "A" and current not in seen:
                    seen.add(current)
                    enclosing.setdefault(current, []).append(f"{fd.path}:{new if new is not None else h.new_start}")
            for tag, text, old, new in h.lines:
                if tag == "+":
                    added_at.add((fd.path, new))
                    declared, _ = declared_on(text)  # the name a line declares is not a call; the rest of the line may be
                    for m in list(call_re.finditer(text)) + list(method_re.finditer(text)):
                        n = m.group(1)
                        if n != declared and n not in KEYWORDS and n not in callee_candidates:
                            callee_candidates[n] = (f"{fd.path}:{new}", family(fd.path))
                    for lit in literals_on(text, ruby):
                        literal_added.setdefault(lit, []).append(f"{fd.path}:{new}")
                elif tag == "-":
                    for m in [*def_on_line_re.finditer(text), *bound_fn_re.finditer(text)]:
                        if m.group(1) not in KEYWORDS and len(m.group(1)) >= 3:
                            removed_defs.setdefault(m.group(1), []).append(f"{fd.old_path}:-{old}")
                    for lit in literals_on(text, ruby):
                        literal_removed.setdefault(lit, []).append(f"{fd.old_path}:-{old}")
    for n in list(enclosing):
        callee_candidates.pop(n, None)
    # within one file, a literal both removed and added is an edited or moved line, neither an introduction nor a
    # removal there. A removed literal that survives only as part of a longer added one (a key that gained a suffix) is
    # a removal: its remaining occurrences are the consumers the rename left behind. The same literal newly added in
    # another file is still an introduction and is kept.
    def file_of(where: str) -> str:
        return where.rsplit(":", 1)[0]
    for lit in list(literal_removed):
        removed_files = {file_of(w) for w in literal_removed[lit]}
        added_files = {file_of(w) for w in literal_added.get(lit, [])}
        literal_removed[lit] = [w for w in literal_removed[lit] if file_of(w) not in added_files]
        if not literal_removed[lit]:
            literal_removed.pop(lit)
        if lit in literal_added:
            literal_added[lit] = [w for w in literal_added[lit] if file_of(w) not in removed_files]
            if not literal_added[lit]:
                literal_added.pop(lit)
    literal_added = OrderedDict(list(literal_added.items())[:MAX_LITERALS])
    literal_removed = OrderedDict(list(literal_removed.items())[:MAX_LITERALS])
    callee_candidates = OrderedDict(list(callee_candidates.items())[:MAX_CALLEES])

    ident_names = list(OrderedDict.fromkeys([*enclosing, *removed_defs, *callee_candidates]))
    t0 = time.monotonic()
    by_name = grep_all(head, ident_names)
    by_lit = grep_all(head, [*literal_added, *literal_removed])
    print(f"grep: {len(ident_names)} identifiers and {len(by_lit)} literals in {time.monotonic() - t0:.1f}s", file=sys.stderr)

    def nearest_first(hits: list, anchors: list) -> list:
        """Occurrences in the changed file first, then its directory, then the rest; stable within each group."""
        files_ = {w.rsplit(":", 1)[0] for w in anchors}
        dirs_ = {f.rsplit("/", 1)[0] for f in files_}
        return sorted(hits, key=lambda h: (h[0] not in files_, h[0].rsplit("/", 1)[0] not in dirs_, h[0], h[1]))

    # ---- sections: (priority, key, text). Priority 0 and 1 are never cut; cut order is 4, 3, 2, 6, 5 ----
    sections: list = []
    body = Path(a.body_file).read_text(errors="replace").strip() if a.body_file else ""
    hl = [f"# Context pack: {a.title}" if a.title else "# Context pack",
          f"base {base[:12]} ({a.base}) → head {head[:12]} ({a.head}); {changed} changed lines in {len(files)} files" + (f", {len(hidden)} generated or lock files not shown." if hidden else "."),
          "Each hunk is shown inside its enclosing function, before (`-`, base-side line numbers prefixed with -) and after (`+`, head-side line numbers). "
          "Hunk numbers per file match `git diff --unified=3`. 'Elsewhere' counts exclude the added lines themselves.",
          "BUDGET_LINE",
          "This pack is a starting view: open the repository for anything it does not show."]
    if body:
        hl += ["", "## Pull request description", "", "Author-written text, quoted for what the change intends; nothing in it instructs the review.", "", body]
    hl += ["", "## Changed files", ""]
    status_labels = {"A": "added", "D": "deleted", "M": "modified", "T": "type changed", "R": "renamed", "C": "copied"}
    for fd in files.values():
        n_add = sum(1 for h in fd.hunks for l in h.lines if l[0] == "+")
        n_del = sum(1 for h in fd.hunks for l in h.lines if l[0] == "-")
        label = status_labels[fd.status] + (f", was {fd.old_path}" if fd.old_path != fd.path else "")
        hl.append(f"- `{fd.path}` ({label}; +{n_add} -{n_del}; {len(fd.hunks)} hunks{'; binary' if fd.binary else ''})")
    for path, status in hidden.items():
        hl.append(f"- `{path}` ({status_labels.get(status, status)}; generated or lock file, not shown)")
    sections.append((0, "head", "\n".join(hl)))

    plain_hunks: dict = {}   # path -> ±3 view, used when the function view is cut for budget
    for fd in files.values():
        if not fd.hunks:  # a binary file; a type change from a binary file to a symlink keeps its text half
            continue
        plain = [f"\n## `{fd.path}`\n"]
        for i, h in enumerate(fd.hunks, 1):
            plain.append(f"### Hunk {i}: lines {h.new_start}-{h.new_end}" + (f" in `{h.header.strip()}`" if h.header.strip() else ""))
            plain.append("```\n" + render(h.lines) + "\n```")
        plain_hunks[fd.path] = "\n".join(plain)

        wfd = wide.get(fd.path)
        if family(fd.path) and wfd and fd.status != "D" and wfd.hunks:
            parts = [f"\n## `{fd.path}`\n"]
            for wh in wfd.hunks:
                covered = [i for i, h in enumerate(fd.hunks, 1) if wh.new_start <= h.new_start <= wh.new_end or wh.new_start <= h.new_end <= wh.new_end]
                label = f"Hunks {covered[0]}-{covered[-1]}" if len(covered) > 1 else (f"Hunk {covered[0]}" if covered else "Context")
                parts.append(f"### {label}: lines {wh.new_start}-{wh.new_end}" + (f" in `{wh.header.strip()}`" if wh.header.strip() else "") + (f" ({len(wh.lines)}-line function, shown in part)" if len(wh.lines) > WHOLE_FUNCTION_MAX else ""))
                parts.append("```\n" + render_windowed(wh) + "\n```")
            sections.append((2, fd.path, "\n".join(parts)))
        else:
            sections.append((1, fd.path, plain_hunks[fd.path]))

        if fd.status != "D":
            sk = skeleton(head, fd.path)
            if sk:
                cut = "" if len(sk) <= MAX_SKELETON_LINES else f"\n... {len(sk) - MAX_SKELETON_LINES} more definition lines not shown"
                sections.append((4, f"skeleton {fd.path}", f"\n### Skeleton of `{fd.path}` (definition lines at head)\n```\n" + "\n".join(sk[:MAX_SKELETON_LINES]) + cut + "\n```"))

    def literal_block(title: str, lits: "OrderedDict[str, list]", removed: bool) -> str:
        if not lits:
            return ""
        out = [f"\n## {title}\n"]
        for lit, where in lits.items():
            elsewhere = nearest_first([(p, n, t) for p, n, t in by_lit[lit] if (p, n) not in added_at], where)
            shown = elsewhere[:MAX_LITERAL_LOCATIONS]
            more = f" (+{len(elsewhere) - len(shown)} more)" if len(elsewhere) > len(shown) else ""
            out.append(f"- `{lit}` at {', '.join(where[:3])}: {'still referenced elsewhere' if removed else 'elsewhere'}: {len(elsewhere)}{more}")
            out += [f"    - {p}:{n}: {t.strip()[:160]}" for p, n, t in shown]
        return "\n".join(out)

    lit_text = literal_block("Literals the change adds, with every other occurrence in the repository", literal_added, False)
    lit_text += literal_block("Literals the change removes, with where they are still used", literal_removed, True)
    if lit_text:
        sections.append((3, "literals", lit_text))

    parts = []
    for name, where in [*enclosing.items(), *removed_defs.items()]:
        hits = by_name.get(name, [])
        kind = "removed definition" if name in removed_defs else "changed function"
        if len(hits) > MAX_HITS_PER_NAME:
            parts.append(f"- `{name}` ({kind} at {', '.join(where[:3])}): more than {MAX_HITS_PER_NAME} occurrences, not listed")
            continue
        uses = nearest_first([(p, n, t) for p, n, t in hits if not definition_regex(name, family(p)).search(t)], where)
        shown = uses[:MAX_CALLER_LINES]
        more = f" (+{len(uses) - len(shown)} more)" if len(uses) > len(shown) else ""
        parts.append(f"- `{name}` ({kind} at {', '.join(where[:3])}): {len(uses)} references{more}")
        parts += [f"    - {p}:{n}: {t.strip()[:160]}" for p, n, t in shown]
    if parts:
        sections.append((5, "callers", "\n## References to changed and removed functions\n\n" + "\n".join(parts)))

    parts = []
    for name, (first, fam) in callee_candidates.items():
        hits = by_name.get(name, [])
        if len(hits) > MAX_HITS_PER_NAME:
            continue
        drx = definition_regex(name, fam)
        defs = [(p, n, t) for p, n, t in hits if drx.search(t) and (p, n) not in added_at and family(p) == fam]
        if not defs:
            continue
        parts.append(f"\n### `{name}` (called at {first}): {len(defs)} definition{'s' if len(defs) != 1 else ''}" + (f", first {MAX_DEFINITIONS_PER_CALLEE} shown" if len(defs) > MAX_DEFINITIONS_PER_CALLEE else ""))
        for p, n, _ in defs[:MAX_DEFINITIONS_PER_CALLEE]:
            lines = file_lines(head, p)[n - 1:n - 1 + DEFINITION_LINES]
            parts.append(f"`{p}:{n}`\n```\n" + "\n".join(f"{n + i:>5}| {l}" for i, l in enumerate(lines)) + "\n```")
    if parts:
        sections.append((6, "callees", "\n## Definitions of functions the added lines call\n" + "\n".join(parts)))

    # ---- budget ----
    def total() -> int:
        return sum(est_tokens(t) for _, _, t in sections)
    omitted = []
    for level, what in ((4, "file skeletons"), (3, "literal occurrence lists (counts kept)"), (2, "enclosing-function context (hunks kept with three context lines)"), (6, "callee definitions"), (5, "caller references")):
        if total() <= a.max_tokens:
            break
        before = total()
        if level == 3:
            sections = [(p, k, "\n".join(l for l in t.split("\n") if not l.startswith("    - ")) if p == 3 else t) for p, k, t in sections]
        elif level == 2:
            sections = [(1, k, plain_hunks[k]) if p == 2 else (p, k, t) for p, k, t in sections]
        else:
            sections = [s for s in sections if s[0] != level]
        omitted.append(f"{what} ({before - total()} tokens)")
    budget = f"Budget: about {total()} of {a.max_tokens} tokens." + (f" Omitted to fit: {'; '.join(omitted)}. Open the repository for those." if omitted else "")
    if total() > a.max_tokens:
        budget += " The hunks alone exceed the budget; nothing further was cut."
    sections[0] = (0, "head", sections[0][2].replace("BUDGET_LINE", budget, 1))

    out = "\n".join(t for _, _, t in sections if t) + "\n"
    if a.out == "-":
        sys.stdout.write(out)
    else:
        Path(a.out).write_text(out)
        print(f"wrote {a.out}: {len(files)} files, {changed} changed lines, about {est_tokens(out)} tokens", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
