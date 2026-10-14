# review-tools

Generic tooling used by Aether's pull-request reviews on public benchmarks. Everything here is structural and
judgment-free: it contains no bug rules, no knowledge of any benchmark, and it decides nothing. Review prompts pin a
commit of this repository so that the exact tool a reported run used can be read.

## context_pack.py

Builds one markdown file, the context pack, that every agent in a review reads before touching the repository. For a
change from `--base` (the merge base) to `--head` it prints:

- every hunk inside its enclosing function, before and after the change, using git's function-context diff with
  language-aware diff drivers (git's builtin drivers plus one for JavaScript and TypeScript); hunks are numbered per
  file exactly as `git diff --unified=3 <base> <head>` numbers them;
- each changed file's definition skeleton (class, function and method signature lines at the head);
- the references to every changed or removed function, nearest to the change first;
- the definitions of the functions the added lines call, from files of the same language;
- every string, number and symbol literal the change adds or removes, with the count and locations of its other
  occurrences in the repository, nearest first. A literal that is removed and added in the same file is an edit and is
  not listed.

The output is deterministic for a given base, head and token budget (`--max-tokens`, default 50,000, estimated at four
characters per token). When the budget is exceeded, sections are dropped in a fixed order (skeletons, then literal
locations, then function context, then callee definitions, then callers) and the header says what was omitted.

```
python3 context_pack.py --base "$(git merge-base origin/main HEAD)" --head HEAD \
    --title "<pull request title>" --body-file pr_body.md --out pack.md
```

Requires git and Python 3.9 or later; nothing else. Tests: `python3 -m unittest test_context_pack` (builds small two-commit repositories). Generated catalogs and snapshots (`*.po`, `*.pot`, `*.pysnap`,
`snapshots/`), lockfiles, minified bundles, source maps and `node_modules`/`vendor`/`dist`/`build` trees are excluded
from occurrence searches.
