#!/usr/bin/env python
"""Check documentation coverage of a Python package.

The tool inspects the public API (via __all__ exports) of a given package using
``griffe``, filters for objects that carry a docstring, and then scans the
markdown documentation tree to see whether those objects are mentioned.  Mentions
are detected by looking at

* inline code spans / markdown headings (`` `Foo` `` or ``## Foo``)
* explicit ``[[autodoc]]`` blocks (HF doc-builder syntax)

Usage::

    python scripts/check_doc_coverage.py --package peft --src src --docs docs/source

The command exits with code 0 and prints a coverage table.  By default the tool
outputs a short summary; pass ``--verbose`` to see every detected / missing
object.
"""

import argparse
import re
import sys
from collections.abc import Iterable
from pathlib import Path

import griffe
from griffe import Alias, AliasResolutionError, Class, Function, Module


# ---------------------------------------------------------------------------
# Constant regexes
# ---------------------------------------------------------------------------

# hf-doc-builder autodoc blocks: [[autodoc]] path.to.ClassOrFunction
RE_AUTODOC = re.compile(r"\[\[autodoc\]\]\s+(\S+)")

# Inline code span `foo` or ``foo``  (or longer runs)
RE_INLINE_CODE = re.compile(r"`{1,2}([^`\s]+)`{1,2}")

# Markdown heading text (we strip the hashes)
RE_HEADING = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)

# Fenced code blocks (language tag is optional)
RE_CODE_BLOCK = re.compile(r"```[\w]*\n(.*?)```", re.DOTALL)


# ---------------------------------------------------------------------------
# Griffe helpers
# ---------------------------------------------------------------------------


def resolve_alias(obj: Alias | Module | Class | Function) -> Alias | Module | Class | Function | None:
    """Follow a chain of aliases until a concrete object is reached."""
    seen: set[int] = set()
    while isinstance(obj, Alias):
        obj_id = id(obj)
        if obj_id in seen:
            return None
        seen.add(obj_id)
        try:
            obj = obj.target
        except AliasResolutionError:
            return None
    return obj


def extract_public_api_items(
    package_name: str,
    src_path: str | None = None,
    recursive: bool = False,
) -> dict[str, str]:
    """Return a mapping *exported_short_name* -> *canonical_path*.

    The set is derived from ``__all__`` lists.  By default only the root
    package's ``__all__`` is inspected; enable *recursive* to also inspect
    every submodule.

    Only objects that actually carry a docstring are kept.
    """
    search_paths = [src_path] if src_path else ["."]
    package: Module = griffe.load(package_name, search_paths=search_paths)

    items: dict[str, str] = {}
    modules = walk_modules(package) if recursive else [package]
    for module in modules:
        exports = resolve_all(module)
        if not exports:
            continue
        for name in exports:
            obj = module.members.get(name)
            if obj is None:
                continue
            obj = resolve_alias(obj)
            if obj is None:
                continue
            if obj.docstring is None or not obj.docstring.value.strip():
                continue
            if isinstance(obj, (Module, Class, Function)):
                items[name] = obj.path
    return items


def walk_modules(package: Module, package_name: str | None = None) -> Iterable[Module]:
    """Yield *package* itself and every submodule recursively."""
    if package_name is None:
        package_name = package.name
    yield package
    for member in package.members.values():
        if isinstance(member, Module):
            yield from walk_modules(member, package_name)
        elif isinstance(member, Alias):
            try:
                target = member.target
            except AliasResolutionError:
                continue
            if isinstance(target, Module) and target.path.startswith(package_name + "."):
                yield from walk_modules(target, package_name)


def resolve_all(module: Module) -> list[str]:
    """Return the names from ``__all__`` for *module*, or empty list.

    Griffe stores ``__all__`` as an Attribute whose ``value`` is an
    ``ExprList`` (or similar expression node).  We convert it to a string and
    use ``ast.literal_eval`` to recover the plain Python list of names.
    """
    attr = module.members.get("__all__")
    if attr is None:
        return []
    val = getattr(attr, "value", None)
    if val is None:
        return []
    import ast

    try:
        parsed = ast.literal_eval(str(val))
        if isinstance(parsed, (list, tuple)):
            return [str(v) for v in parsed]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Doc scanning helpers
# ---------------------------------------------------------------------------


def _add_mention(raw: str, into: set[str]) -> None:
    """Normalise a raw mention and add its token(s) to *into*."""
    raw = raw.lstrip("~")
    # strip trailing call/index syntax, e.g. `foo(...)` or `foo[...]`
    raw = re.sub(r"[\(\[].*?[\)\]]$", "", raw)
    for part in raw.split("."):
        part = part.strip()
        if part:
            into.add(part)


def extract_doc_mentions(docs_dir: str) -> set[str]:
    """Walk every ``*.md`` under *docs_dir* and return the set of names that
    are referenced either inline or via ``[[autodoc]]``."""
    mentions: set[str] = set()
    root = Path(docs_dir)
    for path in root.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        # 1. Autodoc blocks – the token after [[autodoc]] is a dotted path.
        for match in RE_AUTODOC.finditer(text):
            _add_mention(match.group(1), mentions)
        # 2. Inline code spans
        for match in RE_INLINE_CODE.finditer(text):
            _add_mention(match.group(1), mentions)
        # 3. Headings
        for match in RE_HEADING.finditer(text):
            heading = match.group(1)
            for m in RE_INLINE_CODE.finditer(heading):
                _add_mention(m.group(1), mentions)
            for word in re.findall(r"[A-Za-z_]\w*", heading):
                _add_mention(word, mentions)
        # 4. Fenced code blocks – take every identifier-like token.
        for block in RE_CODE_BLOCK.finditer(text):
            for word in re.findall(r"[A-Za-z_]\w*", block.group(1)):
                _add_mention(word, mentions)
    return mentions


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(
    api_items: dict[str, str],
    mentions: set[str],
    verbose: bool = False,
) -> None:
    covered: list[str] = []
    uncovered: list[str] = []

    for short_name in api_items:
        if short_name in mentions:
            covered.append(short_name)
        else:
            uncovered.append(short_name)

    total = len(api_items)
    covered_n = len(covered)
    pct = (covered_n / total * 100) if total else 0.0

    if verbose:
        print("Covered API items:")
        for name in sorted(covered):
            print(f"  - {name} ({api_items[name]})")
        print()
        print("Missing API items:")
        for name in sorted(uncovered):
            print(f"  - {name} ({api_items[name]})")
        print()

    print(f"Documentation coverage: {covered_n}/{total} ({pct:.1f}%)")
    if uncovered:
        print(f"Missing ({len(uncovered)}):")
        for name in sorted(uncovered)[:20]:
            print(f"  - {name}")
        if len(uncovered) > 20:
            print(f"  ... and {len(uncovered) - 20} more")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="peft", help="Package name to inspect.")
    parser.add_argument("--src", default="src", help="Path to the source tree root (containing the package dir).")
    parser.add_argument("--docs", default="docs/source", help="Path to the markdown docs directory.")
    parser.add_argument(
        "--recursive", action="store_true", help="Inspect __all__ in every submodule, not just the root package."
    )
    parser.add_argument("--verbose", action="store_true", help="Print every covered/missing item.")
    args = parser.parse_args(argv)

    if not Path(args.src).is_dir():
        print(f"Error: source path not found: {args.src}", file=sys.stderr)
        return 1
    if not Path(args.docs).is_dir():
        print(f"Error: docs path not found: {args.docs}", file=sys.stderr)
        return 1

    print(f"Loading package '{args.package}' from {args.src} ...")
    api_items = extract_public_api_items(args.package, args.src, recursive=args.recursive)
    print(f"Found {len(api_items)} public objects with docstrings.")

    print(f"Scanning docs in {args.docs} ...")
    mentions = extract_doc_mentions(args.docs)
    print(f"Found {len(mentions)} unique name mentions in docs.")

    print()
    print_report(api_items, mentions, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
