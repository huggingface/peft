#!/usr/bin/env python
"""Check documentation coverage of a Python package.

The tool inspects the public API (objects exported via `__all__`) of a given
package, filters for the ones that carry a docstring, and then scans the
markdown documentation tree to see whether those objects are mentioned.  Mentions
are detected by looking at

* inline code spans / markdown headings (` `Foo` ` or `## Foo`)
* explicit `[[autodoc]]` blocks (HF doc-builder syntax)
* identifier tokens inside fenced code blocks

Public API resolution
---------------------

By default the `__all__` of each module is read *statically* with `griffe`
(neither the package nor its dependencies are imported).  This works for
packages that define `__all__` as a plain list at import time, e.g. PEFT.

Some packages — notably `transformers`, whose top-level `__init__` builds
`__all__` at *runtime* via a lazy `_LazyModule` — expose no static
`__all__`.  For those, pass `--inspect`: the package is imported once purely
to read its runtime `__all__` (a cheap operation that does not pull in the
submodules), and every other piece of information (docstrings, canonical paths)
is still obtained statically from griffe.  Per-object failures while walking the
static tree are tolerated and reported rather than aborting the whole run.

Usage::

    # PEFT (static __all__)
    python scripts/check_doc_coverage.py --package peft --src src --docs docs/source

    # transformers (runtime __all__, lives in doc_check/transformers/src)
    python scripts/check_doc_coverage.py --package transformers \\
        --src doc_check/transformers/src --docs doc_check/transformers/docs/source/en \\
        --inspect

Path-based wildcard excludes (--exclude, repeatable) drop matched objects from
the public API so they count as neither covered nor uncovered, e.g. to ignore
the per-model implementations under transformers' `models` package::

    python scripts/check_doc_coverage.py --package transformers \\
        --src doc_check/transformers/src --docs doc_check/transformers/docs/source/en \\
        --inspect --exclude 'transformers.models.*'

The command exits with code 0 and prints a coverage summary.  Pass `--verbose`
to see every covered / missing object, plus any names that could not be resolved
to a griffe object.
"""

import argparse
import fnmatch
import importlib
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import griffe
from griffe import (
    Alias,
    AliasResolutionError,
    Class,
    CyclicAliasError,
    Function,
    LoadingError,
    Module,
    NameResolutionError,
    UnimportableModuleError,
)


# ---------------------------------------------------------------------------
# Constant regexes
# ---------------------------------------------------------------------------

# hf-doc-builder autodoc blocks: [[autodoc]] path.to.ClassOrFunction
RE_AUTODOC = re.compile(r"\[\[autodoc\]\]\s+(\S+)")

# Inline code span `foo` or `foo` (or longer runs)
RE_INLINE_CODE = re.compile(r"`{1,2}([^`\s]+)`{1,2}")

# Markdown heading text (we strip the hashes)
RE_HEADING = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)

# Fenced code blocks (language tag is optional)
RE_CODE_BLOCK = re.compile(r"```[\w]*\n(.*?)```", re.DOTALL)


# ---------------------------------------------------------------------------
# Griffe helpers
# ---------------------------------------------------------------------------

# Griffe exception types we tolerate while traversing the static tree.  Any of
# these raised for a single member is reported and skipped, never fatal.
_GRIFFE_ERRORS: tuple[type[BaseException], ...] = (
    AliasResolutionError,
    CyclicAliasError,
    LoadingError,
    NameResolutionError,
    UnimportableModuleError,
)


def _is_excluded(path: str, excludes: tuple[str, ...]) -> bool:
    """True if canonical dotted *path* matches any `--exclude` pattern.

    Matching uses `fnmatch`, whose `*` spans dots, so `transformers.models.*`
    matches every object defined under that package.  An empty *excludes* tuple
    excludes nothing (the common `all(fnmatchcase(...)) over ()` returns False).
    """
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in excludes)


def resolve_alias(obj: Alias | Module | Class | Function) -> Alias | Module | Class | Function | None:
    """Follow a chain of aliases until a concrete object is reached.

    Returns `None` if the chain cannot be resolved (missing target, cycle,
    or any other griffe alias-resolution error).
    """
    seen: set[int] = set()
    while isinstance(obj, Alias):
        obj_id = id(obj)
        if obj_id in seen:
            return None
        seen.add(obj_id)
        try:
            obj = obj.target
        except (AliasResolutionError, CyclicAliasError):
            return None
    return obj


def _safe_docstring(obj: Module | Class | Function) -> str:
    """Return the stripped docstring of *obj*, or "" on any resolution error.

    Accessing `obj.docstring` can trigger alias/target resolution that raises
    for partially-loaded trees; we never want a single unreadable object to
    abort the whole run.
    """
    try:
        ds = obj.docstring
    except _GRIFFE_ERRORS:
        return ""
    return ds.value.strip() if ds is not None else ""


def _load_package(package_name: str, src_path: str | None) -> Module:
    """Statically load *package_name* with griffe (no runtime import).

    `src_path` is honoured: it is put on griffe's search path so the checkout
    is loaded rather than whatever happens to be installed.
    """
    search_paths = [src_path] if src_path else ["."]
    package: Module = griffe.load(package_name, search_paths=search_paths, submodules=True)
    return package


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
            except (AliasResolutionError, CyclicAliasError):
                continue
            if isinstance(target, Module) and target.path.startswith(package_name + "."):
                yield from walk_modules(target, package_name)


# ---------------------------------------------------------------------------
# Report container
# ---------------------------------------------------------------------------


@dataclass
class Diagnostics:
    """Collected while resolving the public API; printed at the end."""

    # `__all__` entries (or runtime names) we could not map to a griffe object
    unresolved: list[str] = field(default_factory=list)
    # short names dropped because several distinct objects shared them
    collisions: list[tuple[str, str, str]] = field(default_factory=list)  # (name, kept, dropped)
    # number of objects dropped via `--exclude` (neither covered nor uncovered)
    excluded: int = 0
    # reason the dynamic import (`--inspect`) failed, if it did
    import_error: str | None = None


# ---------------------------------------------------------------------------
# Static public-API resolution
# ---------------------------------------------------------------------------


def _items_from_exports(
    modules: Iterable[Module], diagnostics: Diagnostics, excludes: tuple[str, ...] = ()
) -> dict[str, str]:
    """Public items from the static `__all__` of each module in *modules*.

    Uses griffe's `module.exports` (which reflects `__all__`).  Only
    Class/Function/Module objects with a non-empty docstring are kept.  Objects
    whose canonical path matches an `--exclude` pattern are dropped here.
    """
    items: dict[str, str] = {}
    for module in modules:
        exports = module.exports
        if not exports:
            continue
        for name in exports:
            member = module.members.get(name)
            if member is None:
                diagnostics.unresolved.append(f"{module.path}.__all__ -> {name}")
                continue
            if name.startswith("_"):
                continue
            obj = resolve_alias(member)
            if obj is None:
                diagnostics.unresolved.append(f"{module.path}.__all__ -> {name}")
                continue
            if not isinstance(obj, (Class, Function, Module)):
                continue
            if not _safe_docstring(obj):
                continue
            _record(items, name, obj, diagnostics, excludes)
    return items


def _record(
    items: dict[str, str],
    short_name: str,
    obj: Module | Class | Function,
    diagnostics: Diagnostics,
    excludes: tuple[str, ...] = (),
) -> None:
    """Insert *short_name* -> *obj.path*, recording collisions without overwriting.

    `--exclude` matches are dropped (counted in `diagnostics.excluded`)
    rather than entered as covered or uncovered.
    """
    if _is_excluded(obj.path, excludes):
        diagnostics.excluded += 1
        return
    existing = items.get(short_name)
    if existing is not None and existing != obj.path:
        diagnostics.collisions.append((short_name, existing, obj.path))
        return
    if existing is None:
        items[short_name] = obj.path


# ---------------------------------------------------------------------------
# `--inspect`: dynamic __all__, static everything else
# ---------------------------------------------------------------------------


def _dynamic_all(package_name: str, diagnostics: Diagnostics) -> list[str] | None:
    """Import *package_name* once and return its runtime `__all__`.

    Reading `__all__` is cheap and, crucially, does not trigger the import of
    every submodule (transformers' `__all__` is built by `_LazyModule` at
    `__init__` time).  Returns `None` if the import itself failed.
    """
    try:
        mod = importlib.import_module(package_name)
    except Exception as exc:
        diagnostics.import_error = f"could not import '{package_name}': {type(exc).__name__}: {exc}"
        return None
    all_ = getattr(mod, "__all__", None)
    if all_ is None:
        # Fall back to public top-level names; static griffe will still filter
        # these down to documented Class/Function/Module objects.
        all_ = [n for n in dir(mod) if not n.startswith("_")]
    return list(all_)


def _build_short_name_index(
    package: Module, diagnostics: Diagnostics, excludes: tuple[str, ...] = ()
) -> dict[str, Module | Class | Function]:
    """Walk the whole static tree once, mapping `object.name -> object`.

    Used by `--inspect` to resolve the short names from the runtime
    `__all__` to griffe objects (and thus their docstrings / canonical paths).
    Only modules and their *direct* members are indexed: recursing into class
    bodies would surface method names, which are not part of the public API and
    would cause spurious matches.

    Excluded objects are *kept in the index* so that a later lookup in
    `_items_from_index` still resolves them (and hands them to `_record`, which
    counts them as excluded); we only skip *collision reporting* for them, so an
    `--exclude 'transformers.models.*'` run is not flooded with thousands of
    per-model `*Config`/`*Model` collision entries.
    """
    index: dict[str, Module | Class | Function] = {}
    stack: list[Module] = [package]
    seen: set[str] = set()
    while stack:
        module = stack.pop()
        if module.path in seen:
            continue
        seen.add(module.path)
        for member in module.members.values():
            try:
                obj = resolve_alias(member) if isinstance(member, Alias) else member
            except _GRIFFE_ERRORS:
                continue
            if obj is None:
                continue
            if isinstance(obj, Module):
                stack.append(obj)
            if not isinstance(obj, (Class, Function, Module)):
                continue
            short = obj.name
            existing = index.get(short)
            if existing is not None and existing.path != obj.path:
                if not _is_excluded(existing.path, excludes) and not _is_excluded(obj.path, excludes):
                    diagnostics.collisions.append((short, existing.path, obj.path))
            elif existing is None:
                index[short] = obj
    return index


def _items_from_index(
    names: Iterable[str],
    index: dict[str, Module | Class | Function],
    diagnostics: Diagnostics,
    excludes: tuple[str, ...] = (),
) -> dict[str, str]:
    """Map runtime `__all__` names to griffe objects via the short-name index.

    `__all__` entries may be dotted (e.g. transformers lists `"models.bert"`);
    the *last* segment is the object's own short name and is what the docs and
    this tool match on.  `--exclude` filtering happens in `_record`.
    """
    items: dict[str, str] = {}
    for raw in names:
        short = raw.rsplit(".", 1)[-1]
        obj = index.get(short)
        if obj is None:
            diagnostics.unresolved.append(raw)
            continue
        if not _safe_docstring(obj):
            continue
        _record(items, short, obj, diagnostics, excludes)
    return items


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_public_api_items(
    package_name: str,
    src_path: str | None = None,
    recursive: bool = False,
    inspect: bool = False,
    excludes: tuple[str, ...] = (),
    diagnostics: Diagnostics | None = None,
) -> dict[str, str]:
    """Return a mapping *exported_short_name -> canonical_path* for documented objects.

    Parameters
    ----------
    package_name:
        Dotted package name to inspect, e.g. `"peft"` or `"transformers"`.
    src_path:
        Directory on `sys.path` (and griffe's search path) that contains the
        package checkout.  Honoured by both the static loader and the dynamic
        import under `--inspect`.
    recursive:
        Inspect `__all__` in every submodule, not just the root package.
        (Static mode only.)
    inspect:
        Resolve the export names at runtime by importing the package once; the
        rest (docstrings, paths) is still done statically.  Required for
        packages without a static `__all__` such as transformers.
    excludes:
        `fnmatch` wildcard patterns matched against each object's canonical
        dotted path; matched objects are dropped (counted as excluded, neither
        covered nor uncovered).  Empty tuple = no exclusions.
    diagnostics:
        Optional container collecting unresolved names / collisions / import
        errors / excluded count for the final report.
    """
    if diagnostics is None:
        diagnostics = Diagnostics()

    package = _load_package(package_name, src_path)

    if inspect:
        names = _dynamic_all(package_name, diagnostics)
        if names is None:
            # Dynamic import failed (e.g. missing deps).  Fall back to the static
            # path so the tool still produces *something* useful, and surface the
            # failure in the report.
            return _items_from_exports([package], diagnostics, excludes)
        index = _build_short_name_index(package, diagnostics, excludes)
        return _items_from_index(names, index, diagnostics, excludes)

    modules = walk_modules(package) if recursive else [package]
    return _items_from_exports(modules, diagnostics, excludes)


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
    """Walk every `*.md` under *docs_dir* and return the set of names that
    are referenced either inline or via `[[autodoc]]`."""
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
    diagnostics: Diagnostics | None = None,
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
        print(
            "\nThere are functions with a docstring in the public API (part of `peft.__all__`) \n"
            "that are not mentioned in the docs. Please add them to the docs.\n"
        )
        print(f"Missing ({len(uncovered)}):")
        for name in sorted(uncovered)[:20]:
            print(f"  - {name}")
        if len(uncovered) > 20:
            print(f"  ... and {len(uncovered) - 20} more")

    if diagnostics is not None:
        if diagnostics.excluded:
            print()
            print(f"Excluded {diagnostics.excluded} object(s) matching --exclude patterns.")
        if diagnostics.import_error:
            print()
            print(f"Import (--inspect) failed: {diagnostics.import_error}")
        if diagnostics.unresolved:
            print()
            print(f"Could not resolve {len(diagnostics.unresolved)} exported name(s) to a griffe object:")
            for name in diagnostics.unresolved[:20]:
                print(f"  - {name}")
            if len(diagnostics.unresolved) > 20:
                print(f"  ... and {len(diagnostics.unresolved) - 20} more")
        if diagnostics.collisions:
            print()
            print(f"Short-name collisions ({len(diagnostics.collisions)}, kept the first hit):")
            for short, kept, dropped in diagnostics.collisions[:20]:
                print(f"  - {short}: kept {kept}, dropped {dropped}")
            if len(diagnostics.collisions) > 20:
                print(f"  ... and {len(diagnostics.collisions) - 20} more")

    return bool(uncovered)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--package", default="peft", help="Package name to inspect.")
    parser.add_argument("--src", default="src", help="Path to the source tree root (containing the package dir).")
    parser.add_argument("--docs", default="docs/source", help="Path to the markdown docs directory.")
    parser.add_argument(
        "--recursive", action="store_true", help="Inspect __all__ in every submodule, not just the root package."
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help=(
            "Resolve __all__ at runtime by importing the package once (the rest "
            "is still done statically). Required for packages without a static "
            "__all__, e.g. transformers."
        ),
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="PATTERN",
        help=(
            "fnmatch wildcard matched against canonical dotted paths (repeatable). "
            "Matched objects are dropped from the public API, e.g. "
            "--exclude 'transformers.models.*'. '*' spans dots."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Print every covered/missing item.")
    args = parser.parse_args(argv)

    if not Path(args.src).is_dir():
        print(f"Error: source path not found: {args.src}", file=sys.stderr)
        return 1
    if not Path(args.docs).is_dir():
        print(f"Error: docs path not found: {args.docs}", file=sys.stderr)
        return 1

    # Both griffe's search path and `importlib.import_module` (for --inspect)
    # need the checkout's source root ahead of whatever is installed.
    if args.src not in sys.path:
        sys.path.insert(0, args.src)

    print(f"Loading package '{args.package}' from {args.src} ...")
    diagnostics = Diagnostics()
    api_items = extract_public_api_items(
        args.package,
        args.src,
        recursive=args.recursive,
        inspect=args.inspect,
        excludes=tuple(args.exclude),
        diagnostics=diagnostics,
    )
    print(f"Found {len(api_items)} public objects with docstrings.")

    print(f"Scanning docs in {args.docs} ...")
    mentions = extract_doc_mentions(args.docs)
    print(f"Found {len(mentions)} unique name mentions in docs.")

    print()
    has_uncovered = print_report(api_items, mentions, diagnostics=diagnostics, verbose=args.verbose)
    return 1 if has_uncovered else 0


if __name__ == "__main__":
    raise SystemExit(main())
