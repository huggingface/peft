"""This module implements our CI function calls."""

import nox


@nox.session(name="test")
def run_test(session):
    """Run pytest."""
    session.install(".[tests]")
    session.run("pytest")


@nox.session(name="fast-test")
def run_test_fast(session):
    """Run pytest."""
    session.install(".[tests]")
    session.run("pytest", "-m", "not slow")


@nox.session(name="lint")
def lint(session):
    """Check code conventions."""
    session.install("flake8")
    session.install(
        "flake8-black",
        "flake8-docstrings",
        "flake8-bugbear",
        "flake8-broken-line",
        "pep8-naming",
        "pydocstyle",
        "darglint",
    )
    session.run("flake8", "src", "tests", "noxfile.py")

    session.install("sphinx", "doc8")
    session.run("doc8", "--max-line-length", "120", "docs/")


@nox.session(name="typing")
def mypy(session):
    """Check type hints."""
    session.install(".[typing]")
    session.run(
        "mypy",
        "--install-types",
        "--non-interactive",
        "--ignore-missing-imports",
        "--strict",
        "--no-warn-return-any",
        "--explicit-package-bases",
        "src",
        "tests",
    )


@nox.session(name="docs")
def docs(session):
    """Build docs."""
    session.install(".")
    session.install("sphinx-book-theme")
    session.install("sphinxcontrib-bibtex")
    session.run(
        "python",
        "-m",
        "sphinx",
        "-W",
        "-b",
        "html",
        "-d",
        "docs/build/doctrees",
        "docs/",
        "docs/_build/html",
    )


@nox.session(name="format")
def format(session):
    """Fix common convention problems automatically."""
    session.install("black")
    session.install("isort")
    session.run("isort", ".")
    session.run("black", ".")


@nox.session(name="coverage")
def check_coverage(session):
    """Check test coverage and generate a html report."""
    session.install(".")
    session.install("pytest")
    session.install("coverage")
    try:
        session.run("coverage", "run", "-m", "pytest")
    finally:
        session.run("coverage", "html")


@nox.session(name="coverage-clean")
def clean_coverage(session):
    """Remove the code coverage website."""
    session.run("rm", "-r", "htmlcov", external=True)


@nox.session(name="build")
def build(session):
    """Build a pip package."""
    session.install("wheel")
    session.install("setuptools")
    session.run("python", "setup.py", "-q", "sdist", "bdist_wheel")


@nox.session(name="finish")
def finish(session):
    """Finish this version increase the version number and upload to pypi."""
    session.install("bump2version")
    session.install("twine")
    session.run("bumpversion", "release", external=True)
    build(session)
    session.run("twine", "upload", "--skip-existing", "dist/*", external=True)
    session.run("git", "push", external=True)
    session.run("bumpversion", "patch", external=True)
    session.run("git", "push", external=True)


@nox.session(name="check-package")
def pyroma(session):
    """Run pyroma to check if the package is ok."""
    session.install("pyroma")
    session.run("pyroma", ".")
