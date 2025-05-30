# -*- coding: utf-8 -*-

"""Version information for :mod:`ptwt`.

Run with ``python -m ptwt.version``
"""

import os
from subprocess import CalledProcessError, check_output  # noqa: S404

__all__ = [
    "VERSION",
    "get_version",
    "_get_git_hash",
]

VERSION = "0.1.10-dev"


def _get_git_hash() -> str:
    """Get the :mod:`ptwt` git hash."""
    with open(os.devnull, "w") as devnull:
        try:
            ret = check_output(  # noqa: S603,S607
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(__file__),
                stderr=devnull,
            )
        except CalledProcessError:
            return "UNHASHED"
        else:
            return ret.strip().decode("utf-8")[:8]


def get_version(with_git_hash: bool = False) -> str:
    """Get the :mod:`ptwt` version string, including a git hash."""
    return f"{VERSION}-{_get_git_hash()}" if with_git_hash else VERSION


if __name__ == "__main__":
    print(get_version(with_git_hash=True))
