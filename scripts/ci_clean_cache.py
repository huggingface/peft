"""
Utility to clean cache files that exceed a specific time in days according to their
last access time recorded in the cache.

Exit code:
- 1 if no candidates are found
- 0 if candidates are found

Deletion can be enabled by passing `-d` parameter, otherwise it will only list the candidates.
"""

import sys
from datetime import datetime as dt

from huggingface_hub import scan_cache_dir


def find_old_revisions(scan_results, max_age_days=30):
    """Find commit hashes of objects in the cache. These objects need a last access time that
    is above the passed `max_age_days` parameter. Returns an empty list if no objects are found.
    Time measurement is based of the current time and the recorded last access tiem in the cache.
    """
    now = dt.now()
    revisions = [(i.revisions, i.last_accessed) for i in scan_results.repos]
    revisions_ages = [(rev, (now - dt.fromtimestamp(ts_access)).days) for rev, ts_access in revisions]
    delete_candidates = [rev for rev, age in revisions_ages if age > max_age_days]
    hashes = [n.commit_hash for rev in delete_candidates for n in rev]

    return hashes


def delete_old_revisions(scan_results, delete_candidates, do_delete=False):
    delete_operation = scan_results.delete_revisions(*delete_candidates)
    print(f"Would free {delete_operation.expected_freed_size_str}")
    print(f"Candidates: {delete_candidates}")

    if do_delete:
        print("Deleting now.")
        delete_operation.execute()
    else:
        print("Not deleting, pass the -d flag.")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-a", "--max-age", type=int, default=30, help="Max. age in days items in the cache may have.")
    parser.add_argument(
        "-d",
        "--delete",
        action="store_true",
        help=(
            "Delete mode; Really delete items if there are candidates. Exit code = 0 when we found something to delete, 1 "
            "otherwise."
        ),
    )
    args = parser.parse_args()

    scan_results = scan_cache_dir()

    delete_candidates = find_old_revisions(scan_results, args.max_age)
    if not delete_candidates:
        print("No delete candidates found, not deleting anything.")
        sys.exit(1)

    delete_old_revisions(scan_results, delete_candidates, do_delete=args.delete)
