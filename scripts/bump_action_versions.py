"""
Tool for upgrading workflow dependencies in a focussed manner.

Prints a summary of all version changes for review in the end.
Supports multiple workflows at once.
Caches queries so updates are fast.

"""

import argparse
import os
import sys

from copy import deepcopy
from functools import lru_cache

import requests
from ruamel.yaml import YAML


GITHUB_BEARER_TOKEN_FILE = '.github_bearer_token'


def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


@lru_cache
def get_github_bearer_token():
    if os.path.exists(GITHUB_BEARER_TOKEN_FILE):
        with open(GITHUB_BEARER_TOKEN_FILE) as f:
            return f.read().strip()
    return None


def get_github_auth_header():
    token = get_github_bearer_token()
    if token is None:
        return {}
    return {'Authorization': f'Bearer {token}'}


@lru_cache(None)
def get_latest_release(repo_path):
    """Error 403 is probably a rate limit. Check the URL in the browser.
    Consider creating an API token and placing it to `GITHUB_BEARER_TOKEN_FILE`.
    """
    headers = get_github_auth_header()

    release_url = f"https://api.github.com/repos/{repo_path}/releases/latest"
    print(release_url)
    response = requests.get(release_url, headers=headers)

    if response.status_code != 200:
        return None, f"Error: {response.status_code}"

    release_data = response.json()
    tag_name = release_data.get('tag_name')

    ref_url = f"https://api.github.com/repos/{repo_path}/git/refs/tags/{tag_name}"
    ref_res = requests.get(ref_url, headers=headers)

    if ref_res.status_code != 200:
        return None, f"Error fetching ref: {ref_res.status_code}"

    ref_data = ref_res.json()
    commit_sha = ref_data["object"]["sha"]

    return {
        'tag_name': tag_name,
        'commit_sha': commit_sha,
        'release_url': release_data.get('html_url'),
    }, None


@lru_cache
def get_tag_for_hash(repo_path, commit_sha):
    headers = get_github_auth_header()

    url = f"https://api.github.com/repos/{repo_path}/git/refs/tags"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        tags = response.json()
        for tag in tags:
            if tag['object']['sha'] == commit_sha:
                return tag['ref'].replace('refs/tags/', '')
    return None


def build_compare_url(project, old_version, new_version):
    return f'https://github.com/{project}/compare/{old_version}...{new_version}'


def process_workflow_file(workflow_file):
    # note: not using 'safe' loading but round-trip to be able to modify the data without
    # changing as little as possible except for the version tags and their comments.
    yaml = YAML(typ='rt')
    yaml.preserve_quotes = True
    yaml.width = 200  # prevents inserted line breaks
    yaml_data = yaml.load(workflow_file)


    if 'jobs' not in yaml_data:
        log("no jobs section in workflow file, not updating any versions")
        return

    update_summary = {}
    yaml_data_new = deepcopy(yaml_data)

    for job, job_description in yaml_data['jobs'].items():
        log(f"investigating {job=}")
        if 'steps' not in job_description:
            log("no steps in this job, not updating any versions")
            continue

        for step_idx, step_description in enumerate(job_description['steps']):
            if 'uses' not in step_description:
                continue

            step_name = step_description.get('name', 'NONE')
            uses = step_description['uses']

            if '@' not in uses:
                # not a pinned version, warning
                log(f"Warning: step {step_name} has use statement without version pin.")
                continue

            repo, version = uses.rsplit('@', 1)
            print(repo, version)

            # we assume that subfolders share the same release
            repo = '/'.join(repo.split('/')[:2])

            release_data, error = get_latest_release(repo)
            if error is not None:
                log(f"Error {step_name=} release fetch {repo=}: {error}")
                continue

            new_version = release_data['commit_sha']
            update_summary[(repo, version)] = {"old_version": version, 'new_version': new_version, 'release': release_data}
            yaml_data_new['jobs'][job]['steps'][step_idx]['uses'] = f"{repo}@{new_version}"
            yaml_data_new['jobs'][job]['steps'][step_idx].yaml_add_eol_comment(f'# {release_data["tag_name"]}', 'uses')

            log(f"Updated: {step_name=} {repo=} {version} -> {new_version}")

    return yaml, yaml_data_new, update_summary


def print_summary(update_summaries):
    print("Summary of combined updates:")
    print("----------------------------")
    print()
    print("The summary will include repos more than once if their current version differs.")
    print()

    summaries_merged = {}
    for summary in update_summaries:
        summaries_merged.update(summary)

    for (repo, version), summary in summaries_merged.items():
        release_data = summary['release']

        tag_current = get_tag_for_hash(repo, version)
        if not tag_current:
            tag_current = "UNKNOWN TAG"

        print(f"* Repository: {repo}, current version: {version} (tag: {tag_current})")
        print(f"  This was updated to version {summary['new_version']}")
        print(f"  This corresponds to release {release_data['tag_name']}")
        print(f"  Release information: {release_data['release_url']}")
        print(f"  See what changed: {build_compare_url(repo, version, summary['new_version'])}")
        print()


def main(args):
    """Update all the YAML files and print a summary of the changes for review."""
    update_summaries = []

    for workflow_file in args.workflow_files:
        yaml, yaml_data_new, update_summary = process_workflow_file(workflow_file)

        update_summaries.append(update_summary)

        with open(workflow_file.name, 'w') as f:
            yaml.dump(yaml_data_new, f)

    print('\n')
    print_summary(update_summaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('workflow_files', type=argparse.FileType('r'), nargs='+')

    args = parser.parse_args()

    main(args)
