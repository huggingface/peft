# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Enforce issue approval for new PRs. Setup and policy are documented in triage_prs.yml."""

import os
import re
from datetime import datetime, timezone
from functools import cache
from urllib.parse import urlsplit

import requests


TRIAGED_LABEL = "triaged"
CLOSURE_MARKER = "<!-- peft-pr-triage: approval-required -->"
HUMAN_MARKER = "This PR was authored by a human"
REPOSITORY = "huggingface/peft"
START_DATE = "2026-09-09"  # Inclusive creation date in UTC.
BOT_NAME = "peft-triage"

# see: https://api.github.com/users/<user-id>
MAINTAINERS = {"BenjaminBossan": 6229650, "githubnemo": 264196}
ALLOW_LIST_USERS = {"dependabot[bot]": 49699333, "peft-jambot": 295153068}
ALLOW_LIST_ORGANIZATIONS = {}  # Organization name -> immutable organization ID.
# see: https://api.github.com/orgs/huggingface
HF_ORGANIZATION_ID = 25720743
GITHUB_ACTIONS_BOT_ID = 41898282
MAX_ISSUE_REFERENCES = 10

CLOSURE_MESSAGE = (
    "{closure_marker}\n\n"
    "Thank you for your interest in contributing to PEFT. This PR is being closed because its description "
    "does not reference a PEFT issue with explicit approval from a maintainer or public Hugging Face "
    "organization member.\n\n"
    "Please open or find an issue, discuss the proposed contribution, and wait for an authorized person to "
    "comment `@{bot_name} approved` on its own line before opening a PR. "
    "Once approved, reference the issue in this PR's description (for example, `Fixes #123`) and reopen it; "
    "there is no need to create another PR.\n\n"
    "If you believe this PR was closed incorrectly, please ping the maintainers here.\n\n"
    "See the [contribution guidelines]"
    "(https://github.com/{repository}/blob/main/docs/source/developer_guides/contributing.md)."
)


def str_to_bool(value: str) -> bool:
    """
    Converts a string representation of truth to `True` (1) or `False` (0).

    True values are `y`, `yes`, `t`, `true`, `on`, and `1`; False value are `n`, `no`, `f`, `false`, `off`, and `0`;
    """
    # Same as function as in accelerate.utils, which replaces the deprecated distutils.util.strtobool, and returns bool
    # instead of int.
    value = value.lower()
    if value in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif value in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError(f"invalid truth value {value}")


class GitHubClient:
    """Small REST client; failed or incomplete reads propagate before any dependent writes."""

    def __init__(self, token):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )

    def request(self, method, path, *, public=False, **kwargs):
        # Public membership needs no organization credential. Do not follow redirects or URLs supplied in contribution
        # text with the repository token.
        response = self.session.request(
            method,
            f"https://api.github.com{path}",
            headers={"Authorization": None} if public else {},
            timeout=30,
            allow_redirects=False,
            **kwargs,
        )
        response.raise_for_status()
        if not 200 <= response.status_code < 300:
            raise RuntimeError(f"Unexpected GitHub status {response.status_code} for {path}")
        return response

    def get(self, path):
        return self.request("GET", path).json()

    def list_items(self, path, *, public=False, **params):
        """Read every page before returning, including when later pages fail."""
        items = []
        page = 1
        per_page = 100
        while True:
            batch = self.request(
                "GET", path, public=public, params={**params, "per_page": per_page, "page": page}
            ).json()
            items.extend(batch)
            if len(batch) < per_page:  # hit the end
                return items
            page += 1

    @cache  # # ruff: ignore[B019]
    def public_members(self, organization, organization_id):
        # An organization's old name can be claimed by someone else after a rename.
        metadata = self.request("GET", f"/orgs/{organization}", public=True).json()
        if metadata["id"] != organization_id:
            raise RuntimeError(f"Organization ID mismatch for {organization}; review the configured identity.")
        return {member["id"] for member in self.list_items(f"/orgs/{organization}/public_members", public=True)}


def extract_issue_numbers(body, repository):
    """Extract local #123, owner/repo#123 and GitHub issue URLs in description order."""
    numbers = []
    # Consume whole URLs and qualified references so foreign references cannot be
    # mistaken for local issue numbers (including URL fragments).
    references = re.finditer(r"https?://[^\s<>`]+|(?<![\w/.-])(?:[\w.-]+/[\w.-]+)?#[1-9][0-9]*", body or "")
    for reference in references:
        text = reference.group().rstrip(".,;:)]}")
        if text.startswith(("https://", "http://")):
            url = urlsplit(text)
            match = re.fullmatch(r"/([^/]+/[^/]+)/issues/([1-9][0-9]*)/?", url.path)
            if url.netloc.lower() != "github.com" or not match or match[1].lower() != repository.lower():
                continue
            number = match[2]
        else:
            repo, number = text.split("#")
            if repo and repo.lower() != repository.lower():
                continue

        # Bound integer conversion and API work on contributor-controlled input.
        if len(number) > 20:
            raise ValueError("Issue reference exceeds the supported number length.")

        number = int(number)
        if number not in numbers:
            numbers.append(number)
            if len(numbers) > MAX_ISSUE_REFERENCES:
                raise ValueError(f"More than {MAX_ISSUE_REFERENCES} issue references; manual triage required.")

    return numbers


def contains_approval(body, bot_name):
    """Require the command on its own line, outside Markdown code fences."""
    fence = None
    for line in (body or "").splitlines():
        if fence is not None:
            # A shorter fence or one with trailing text cannot close a code block.
            if re.fullmatch(rf" {{0,3}}{fence[0]}{{{len(fence)},}}[ \t]*", line):
                fence = None
            continue

        opening = re.match(r" {0,3}(`{3,}|~{3,})", line)
        if opening:
            fence = opening[1]
        elif re.fullmatch(rf" {{0,3}}@{re.escape(bot_name)} approved[ \t]*", line):
            return True

    return False


def is_eligible_pr(pr, since):
    return (
        pr["state"] == "open"
        and datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00")) >= since
        and not any(label["name"].lower() == TRIAGED_LABEL for label in pr["labels"])
    )


class PullRequestTriage:
    def __init__(self, client, repository, since, bot_name, maintainers, allowed_users=(), allowed_organizations=()):
        self.client = client
        self.repository = repository
        self.path = f"/repos/{repository}"
        self.since = since
        self.bot_name = bot_name
        self.maintainers = set(maintainers)
        self.allowed_users = set(allowed_users)
        self.allowed_organizations = dict(allowed_organizations)

    def can_approve(self, user_id):
        return user_id in self.maintainers or user_id in self.client.public_members("huggingface", HF_ORGANIZATION_ID)

    def is_exempt_author(self, user_id):
        return (
            user_id in self.maintainers
            or user_id in self.allowed_users
            or user_id in self.client.public_members("huggingface", HF_ORGANIZATION_ID)
            or any(
                user_id in self.client.public_members(org, org_id)
                for org, org_id in sorted(self.allowed_organizations.items())
            )
        )

    def has_approved_issue(self, body):
        for number in extract_issue_numbers(body, self.repository):
            issue = self.client.get(f"{self.path}/issues/{number}")
            if "pull_request" in issue:
                continue

            comments = self.client.list_items(f"{self.path}/issues/{number}/comments")
            if any(
                contains_approval(comment["body"], self.bot_name) and self.can_approve(comment["user"]["id"])
                for comment in comments
            ):
                return True

        return False

    def is_human_author(self, body):
        body = body or ""
        return HUMAN_MARKER.lower() in body.lower()

    def closure_message(self):
        return CLOSURE_MESSAGE.format(
            closure_marker=CLOSURE_MARKER, bot_name=self.bot_name, repository=self.repository
        )

    def triage_pr(self, pr, *, dry_run=False):
        if not is_eligible_pr(pr, self.since):
            return

        number = pr["number"]
        # Fetch current labels, state and description before making the decision.
        pr = self.client.get(f"{self.path}/pulls/{number}")
        if not is_eligible_pr(pr, self.since):
            return

        approved = self.is_exempt_author(pr["user"]["id"]) or self.has_approved_issue(pr["body"]) or self.is_human_author(pr["body"])
        comments = [] if approved else self.client.list_items(f"{self.path}/issues/{number}/comments")
        current = self.client.get(f"{self.path}/pulls/{number}")
        if not is_eligible_pr(current, self.since) or current["updated_at"] != pr["updated_at"]:
            print(f"Skipping PR #{number}: changed during triage; will reconsider next run.")
            return

        action = "label triaged" if approved else "close: missing issue approval"
        print(f"{'Would' if dry_run else 'Will'} {action} on PR #{number}")
        if dry_run:
            return

        if approved:
            self.client.request("POST", f"{self.path}/issues/{number}/labels", json={"labels": [TRIAGED_LABEL]})
            return

        if not any(
            comment["user"]["id"] == GITHUB_ACTIONS_BOT_ID and CLOSURE_MARKER in (comment["body"] or "")
            for comment in comments
        ):
            self.client.request("POST", f"{self.path}/issues/{number}/comments", json={"body": self.closure_message()})
        # Recheck after commenting as well; the comment itself updates updated_at.
        current = self.client.get(f"{self.path}/pulls/{number}")
        if is_eligible_pr(current, self.since) and current["body"] == pr["body"]:
            self.client.request("PATCH", f"{self.path}/pulls/{number}", json={"state": "closed"})

    def run(self, *, dry_run=False):
        # Snapshot all pages before closing PRs, which changes pagination of open PRs.
        prs = self.client.list_items(f"{self.path}/pulls", state="open")
        failures = []
        for pr in prs:
            try:
                self.triage_pr(pr, dry_run=dry_run)
            except (requests.RequestException, RuntimeError, ValueError) as error:
                failures.append(pr["number"])
                # Escape newlines and both current and legacy Actions command delimiters.
                message = repr(error).replace("::", ": :").replace("##[", "# #[")
                print(f"Failed to triage PR #{pr['number']}: {message}")
        if failures:
            raise RuntimeError(f"Triage failed for PRs {failures}; see earlier errors. Rerun after resolving them.")


def main():
    triage = PullRequestTriage(
        client=GitHubClient(os.environ["GITHUB_TOKEN"]),
        repository=REPOSITORY,
        since=datetime.strptime(START_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc),
        bot_name=BOT_NAME,
        maintainers=MAINTAINERS.values(),
        allowed_users=ALLOW_LIST_USERS.values(),
        allowed_organizations=ALLOW_LIST_ORGANIZATIONS,
    )
    triage.run(dry_run=str_to_bool(os.environ.get("DRY_RUN", "false")))


if __name__ == "__main__":
    main()
