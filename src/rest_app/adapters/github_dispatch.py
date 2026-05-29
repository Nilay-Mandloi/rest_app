"""GitHub repository_dispatch adapter for OrchestrationAdapter.

Fires a `train-model` event on the configured training repo. Only file in
the package that imports urllib for GitHub API access; swap with
JenkinsAdapter / ArgoAdapter without touching publisher.py.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from loguru import logger

from rest_app.ports.orchestration import OrchestrationAdapter


class GitHubDispatchAdapter(OrchestrationAdapter):
    _RETRYABLE_HTTP = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        *,
        training_repo: str,
        training_repo_token: str,
        training_branch: str = "main",
        timeout_s: int = 10,
        retry_pause_s: int = 2,
    ) -> None:
        if not training_repo or not training_repo_token:
            raise ValueError("training_repo and training_repo_token are both required")
        self._repo = training_repo
        self._token = training_repo_token
        self._branch = training_branch or "main"
        self._timeout_s = timeout_s
        self._retry_pause_s = retry_pause_s

    def dispatch_training(
        self,
        *,
        trigger_id: str,
        category: str,
        project: str,
        model_name: str,
        bucket: str,
        prefix: str,
        auto_promote: bool,
    ) -> None:
        # workflow_dispatch lets us target a specific branch (ref); repository_dispatch
        # is locked to the default branch and cannot be redirected.
        url = f"https://api.github.com/repos/{self._repo}/actions/workflows/train.yaml/dispatches"
        payload = json.dumps(
            {
                "ref": self._branch,
                "inputs": {
                    "source": "rest_app",
                    "trigger_id": trigger_id,
                    "auto_promote": str(auto_promote).lower(),
                    "category": category,
                    "project": project,
                    "model_name": model_name,
                    "artifact_store_bucket": bucket,
                    "artifact_store_prefix": prefix,
                },
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            method="POST",
        )

        last_exc: Exception | None = None
        status: int | None = None
        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                    status = resp.status
                break
            except urllib.error.HTTPError as exc:
                if exc.code in self._RETRYABLE_HTTP and attempt == 0:
                    last_exc = exc
                    logger.warning(
                        "GitHub dispatch HTTP {} (attempt 1/2), retrying in {}s",
                        exc.code,
                        self._retry_pause_s,
                    )
                    time.sleep(self._retry_pause_s)
                else:
                    raise RuntimeError(
                        f"GitHub dispatch failed: HTTP {exc.code} for repo={self._repo} "
                        f"trigger_id={trigger_id}. Check token Contents:write permissions."
                    ) from exc
            except OSError as exc:
                last_exc = exc
                if attempt == 0:
                    logger.warning(
                        "GitHub dispatch network error (attempt 1/2), retrying in {}s: {}",
                        self._retry_pause_s,
                        exc,
                    )
                    time.sleep(self._retry_pause_s)
        else:
            raise RuntimeError(
                f"GitHub dispatch failed after retry for trigger_id={trigger_id}: {last_exc}"
            )

        if status not in (200, 201, 204):
            raise RuntimeError(
                f"GitHub dispatch returned unexpected status {status} "
                f"for repo={self._repo} trigger_id={trigger_id}."
            )
        logger.info(
            "Dispatched train-model event to {} (trigger_id={} auto_promote={})",
            self._repo,
            trigger_id,
            auto_promote,
        )


class NoopDispatchAdapter(OrchestrationAdapter):
    """Local-dev / test adapter: logs and returns."""

    def dispatch_training(
        self,
        *,
        trigger_id: str,
        category: str,
        project: str,
        model_name: str,
        bucket: str,
        prefix: str,
        auto_promote: bool,
    ) -> None:
        logger.info(
            "noop dispatch: trigger_id={} target={}/{}/{} bucket={} prefix={} auto_promote={}",
            trigger_id,
            category,
            project,
            model_name,
            bucket,
            prefix,
            auto_promote,
        )
