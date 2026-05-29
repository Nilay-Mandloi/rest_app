"""Orchestration port — start a training run on whatever orchestrator the
deployment uses (GitHub Actions today; Argo, Jenkins, Airflow, Step Functions
tomorrow). Business logic depends only on this port."""

from __future__ import annotations

from abc import ABC, abstractmethod


class OrchestrationAdapter(ABC):
    @abstractmethod
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
        """Kick off a training run for trigger_id.

        Note on idempotency: ``trigger_id`` is generated server-side per
        request by the publisher, so duplicate dispatches do not occur in the
        normal request flow. Concrete adapters (e.g. GitHub
        ``repository_dispatch``) do NOT deduplicate at the API level — a
        second call with the same trigger_id WILL enqueue a second run.
        Callers must therefore not retry dispatch with the same trigger_id.

        Raise RuntimeError when the orchestrator refuses the request (auth,
        rate-limit, persistent failure). Callers treat that as a hard failure
        and write a failed.json marker so consumers don't poll forever.
        """
