"""Thin registry facade for best deployable configuration selection."""

from __future__ import annotations

from pathlib import Path

from realtime.deployment_bundle import best_deployable
from realtime.deployment_selection import DEPLOYMENT_SPIKE_SOURCE
from realtime.live.config import DeployableConfiguration


class DeploymentRegistry:
    """Query offline comparison / public registry for deployable winners."""

    def __init__(self, experiment_dir: Path | str):
        self.experiment_dir = Path(experiment_dir)

    def best(
        self,
        target: str,
        *,
        spike_source: str = DEPLOYMENT_SPIKE_SOURCE,
        deployable_only: bool = True,
        selection_policy: str = "shortest_near_optimal",
    ) -> DeployableConfiguration:
        return best_deployable(
            self.experiment_dir,
            target,
            spike_source=spike_source,
            deployable_only=deployable_only,
            selection_policy=selection_policy,
        )


def registry_for(experiment_dir: Path | str) -> DeploymentRegistry:
    return DeploymentRegistry(experiment_dir)
