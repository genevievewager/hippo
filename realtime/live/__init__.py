"""Live deployment: streams, causal buffer, unit mapping, session logging."""

from realtime.live.config import DeployableConfiguration, RuntimeState
from realtime.live.spike_buffer import CausalSpikeBuffer
from realtime.live.spike_stream import (
    OpenEphysSpikeStream,
    ReplaySpikeStream,
    SpikeStream,
)
from realtime.live.unit_mapping import UnitMappingReport, map_units

__all__ = [
    "CausalSpikeBuffer",
    "DeployableConfiguration",
    "DeploymentRegistry",
    "OpenEphysSpikeStream",
    "ReplaySpikeStream",
    "RuntimeState",
    "SpikeStream",
    "UnitMappingReport",
    "map_units",
    "registry_for",
]


def __getattr__(name: str):
    # Lazy exports to avoid circular imports with deployment_bundle.
    if name in ("DeploymentRegistry", "registry_for"):
        from realtime.live import registry as _registry

        return getattr(_registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

