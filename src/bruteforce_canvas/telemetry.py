"""Runtime telemetry for the bruteforce_canvas application.

This module provides lightweight, optional telemetry helpers that probe
NVIDIA GPU VRAM usage via the ``pynvml`` driver API.  ``pynvml`` lives in
the ``[ml]`` extras, so the helpers MUST degrade gracefully when the
package is not installed, the driver is missing, or no GPU is visible.

The model is intentionally a frozen ``StrictModel`` so the snapshot can
be embedded inside other Pydantic models (such as runtime-state
snapshots) without leaking extra fields.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from bruteforce_canvas.shared import StrictModel

if TYPE_CHECKING:
    from pynvml import NvmlDevice  # pragma: no cover - type hint only


_BYTES_PER_GIB: int = 1024 ** 3


class VRAMTelemetry(StrictModel):
    """A single point-in-time snapshot of NVIDIA VRAM usage.

    All values are reported in gibibytes (GiB) so consumers can compare
    telemetry across machines with different memory sizes.  ``timestamp``
    is a POSIX float (seconds since epoch) captured at the moment the
    measurement was taken.
    """

    total_gib: float
    used_gib: float
    free_gib: float
    timestamp: float


def measure_vram_gib() -> float:
    """Return the currently used VRAM in GiB, or ``0.0`` if unavailable.

    This function never raises: any failure to import ``pynvml``,
    initialize the driver, query a device, or interpret the result is
    swallowed and reported as ``0.0``.  That keeps the rest of the
    application — which may run in pure CPU mode — blissfully unaware of
    GPU telemetry problems.
    """

    try:
        import pynvml  # type: ignore[import-untyped]  # lazy import
    except Exception:
        return 0.0

    try:
        pynvml.nvmlInit()
    except Exception:
        return 0.0

    try:
        try:
            device_count = pynvml.nvmlDeviceGetCount()
        except Exception:
            return 0.0
        if device_count <= 0:
            return 0.0
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            return 0.0
        try:
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        except Exception:
            return 0.0
        return float(memory.used) / float(_BYTES_PER_GIB)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def collect_vram_telemetry() -> VRAMTelemetry:
    """Build a full :class:`VRAMTelemetry` snapshot.

    The implementation queries the first visible NVIDIA device for
    total/free/used VRAM, falling back to ``0.0`` for every field if
    ``pynvml`` is unavailable or the probe fails for any reason.  The
    ``timestamp`` is always populated so downstream consumers can age
    out stale samples.
    """

    try:
        import pynvml  # type: ignore[import-untyped]  # lazy import
    except Exception:
        return VRAMTelemetry(total_gib=0.0, used_gib=0.0, free_gib=0.0, timestamp=time.time())

    try:
        pynvml.nvmlInit()
    except Exception:
        return VRAMTelemetry(total_gib=0.0, used_gib=0.0, free_gib=0.0, timestamp=time.time())

    try:
        try:
            device_count = pynvml.nvmlDeviceGetCount()
        except Exception:
            return VRAMTelemetry(total_gib=0.0, used_gib=0.0, free_gib=0.0, timestamp=time.time())
        if device_count <= 0:
            return VRAMTelemetry(total_gib=0.0, used_gib=0.0, free_gib=0.0, timestamp=time.time())
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            return VRAMTelemetry(total_gib=0.0, used_gib=0.0, free_gib=0.0, timestamp=time.time())
        try:
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        except Exception:
            return VRAMTelemetry(total_gib=0.0, used_gib=0.0, free_gib=0.0, timestamp=time.time())
        return VRAMTelemetry(
            total_gib=float(memory.total) / float(_BYTES_PER_GIB),
            used_gib=float(memory.used) / float(_BYTES_PER_GIB),
            free_gib=float(memory.free) / float(_BYTES_PER_GIB),
            timestamp=time.time(),
        )
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
