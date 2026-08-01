"""Single source of truth for compute-device selection.

worldproof runs on Apple Silicon, Linux, and Windows. No code calls
``.cuda()`` directly; everything that needs a device goes through
:func:`get_device`, which autodetects ``cuda`` > ``mps`` > ``cpu``.
"""

from __future__ import annotations

import torch

__all__ = ["available_devices", "get_device"]

_VALID_DEVICES = ("cpu", "mps", "cuda")


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def available_devices() -> list[str]:
    """Return the device backends usable on this machine, ``cpu`` first."""
    devices = ["cpu"]
    if _mps_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def get_device(prefer: str | None = None) -> torch.device:
    """Return the compute device to use.

    With no argument, autodetects in order of preference: ``cuda``, then
    ``mps``, then ``cpu``. Pass ``prefer`` to force a specific backend
    (``"cpu"``, ``"mps"``, or ``"cuda"``).

    Args:
        prefer: Optional backend to force. Must be available on this machine.

    Returns:
        The selected :class:`torch.device`.

    Raises:
        ValueError: If ``prefer`` is not a recognized backend name.
        RuntimeError: If ``prefer`` is recognized but unavailable here.
    """
    if prefer is not None:
        choice = prefer.lower()
        if choice not in _VALID_DEVICES:
            raise ValueError(f"unknown device {prefer!r}; choose from {_VALID_DEVICES}")
        if choice not in available_devices():
            raise RuntimeError(
                f"device {prefer!r} was requested but is not available on this "
                f"machine (available: {available_devices()}). Omit `prefer` to "
                "autodetect a supported device."
            )
        return torch.device(choice)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")
