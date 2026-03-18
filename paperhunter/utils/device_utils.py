"""Utilities per il rilevamento automatico del dispositivo di calcolo.

Questo modulo fornisce un'astrazione device-agnostic che seleziona
automaticamente il miglior acceleratore disponibile sulla macchina corrente,
in ordine di preferenza: CUDA (GPU NVIDIA) → MPS (GPU Apple Silicon) → CPU.

Usare queste funzioni al posto di qualsiasi .cuda() o torch.device("cuda")
hardcoded, in modo che il codice funzioni correttamente su qualsiasi hardware.
"""

from __future__ import annotations

import torch

from paperhunter.utils.logging_setup import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Funzioni pubbliche
# ---------------------------------------------------------------------------


def get_device() -> torch.device:
    """Rileva automaticamente il miglior dispositivo disponibile.

    Priorità:
      1. CUDA  — GPU NVIDIA (Windows/Linux con driver CUDA installati)
      2. MPS   — GPU Apple Silicon (macOS M1/M2/M3/M4)
      3. CPU   — Fallback universale

    Returns:
        ``torch.device`` pronto all'uso.

    Example::

        device = get_device()
        model = model.to(device)
        tensor = torch.tensor(data).to(device)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.debug("Dispositivo selezionato: CUDA (%s)", torch.cuda.get_device_name(0))
        return device

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.debug("Dispositivo selezionato: MPS (Apple Silicon)")
        return torch.device("mps")

    logger.debug("Dispositivo selezionato: CPU (nessun acceleratore disponibile)")
    return torch.device("cpu")


def get_autocast_dtype(device: torch.device) -> torch.dtype:
    """Restituisce il dtype ottimale per ``torch.autocast`` sul dispositivo dato.

    - ``cuda`` / ``mps``: ``bfloat16`` — più stabile di float16, nativo su
      GPU NVIDIA Ampere+ e Apple M1/M2/M3/M4 con PyTorch >= 2.0.
    - ``cpu``: ``float32`` — bfloat16 su CPU è significativamente più lento.

    Args:
        device: Il dispositivo restituito da :func:`get_device`.

    Returns:
        ``torch.dtype`` da passare a ``torch.autocast``.
    """
    if device.type in ("cuda", "mps"):
        return torch.bfloat16
    return torch.float32


def move_to_device(
    tensor: torch.Tensor,
    device: torch.device,
    *,
    non_blocking: bool = True,
) -> torch.Tensor:
    """Sposta un tensor sul dispositivo target in modo ottimale.

    Su CUDA utilizza la pinned memory per massimizzare la bandwidth DMA.
    Su MPS e CPU esegue un semplice ``.to(device)``.

    Args:
        tensor: Tensor da spostare (deve essere su CPU).
        device: Dispositivo di destinazione.
        non_blocking: Se ``True`` (default), il trasferimento su CUDA è
            asincrono (il chiamante deve sincronizzare prima di leggere
            il risultato).

    Returns:
        Il tensor sul dispositivo target.
    """
    if device.type == "cuda" and not tensor.is_pinned():
        return tensor.pin_memory().to(device, non_blocking=non_blocking)
    return tensor.to(device)
