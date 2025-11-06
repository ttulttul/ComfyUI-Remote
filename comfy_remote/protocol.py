"""Serialization and protocol helpers for the ComfyUI Remote node."""

from __future__ import annotations

import base64
import logging
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # ComfyUI always ships with torch, but guard for lint/tests.
    import torch
except ModuleNotFoundError:  # pragma: no cover - only triggered in limited envs.
    torch = None  # type: ignore

logger = logging.getLogger(__name__)

ENCODING_BASE64_PICKLE = "base64+pickle"
DEFAULT_PROGRESS_MAX = 100.0


@dataclass(frozen=True)
class SerializedValue:
    """Container describing a serialized input or output payload."""

    name: str
    comfy_type: str
    python_type: str
    encoding: str
    data: str
    metadata: Dict[str, Any] | None = None


def infer_comfy_type(value: Any) -> str:
    """Best-effort guess of the ComfyUI IO type for the given value."""
    if value is None:
        return "NONE"
    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, int):
        return "INT"
    if isinstance(value, float):
        return "FLOAT"
    if isinstance(value, str):
        return "STRING"
    if torch is not None and isinstance(value, torch.Tensor):
        dims = value.ndim
        if dims >= 4:
            # Images are typically [B, H, W, C], latents [B, C, H, W].
            if value.shape[-1] in (3, 4):
                return "IMAGE"
            if value.shape[1] in (3, 4):
                return "LATENT"
        if dims == 3 and value.shape[-1] in (3, 4):
            return "IMAGE"
        if dims == 3 and value.shape[0] in (3, 4):
            return "LATENT"
        if dims == 2:
            return "MASK"
        return "TENSOR"
    if isinstance(value, list):
        if value and isinstance(value[0], tuple) and len(value[0]) == 2:
            return "CONDITIONING"
        return "LIST"
    if isinstance(value, dict):
        return "DICT"
    return "ANY"


def _encode_pickle(value: Any) -> str:
    """Serialize *value* using pickle and encode as a base64 string."""
    try:
        raw_bytes = pickle.dumps(value)
    except Exception as exc:  # pragma: no cover - depends on upstream objects.
        logger.error("Failed to pickle value of type %s: %s", type(value), exc)
        raise
    return base64.b64encode(raw_bytes).decode("utf-8")


def serialize_value(name: str, value: Any) -> SerializedValue:
    """Serialize a single input for transmission to the remote service."""
    comfy_type = infer_comfy_type(value)
    python_type = f"{type(value).__module__}.{type(value).__qualname__}"
    encoded = _encode_pickle(value)
    serialized = SerializedValue(
        name=name,
        comfy_type=comfy_type,
        python_type=python_type,
        encoding=ENCODING_BASE64_PICKLE,
        data=encoded,
    )
    logger.debug(
        "Serialized input %s as type %s (python=%s)", name, comfy_type, python_type
    )
    return serialized


def serialize_inputs(items: Sequence[Tuple[str, Any]]) -> List[SerializedValue]:
    """Serialize ordered inputs, skipping any that were not connected."""
    serialized: list[SerializedValue] = []
    for name, value in items:
        if value is None:
            logger.debug("Skipping input %s because value is None", name)
            continue
        serialized.append(serialize_value(name, value))
    return serialized


def deserialize_value(payload: Dict[str, Any]) -> Any:
    """Reconstruct a Python object from a remote payload."""
    encoding = payload.get("encoding", ENCODING_BASE64_PICKLE)
    data = payload.get("data")
    if data is None:
        raise ValueError("Remote payload missing 'data' field")
    if encoding == ENCODING_BASE64_PICKLE:
        decoded = base64.b64decode(data)
        return pickle.loads(decoded)
    if encoding == "identity":
        return data
    raise ValueError(f"Unsupported remote payload encoding: {encoding}")


def normalise_result_envelope(response: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the result section from an arbitrary response envelope."""
    for key in ("result", "output", "data"):
        if key in response and isinstance(response[key], dict):
            return response[key]
    return response


def extract_progress(response: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Pick the best progress tuple (value, max_value) if present."""
    progress = response.get("progress")
    if progress is None:
        return None
    if isinstance(progress, dict):
        value = progress.get("value")
        total = progress.get("max") or progress.get("total") or progress.get("goal")
        if value is not None and total:
            try:
                return float(value), float(total)
            except (TypeError, ValueError):
                logger.debug("Progress value/total not numeric: %s", progress)
        percent = progress.get("percent") or progress.get("percentage")
        if percent is not None:
            try:
                percent_float = float(percent)
            except (TypeError, ValueError):
                logger.debug("Progress percent not numeric: %s", percent)
            else:
                if percent_float <= 1:
                    return percent_float, 1.0
                return percent_float, 100.0
        ratio = progress.get("ratio")
        if ratio is not None:
            try:
                ratio_float = float(ratio)
            except (TypeError, ValueError):
                logger.debug("Progress ratio not numeric: %s", ratio)
            else:
                return ratio_float, 1.0
    elif isinstance(progress, (int, float)):
        progress_float = float(progress)
        if 0.0 <= progress_float <= 1.0:
            return progress_float, 1.0
        return progress_float, DEFAULT_PROGRESS_MAX
    return None


def extract_progress_message(response: Dict[str, Any]) -> Optional[str]:
    """Return any human-readable progress message provided by the remote service."""
    for key in ("message", "status_text", "detail", "description"):
        message = response.get(key)
        if isinstance(message, str) and message.strip():
            return message
    progress = response.get("progress")
    if isinstance(progress, dict):
        for key in ("message", "detail", "description", "stage"):
            message = progress.get(key)
            if isinstance(message, str) and message.strip():
                return message
    return None


def extract_poll_url(response: Dict[str, Any], base_url: str | None = None) -> Optional[str]:
    """Retrieve a poll URL (absolute or relative) if the remote requests polling."""
    from urllib.parse import urljoin

    for key in ("poll_url", "status_url", "next", "poll" ):
        candidate = response.get(key)
        if isinstance(candidate, str) and candidate.strip():
            if base_url:
                return urljoin(base_url, candidate)
            return candidate
    return None


def extract_retry_after(response: Dict[str, Any], default: float) -> float:
    """Determine how long to wait before polling again."""
    for key in ("retry_in", "retry_after", "poll_interval", "interval"):
        value = response.get(key)
        if value is None:
            continue
        try:
            return max(float(value), 0.1)
        except (TypeError, ValueError):
            logger.debug("Invalid retry interval %s in response", value)
    return default
