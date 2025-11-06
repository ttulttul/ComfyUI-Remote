"""Remote ComfyUI node that proxies execution to an HTTP endpoint."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import asdict
from typing import Any, Awaitable, Dict, List, Tuple

import aiohttp
import numpy as np
from PIL import Image

try:  # ComfyUI runtime always has torch, but guard for tests.
    import torch
except ModuleNotFoundError:  # pragma: no cover - only triggered in limited envs.
    torch = None  # type: ignore

from comfy_api.latest import ComfyAPI, io
from comfy_api.latest._io import AutogrowDynamic
from server import PromptServer

from .protocol import (
    SerializedValue,
    deserialize_value,
    extract_poll_url,
    extract_progress,
    extract_progress_message,
    extract_retry_after,
    infer_comfy_type,
    normalise_result_envelope,
    serialize_inputs,
)

logger = logging.getLogger(__name__)

api = ComfyAPI()

DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_POLL_INTERVAL_SECONDS = 1.5
MAX_REMOTE_INPUTS = 8
_DYNAMIC_SUFFIX = "_$remote_inputs_ag$"
COMPLETED_STATES = {"completed", "complete", "done", "success"}
PENDING_STATES = {"pending", "queued", "running", "processing", "in_progress"}
FAILED_STATES = {"failed", "error", "cancelled", "aborted"}


class RemoteAPIError(RuntimeError):
    """Raised when the remote service returns an error response."""


class RemoteAPINode(io.ComfyNode):
    """ComfyUI node that forwards execution to a remote REST API."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        inputs: list = [
            io.String.Input(
                "url",
                display_name="Remote URL",
                tooltip="Base URL of the remote REST endpoint that will execute this node.",
                placeholder="https://example.com/api/task",
            ),
            io.Float.Input(
                "request_timeout",
                display_name="Timeout (s)",
                default=DEFAULT_TIMEOUT_SECONDS,
                min=1.0,
                max=3600.0,
                tooltip="Maximum seconds to wait for the initial request to respond.",
                optional=True,
            ),
            io.Float.Input(
                "poll_interval",
                display_name="Poll Interval (s)",
                default=DEFAULT_POLL_INTERVAL_SECONDS,
                min=0.1,
                max=120.0,
                tooltip="How often to poll the remote status endpoint when running async tasks.",
                optional=True,
            ),
            AutogrowDynamic.Input(
                id="remote_inputs",
                template_input=io.AnyType.Input(
                    "input_",
                    display_name="Input ",
                    tooltip="Value forwarded to the remote API. At least one input must be connected.",
                ),
                min=1,
                max=MAX_REMOTE_INPUTS,
                display_name="Inputs",
            ),
        ]

        outputs = [
            io.AnyType.Output(
                "result",
                display_name="Result",
                tooltip="Deserialized output returned by the remote API.",
            )
        ]

        return io.Schema(
            node_id="ComfyRemoteNode",
            display_name="Remote API",
            category="utils/remote",
            inputs=inputs,
            outputs=outputs,
            description="Send inputs to a remote REST endpoint that can stream progress and return a result.",
        )

    @classmethod
    async def execute(
        cls,
        url: str,
        request_timeout: float,
        poll_interval: float,
        **remote_inputs: Any,
    ) -> io.NodeOutput:
        if not url:
            raise ValueError("The remote URL is required.")

        timeout_seconds = float(request_timeout) if request_timeout else DEFAULT_TIMEOUT_SECONDS
        poll_seconds = float(poll_interval) if poll_interval else DEFAULT_POLL_INTERVAL_SECONDS

        ordered_inputs = cls._ordered_remote_inputs(remote_inputs)
        if not ordered_inputs:
            raise ValueError("At least one input must be connected to the Remote API node.")

        serialized_inputs = serialize_inputs(ordered_inputs)
        payload = cls._build_payload(serialized_inputs, ordered_inputs)

        logger.info(
            "Dispatching %d inputs to remote endpoint %s (timeout=%.2fs)",
            len(serialized_inputs),
            url,
            timeout_seconds,
        )

        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                initial = await cls._perform_request(session, url, payload)
                result = await cls._await_completion(
                    session,
                    initial,
                    url,
                    poll_seconds,
                )
        except aiohttp.ClientError as exc:
            logger.error("Network error contacting remote API: %s", exc)
            raise RemoteAPIError(f"Network error contacting remote API: {exc}") from exc

        logger.info("Remote execution completed for node %s", cls.hidden.unique_id)
        await api.execution.set_progress(1.0, 1.0, node_id=cls.hidden.unique_id)
        return io.NodeOutput(result)

    @classmethod
    def _build_payload(
        cls,
        serialized_inputs: List[SerializedValue],
        ordered_inputs: List[Tuple[str, Any]],
    ) -> Dict[str, Any]:
        payload = {
            "node": {
                "id": cls.hidden.unique_id,
                "name": cls.__name__,
            },
            "inputs": [asdict(value) for value in serialized_inputs],
            "input_order": [name for name, _ in ordered_inputs],
        }
        # Provide light-weight context so remote services can reason about the workflow when needed.
        if cls.hidden.prompt is not None:
            prompt_obj = cls.hidden.prompt
            if isinstance(prompt_obj, dict):
                payload["workflow"] = prompt_obj.get("workflow", prompt_obj)
            else:
                payload["workflow"] = prompt_obj
        if cls.hidden.extra_pnginfo is not None:
            payload["extra_pnginfo"] = cls.hidden.extra_pnginfo
        return payload

    @classmethod
    def _ordered_remote_inputs(cls, inputs: Dict[str, Any]) -> List[Tuple[str, Any]]:
        ordered: list[tuple[int, str, Any]] = []
        for raw_name, value in inputs.items():
            if not raw_name.endswith(_DYNAMIC_SUFFIX):
                ordered.append((len(ordered), raw_name, value))
                continue
            base = raw_name[: -len(_DYNAMIC_SUFFIX)]
            match = re.search(r"(\d+)$", base)
            index = int(match.group(1)) if match else len(ordered)
            ordered_name = f"input_{index}"
            ordered.append((index, ordered_name, value))
        ordered.sort(key=lambda item: item[0])
        return [(name, value) for _, name, value in ordered if value is not None]

    @classmethod
    async def _perform_request(
        cls,
        session: aiohttp.ClientSession,
        url: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        logger.debug("Posting payload to %s", url)
        response = await cls._request_json(session, "POST", url, payload)
        cls._handle_progress_update(response)
        return response

    @classmethod
    async def _await_completion(
        cls,
        session: aiohttp.ClientSession,
        response: Dict[str, Any],
        base_url: str,
        poll_seconds: float,
    ) -> Any:
        current = response
        last_progress: tuple[float, float] | None = None
        while True:
            status = cls._normalise_status(current)
            logger.debug("Remote status response: %s", status or "<missing>")
            cls._handle_progress_update(current, last_progress)
            progress_pair = extract_progress(current)
            if progress_pair:
                last_progress = progress_pair
            if status in COMPLETED_STATES or (status is None and "data" in normalise_result_envelope(current)):
                result_payload = cls._coerce_result_payload(current)
                return cls._deserialize_output(result_payload)
            if status in FAILED_STATES:
                message = cls._extract_error_message(current)
                raise RemoteAPIError(message)
            if status not in PENDING_STATES and status not in COMPLETED_STATES:
                # Treat unknown status that includes a result payload as success.
                result_payload = cls._maybe_extract_result(current)
                if result_payload is not None:
                    return cls._deserialize_output(result_payload)
                logger.warning("Remote responded with unknown status '%s'", status)
            poll_url = extract_poll_url(current, base_url)
            if not poll_url:
                raise RemoteAPIError("Remote API did not provide a poll URL for pending task state.")
            delay = extract_retry_after(current, poll_seconds)
            logger.debug("Polling %s again in %.2fs", poll_url, delay)
            await asyncio.sleep(delay)
            current = await cls._request_json(session, "GET", poll_url)

    @classmethod
    def _handle_progress_update(
        cls,
        response: Dict[str, Any],
        previous: tuple[float, float] | None = None,
    ) -> None:
        progress_pair = extract_progress(response)
        if progress_pair and progress_pair != previous:
            value, total = progress_pair
            cls._schedule_progress_update(
                api.execution.set_progress(
                    value=value,
                    max_value=total,
                    node_id=cls.hidden.unique_id,
                )
            )
        message = extract_progress_message(response)
        if message:
            PromptServer.instance.send_progress_text(message, cls.hidden.unique_id)

    @staticmethod
    def _schedule_progress_update(coro: Awaitable[None]) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("No running loop available; skipping progress update")
            return

        task = loop.create_task(coro)

        def _log_exception(task: asyncio.Task) -> None:
            try:
                task.result()
            except Exception as exc:  # pragma: no cover - logging only.
                logger.error("Progress update failed: %s", exc)

        task.add_done_callback(_log_exception)

    @classmethod
    def _coerce_result_payload(cls, response: Dict[str, Any]) -> Dict[str, Any]:
        payload = normalise_result_envelope(response)
        if "data" not in payload and "value" in payload:
            payload = {
                **payload,
                "data": payload["value"],
                "encoding": payload.get("encoding", "identity"),
            }
        return payload

    @classmethod
    def _maybe_extract_result(cls, response: Dict[str, Any]) -> Dict[str, Any] | None:
        try:
            payload = cls._coerce_result_payload(response)
        except Exception:
            return None
        if "data" in payload:
            return payload
        return None

    @classmethod
    def _deserialize_output(cls, payload: Dict[str, Any]) -> Any:
        declared_type = payload.get("type") or payload.get("comfy_type")
        python_type = payload.get("python_type")
        value = deserialize_value(payload)
        comfy_type = declared_type or infer_comfy_type(value)
        logger.debug(
            "Remote result declared type=%s python=%s", declared_type, python_type
        )
        return cls._coerce_to_comfy_type(value, comfy_type)

    @classmethod
    def _coerce_to_comfy_type(cls, value: Any, comfy_type: str) -> Any:
        if torch is None:
            return value
        type_upper = comfy_type.upper() if isinstance(comfy_type, str) else "ANY"
        if type_upper == "IMAGE":
            return cls._ensure_image_tensor(value)
        if type_upper == "MASK":
            return cls._ensure_mask_tensor(value)
        if type_upper == "LATENT":
            return cls._ensure_tensor(value)
        return value

    @staticmethod
    def _ensure_image_tensor(value: Any) -> Any:
        tensor = RemoteAPINode._ensure_tensor(value)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise RemoteAPIError(
                f"Image tensor must have 4 dimensions (B, H, W, C); received shape {tuple(tensor.shape)}"
            )
        return tensor

    @staticmethod
    def _ensure_mask_tensor(value: Any) -> Any:
        tensor = RemoteAPINode._ensure_tensor(value)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(-1)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise RemoteAPIError(
                f"Mask tensor must broadcast to (B, H, W, 1); received shape {tuple(tensor.shape)}"
            )
        return tensor

    @staticmethod
    def _ensure_tensor(value: Any) -> "torch.Tensor":
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        if isinstance(value, Image.Image):
            arr = np.asarray(value).astype("float32") / 255.0
            tensor = torch.from_numpy(arr)
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(-1)
            return tensor
        if isinstance(value, list):
            return torch.tensor(value)
        raise RemoteAPIError(f"Cannot coerce value of type {type(value)} into a tensor")

    @classmethod
    def _extract_error_message(cls, response: Dict[str, Any]) -> str:
        for key in ("error", "detail", "message", "status_text"):
            message = response.get(key)
            if isinstance(message, str) and message.strip():
                return message
        progress_msg = extract_progress_message(response)
        if progress_msg:
            return progress_msg
        return "Remote API reported failure without an error message."

    @classmethod
    def _normalise_status(cls, response: Dict[str, Any]) -> str | None:
        status = response.get("status") or response.get("state")
        if isinstance(status, str):
            return status.lower()
        return None

    @classmethod
    async def _request_json(
        cls,
        session: aiohttp.ClientSession,
        method: str,
        url: str,
        payload: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        logger.debug("%s %s", method.upper(), url)
        async with session.request(method.upper(), url, json=payload) as response:
            text = await response.text()
            if response.status >= 400:
                logger.error("Remote API responded with %s: %s", response.status, text)
                raise RemoteAPIError(
                    f"Remote API responded with status {response.status}: {text.strip() or 'No details provided.'}"
                )
            if not text.strip():
                return {}
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                logger.error("Invalid JSON from remote API: %s", text)
                raise RemoteAPIError("Remote API did not return valid JSON.") from exc
