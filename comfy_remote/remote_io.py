"""Remote input/output node definitions shared between runtime environments."""

from __future__ import annotations

import logging
from typing import Any

from ._comfy import io

logger = logging.getLogger(__name__)


class RemoteInputNode(io.ComfyNode):
    """Receives payload injected by the remote HTTP bridge."""

    _payload_store: dict[str, dict[str, Any]] = {}

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ComfyRemoteInput",
            display_name="Remote Input",
            category="utils/remote",
            inputs=[
                io.String.Input(
                    "key",
                    display_name="Payload Key",
                    default="payload",
                    tooltip="Name of the field in the remote JSON payload to read.",
                ),
                io.AnyType.Input(
                    "fallback",
                    display_name="Fallback Value",
                    optional=True,
                    tooltip="Local fallback value used when no remote payload was provided.",
                ),
            ],
            outputs=[io.AnyType.Output("value", display_name="Value")],
            description="Entry point for remote data supplied by the Comfy Remote API node.",
        )

    @classmethod
    def execute(cls, key: str, fallback: Any | None = None) -> io.NodeOutput:
        value = cls._resolve_payload(cls.hidden.unique_id, key, fallback)
        return io.NodeOutput(value)

    @classmethod
    def inject_payload(cls, node_id: str, payload: dict[str, Any]) -> None:
        logger.debug("Injecting payload for RemoteInputNode %s", node_id)
        cls._payload_store[node_id] = payload

    @classmethod
    def pop_payload(cls, node_id: str) -> dict[str, Any] | None:
        return cls._payload_store.pop(node_id, None)

    @classmethod
    def _resolve_payload(cls, node_id: str, key: str, fallback: Any | None) -> Any:
        payload = cls._payload_store.get(node_id)
        if payload is None:
            logger.debug(
                "No payload injected for RemoteInputNode %s; using fallback for key '%s'",
                node_id,
                key,
            )
            return fallback
        if key not in payload:
            logger.debug(
                "Injected payload for node %s missing key '%s'; using fallback", node_id, key
            )
            return fallback
        return payload[key]


class RemoteOutputNode(io.ComfyNode):
    """Captures payload that needs to be sent back to the remote caller."""

    _output_store: dict[str, dict[str, Any]] = {}
    OUTPUT_NODE = True

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ComfyRemoteOutput",
            display_name="Remote Output",
            category="utils/remote",
            inputs=[
                io.AnyType.Input(
                    "value",
                    display_name="Value",
                    tooltip="Value to send back to the remote caller.",
                ),
                io.String.Input(
                    "key",
                    display_name="Payload Key",
                    default="result",
                    tooltip="Name of the field to use in the remote response payload.",
                ),
            ],
            outputs=[
                io.AnyType.Output(
                    "passthrough",
                    display_name="Value",
                    tooltip="Pass-through output so this node can continue the graph.",
                )
            ],
            description="Terminal node marking the value to be returned from the remote workflow.",
        )

    @classmethod
    def execute(cls, value: Any, key: str) -> io.NodeOutput:
        logger.info(
            "RemoteOutputNode %s storing key '%s' (%s)",
            cls.hidden.unique_id,
            key,
            type(value).__name__,
        )
        cls._register_output(cls.hidden.unique_id, key, value)
        return io.NodeOutput(value)

    @classmethod
    def pop_output(cls, node_id: str) -> dict[str, Any] | None:
        return cls._output_store.pop(node_id, None)

    @classmethod
    def _register_output(cls, node_id: str, key: str, value: Any) -> None:
        bucket = cls._output_store.setdefault(node_id, {})
        bucket[key] = value
        logger.debug("Stored output for node %s key '%s'", node_id, key)


__all__ = ["RemoteInputNode", "RemoteOutputNode"]
