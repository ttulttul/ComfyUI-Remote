"""Generated Modal workflow entrypoint."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import pickle
import sys
import uuid
from pathlib import Path
from typing import Any, Dict

import modal
from fastapi import Request

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional GPU detection for typing friendliness.
    import torch
except ModuleNotFoundError:  # pragma: no cover - runtime injects torch via ComfyUI requirements.
    torch = None  # type: ignore

APP_NAME = $APP_NAME
EXTRA_PIP_PACKAGES = $PIP_PACKAGES
EXTRA_SYSTEM_PACKAGES = $SYSTEM_PACKAGES
GPU_TYPE = $GPU_LITERAL

TORCH_VERSION = "2.3.0"
TORCHVISION_VERSION = "0.18.0"
TORCHAUDIO_VERSION = "2.3.0"
CUDA_WHEEL_INDEX = "https://download.pytorch.org/whl/cu121"
CPU_WHEEL_INDEX = "https://download.pytorch.org/whl/cpu"
XFORMERS_VERSION = "0.0.26.post1"

PROJECT_ROOT = Path(__file__).parent
PROMPT_PATH = PROJECT_ROOT / "prompt.json"
COMFY_ROOT = Path("/workspace/ComfyUI")


def _ensure_comfy_module_resolution() -> None:
    """Guarantee that ComfyUI's bundled packages shadow similarly named PyPI modules."""

    comfy_path = str(COMFY_ROOT)
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)

    existing_python_path = os.environ.get("PYTHONPATH")
    if not existing_python_path:
        os.environ["PYTHONPATH"] = comfy_path
    elif comfy_path not in existing_python_path.split(os.pathsep):
        os.environ["PYTHONPATH"] = os.pathsep.join([comfy_path, existing_python_path])

    preloaded_utils = sys.modules.get("utils")
    if preloaded_utils is None:
        return

    module_file = getattr(preloaded_utils, "__file__", "")
    try:
        module_path = Path(module_file).resolve()
    except Exception:  # pragma: no cover - defensive; some modules lack real paths.
        module_path = None

    if module_path and str(COMFY_ROOT) in str(module_path):
        return

    logger.debug(
        "Removing preloaded third-party 'utils' module (%s) so ComfyUI's package can be imported",
        module_file or "unknown",
    )
    sys.modules.pop("utils", None)


_ensure_comfy_module_resolution()

try:
    from comfy_api.latest import io  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard for missing Comfy.
    raise RuntimeError(
        "ComfyUI must be installed inside the Modal image; run modal deploy after building the image."
    ) from exc

SYSTEM_PACKAGES = ["git", "ffmpeg", *EXTRA_SYSTEM_PACKAGES]

TORCH_INDEX_URL = CUDA_WHEEL_INDEX if GPU_TYPE else CPU_WHEEL_INDEX
TORCH_INSTALL_COMMAND = (
    "pip install --no-cache-dir --force-reinstall --index-url "
    f"{TORCH_INDEX_URL} torch=={TORCH_VERSION} torchvision=={TORCHVISION_VERSION} torchaudio=={TORCHAUDIO_VERSION}"
)

base_commands = [
    "pip install --upgrade pip",
    "pip install --upgrade setuptools wheel",
    "test -d /workspace/ComfyUI || git clone --depth=1 https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI",
    "cd /workspace/ComfyUI && pip install -r requirements.txt",
    TORCH_INSTALL_COMMAND,
    "pip install fastapi",
]

if GPU_TYPE:
    base_commands.append(
        "pip install --no-cache-dir --no-deps --index-url "
        f"{CUDA_WHEEL_INDEX} xformers=={XFORMERS_VERSION}"
    )

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(*SYSTEM_PACKAGES)
    .run_commands(base_commands)
)

if EXTRA_PIP_PACKAGES:
    image = image.pip_install(*EXTRA_PIP_PACKAGES)

image = image.env({"PYTHONPATH": "/workspace/ComfyUI"})

function_kwargs: Dict[str, Any] = {"image": image, "min_containers": 1, "timeout": 900}

if GPU_TYPE:
    function_kwargs["gpu"] = str(GPU_TYPE)

app = modal.App(APP_NAME)


class RemoteInputNode(io.ComfyNode):
    """Receives payload injected by the remote HTTP bridge."""

    _payload_store: Dict[str, Dict[str, Any]] = {}

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
    def inject_payload(cls, node_id: str, payload: Dict[str, Any]) -> None:
        logger.debug("Injecting payload for RemoteInputNode %s", node_id)
        cls._payload_store[node_id] = payload

    @classmethod
    def pop_payload(cls, node_id: str) -> Dict[str, Any] | None:
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

    _output_store: Dict[str, Dict[str, Any]] = {}

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
        cls._register_output(cls.hidden.unique_id, key, value)
        return io.NodeOutput(value)

    @classmethod
    def pop_output(cls, node_id: str) -> Dict[str, Any] | None:
        return cls._output_store.pop(node_id, None)

    @classmethod
    def _register_output(cls, node_id: str, key: str, value: Any) -> None:
        bucket = cls._output_store.setdefault(node_id, {})
        bucket[key] = value
        logger.debug("Stored output for node %s key '%s'", node_id, key)


def infer_comfy_type(value: Any) -> str:
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


def serialize_value(name: str, value: Any) -> Dict[str, Any]:
    python_type = f"{type(value).__module__}.{type(value).__qualname__}"
    comfy_type = infer_comfy_type(value)
    encoded = base64.b64encode(pickle.dumps(value)).decode("utf-8")
    return {
        "name": name,
        "type": comfy_type,
        "python_type": python_type,
        "encoding": "base64+pickle",
        "data": encoded,
    }


def deserialize_value(payload: Dict[str, Any]) -> Any:
    encoding = payload.get("encoding", "base64+pickle")
    data = payload.get("data")
    if data is None:
        raise ValueError("Remote payload missing 'data' field")
    if encoding == "base64+pickle":
        return pickle.loads(base64.b64decode(data))
    if encoding == "identity":
        return data
    raise ValueError(f"Unsupported remote payload encoding: {encoding}")


class ComfyRuntime:
    """Minimal harness that runs a ComfyUI workflow from Modal requests."""

    def __init__(self) -> None:
        self._executor = None
        self._server = None
        self._initialized = False

    async def ensure_ready(self) -> None:
        if self._initialized:
            return

        import comfy.options  # type: ignore

        comfy.options.enable_args_parsing()

        from server import PromptServer  # type: ignore
        import execution  # type: ignore
        import nodes  # type: ignore

        loop = asyncio.get_running_loop()
        self._server = PromptServer(loop)
        self._executor = execution.PromptExecutor(self._server)
        self._register_remote_nodes(nodes)
        self._initialized = True
        logger.info("Comfy runtime initialised inside Modal worker")

    def _register_remote_nodes(self, nodes_module) -> None:
        input_schema = RemoteInputNode.define_schema()
        output_schema = RemoteOutputNode.define_schema()
        nodes_module.NODE_CLASS_MAPPINGS[input_schema.node_id] = RemoteInputNode
        nodes_module.NODE_CLASS_MAPPINGS[output_schema.node_id] = RemoteOutputNode
        nodes_module.NODE_DISPLAY_NAME_MAPPINGS[input_schema.node_id] = input_schema.display_name
        nodes_module.NODE_DISPLAY_NAME_MAPPINGS[output_schema.node_id] = output_schema.display_name

    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        await self.ensure_ready()

        prompt = json.loads(PROMPT_PATH.read_text())
        decoded_inputs = self._decode_inputs(payload)
        self._inject_payloads(prompt, decoded_inputs)

        prompt_id = payload.get("node", {}).get("id", uuid.uuid4().hex)
        extra_data = {}
        if "extra_pnginfo" in payload:
            extra_data["extra_pnginfo"] = payload["extra_pnginfo"]

        assert self._executor is not None
        await self._executor.execute_async(prompt, prompt_id, extra_data=extra_data)

        if not self._executor.success:
            raise RuntimeError("Comfy workflow reported failure; inspect status messages for details.")

        outputs = self._collect_outputs(prompt)
        if not outputs:
            raise RuntimeError("Remote workflow did not produce any Remote Output values.")

        serialized = serialize_value("remote_outputs", outputs)
        logger.info("Workflow completed for prompt %s with %d outputs", prompt_id, len(outputs))
        return {
            "status": "completed",
            "progress": {"value": 1, "max": 1},
            "result": serialized,
            "outputs": outputs,
        }

    def _decode_inputs(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        decoded: Dict[str, Any] = {}
        for item in payload.get("inputs", []):
            name = item.get("name")
            if not name:
                continue
            decoded[name] = deserialize_value(item)
        return decoded

    def _inject_payloads(self, prompt: Dict[str, Any], payload: Dict[str, Any]) -> None:
        for node_id, node_data in prompt.items():
            if node_data.get("class_type") == "ComfyRemoteInput":
                RemoteInputNode.inject_payload(node_id, payload)

    def _collect_outputs(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        collected: Dict[str, Any] = {}
        for node_id, node_data in prompt.items():
            if node_data.get("class_type") == "ComfyRemoteOutput":
                bucket = RemoteOutputNode.pop_output(node_id)
                if bucket:
                    collected.update(bucket)
        return collected


runtime = ComfyRuntime()


@app.function(**function_kwargs)
@modal.fastapi_endpoint(method="POST", label="infer-sync")
async def infer_sync(request: Request) -> Dict[str, Any]:
    payload = await request.json()
    try:
        return await runtime.run(payload)
    except Exception as exc:  # pragma: no cover - best effort to surface runtime errors.
        logger.exception("Remote workflow execution failed: %s", exc)
        return {
            "status": "failed",
            "message": str(exc),
        }
