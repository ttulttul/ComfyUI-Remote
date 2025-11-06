"""Modal deployment node and remote workflow scaffolding helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

try:
    from comfy_api.latest import ComfyAPI, io
except ModuleNotFoundError:  # pragma: no cover - enables running unit tests without ComfyUI.
    class _DummyExecution:
        async def set_progress(self, *args, **kwargs) -> None:
            return None

    class _DummyAPI:
        def __init__(self) -> None:
            self.execution = _DummyExecution()

    class _DummySchema:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    class _DummyIO:
        Schema = _DummySchema

        class ComfyNode:
            class hidden:  # type: ignore
                unique_id = "dummy-node"

        class String:
            class Input:
                def __init__(self, *args, **kwargs) -> None:
                    pass

            class Output:
                def __init__(self, *args, **kwargs) -> None:
                    pass

        class Boolean:
            class Input:
                def __init__(self, *args, **kwargs) -> None:
                    pass

        class AnyType:
            class Input:
                def __init__(self, *args, **kwargs) -> None:
                    pass

            class Output:
                def __init__(self, *args, **kwargs) -> None:
                    pass

    ComfyAPI = _DummyAPI  # type: ignore
    io = _DummyIO()  # type: ignore

logger = logging.getLogger(__name__)

api = ComfyAPI()

MODAL_URL_REGEX = re.compile(r"https://[\w.-]+\.modal\.run[\w/\-]*")
DEFAULT_APP_NAME = "comfy-remote"
DEFAULT_REPOSITORY_URL = "https://github.com/tolgahanuzun/modal-comfy-worker.git"


@dataclass
class ModalProjectPaths:
    root: Path
    prompt: Path
    snapshot: Path
    workflow: Path


class ModalDeploymentError(RuntimeError):
    """Raised when Modal deployment cannot be completed."""


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
            outputs=[io.AnyType.Output("value", display_name="Value")],
            description="Terminal node marking the value to be returned from the remote workflow.",
        )

    @classmethod
    def execute(cls, value: Any, key: str) -> io.NodeOutput:
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


class ModalDeploymentNode(io.ComfyNode):
    """Builds and optionally deploys a Modal app for a ComfyUI workflow."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ComfyRemoteModalDeployment",
            display_name="Modal Deployment",
            category="utils/remote",
            inputs=[
                io.String.Input(
                    "workflow_path",
                    display_name="Workflow Path",
                    tooltip="Path to the ComfyUI workflow JSON that should be deployed.",
                ),
                io.String.Input(
                    "app_name",
                    display_name="Modal App Name",
                    default=DEFAULT_APP_NAME,
                    tooltip="Name of the Modal app to create/update.",
                    optional=True,
                ),
                io.String.Input(
                    "repository_url",
                    display_name="Modal Worker Repo",
                    default=DEFAULT_REPOSITORY_URL,
                    tooltip="Git repository providing the Modal Comfy worker template.",
                    optional=True,
                ),
                io.String.Input(
                    "working_directory",
                    display_name="Build Directory",
                    optional=True,
                    tooltip="Optional directory where build artifacts should be written.",
                ),
                io.Boolean.Input(
                    "deploy",
                    display_name="Deploy",
                    default=True,
                    tooltip="When enabled the node runs `modal deploy` and returns the live service URL.",
                ),
                io.Boolean.Input(
                    "dry_run",
                    display_name="Dry Run",
                    default=False,
                    tooltip="Skip modal CLI invocation and return an illustrative service URL.",
                    optional=True,
                ),
            ],
            outputs=[
                io.String.Output(
                    "service_url",
                    display_name="Service URL",
                    tooltip="The Modal endpoint URL hosting the remote workflow.",
                )
            ],
            description=(
                "Generate a Modal project from a ComfyUI workflow, deploy it, and return the "
                "produced service URL. This pairs with the Remote API node to execute workflows "
                "in a remote environment."
            ),
        )

    @classmethod
    async def execute(
        cls,
        workflow_path: str,
        app_name: str | None = None,
        repository_url: str | None = None,
        working_directory: str | None = None,
        deploy: bool = True,
        dry_run: bool = False,
    ) -> io.NodeOutput:
        app_name = app_name or DEFAULT_APP_NAME
        repository_url = repository_url or DEFAULT_REPOSITORY_URL

        workflow_file = Path(workflow_path).expanduser().resolve()
        if not workflow_file.exists():
            raise ModalDeploymentError(f"Workflow file '{workflow_file}' does not exist.")

        await api.execution.set_progress(0.05, 1.0, node_id=cls.hidden.unique_id)
        project_paths = cls._prepare_modal_project(
            workflow_file=workflow_file,
            repository_url=repository_url,
            app_name=app_name,
            working_directory=working_directory,
        )

        await api.execution.set_progress(0.25, 1.0, node_id=cls.hidden.unique_id)

        if not deploy or dry_run:
            placeholder_url = f"https://{app_name}.generated.modal"
            logger.info(
                "Dry run or deployment disabled; returning placeholder URL %s", placeholder_url
            )
            return io.NodeOutput(placeholder_url)

        await api.execution.set_progress(0.4, 1.0, node_id=cls.hidden.unique_id)
        service_url = await cls._run_modal_deploy(project_paths.workflow)
        await api.execution.set_progress(1.0, 1.0, node_id=cls.hidden.unique_id)
        return io.NodeOutput(service_url)

    @classmethod
    def _prepare_modal_project(
        cls,
        workflow_file: Path,
        repository_url: str,
        app_name: str,
        working_directory: str | None,
    ) -> ModalProjectPaths:
        target_root = cls._resolve_target_directory(workflow_file, working_directory, app_name)
        target_root.mkdir(parents=True, exist_ok=True)

        prompt_path = target_root / "prompt.json"
        snapshot_path = target_root / "snapshot.json"
        workflow_py = target_root / "workflow.py"

        shutil.copyfile(workflow_file, prompt_path)
        logger.debug("Copied workflow JSON to %s", prompt_path)

        snapshot_data = cls._generate_snapshot_stub(repository_url)
        snapshot_path.write_text(json.dumps(snapshot_data, indent=2))
        logger.debug("Wrote snapshot stub to %s", snapshot_path)

        workflow_py.write_text(cls._render_workflow_template(app_name))
        logger.debug("Generated workflow.py template at %s", workflow_py)

        return ModalProjectPaths(root=target_root, prompt=prompt_path, snapshot=snapshot_path, workflow=workflow_py)

    @classmethod
    def _resolve_target_directory(
        cls, workflow_file: Path, working_directory: str | None, app_name: str
    ) -> Path:
        if working_directory:
            return Path(working_directory).expanduser().resolve() / app_name
        return workflow_file.parent / f"modal_{app_name}"

    @classmethod
    def _generate_snapshot_stub(cls, repository_url: str) -> Dict[str, Any]:
        return {
            "repositories": [
                {
                    "name": "modal-comfy-worker",
                    "url": repository_url,
                }
            ],
            "pips": [],
            "file_custom_nodes": [],
        }

    @classmethod
    def _render_workflow_template(cls, app_name: str) -> str:
        return f'''"""Generated Modal workflow entrypoint."""

import json
from pathlib import Path

import modal

APP_NAME = {app_name!r}
PROMPT_PATH = Path(__file__).parent / "prompt.json"
SNAPSHOT_PATH = Path(__file__).parent / "snapshot.json"

app = modal.App(APP_NAME)

# NOTE: This image assumes the modal-comfy-worker repository is installed inside the container.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install("uv")
    .run_commands(
        [
            "test -d /workspace/modal-comfy-worker || git clone --depth=1 https://github.com/tolgahanuzun/modal-comfy-worker.git /workspace/modal-comfy-worker",
            "cd /workspace/modal-comfy-worker && uv pip install -r requirements.txt",
        ]
    )
)

@app.function(image=image, keep_warm=1, timeout=900)
@modal.web_endpoint(methods=["POST"], label="infer-sync")
def infer_sync(request: modal.web.Request):
    """Serve the Comfy workflow using the modal-comfy-worker project."""
    from modal_comfy_worker import api as comfy_worker_api

    payload = request.json()
    prompt = json.loads(PROMPT_PATH.read_text())
    snapshot = json.loads(SNAPSHOT_PATH.read_text())

    result = comfy_worker_api.run_workflow(
        prompt=prompt,
        snapshot=snapshot,
        request_payload=payload,
    )
    return result
'''

    @classmethod
    async def _run_modal_deploy(cls, workflow_py: Path) -> str:
        command = ["modal", "deploy", str(workflow_py)]
        logger.info("Running Modal deploy: %s", " ".join(command))

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await process.communicate()
        stdout = stdout_bytes.decode("utf-8", "ignore")
        stderr = stderr_bytes.decode("utf-8", "ignore")

        if process.returncode != 0:
            logger.error("Modal deploy failed with code %s: %s", process.returncode, stderr)
            raise ModalDeploymentError(
                f"Modal deploy failed with exit code {process.returncode}: {stderr.strip() or stdout.strip()}"
            )

        match = MODAL_URL_REGEX.search(stdout)
        if not match:
            logger.debug("Modal deploy stdout: %s", stdout)
            raise ModalDeploymentError(
                "Modal deploy completed but no service URL was detected in the output."
            )
        service_url = match.group(0)
        logger.info("Modal deployment available at %s", service_url)
        return service_url


__all__ = [
    "ModalDeploymentNode",
    "ModalDeploymentError",
    "RemoteInputNode",
    "RemoteOutputNode",
]
