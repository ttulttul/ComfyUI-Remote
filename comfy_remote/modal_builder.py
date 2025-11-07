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

from ._comfy import ComfyAPI, get_execution_interface, io

try:  # pragma: no cover - PromptServer only exists within a ComfyUI runtime.
    from server import PromptServer
except ModuleNotFoundError:  # pragma: no cover - unit tests run without the ComfyUI server.
    PromptServer = None  # type: ignore

logger = logging.getLogger(__name__)

api = ComfyAPI()

MODAL_URL_REGEX = re.compile(r"https://[\w.-]+\.modal\.run[\w/\-]*")
DEFAULT_APP_NAME = "comfy-remote"


@dataclass
class ModalProjectPaths:
    root: Path
    prompt: Path
    workflow: Path
    config: Path


class ModalDeploymentError(RuntimeError):
    """Raised when Modal deployment cannot be completed."""


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
                    "working_directory",
                    display_name="Build Directory",
                    optional=True,
                    tooltip="Optional directory where build artifacts should be written.",
                ),
                io.String.Input(
                    "extra_pip_requirements",
                    display_name="Extra Pip Packages",
                    optional=True,
                    tooltip="Optional newline- or comma-separated pip packages installed on top of ComfyUI's requirements.",
                ),
                io.String.Input(
                    "extra_system_packages",
                    display_name="Extra Apt Packages",
                    optional=True,
                    tooltip="Optional newline- or comma-separated Debian packages appended to the build image.",
                ),
                io.String.Input(
                    "gpu_type",
                    display_name="Modal GPU",
                    optional=True,
                    tooltip="Optional Modal GPU type, e.g. 'A10G' or 'L4'. Leave blank for CPU-only execution.",
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
                io.Boolean.Input(
                    "clean_build",
                    display_name="Clean Build",
                    default=False,
                    tooltip="Delete the existing Modal build directory before regenerating project files.",
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
        working_directory: str | None = None,
        extra_pip_requirements: str | None = None,
        extra_system_packages: str | None = None,
        gpu_type: str | None = None,
        deploy: bool = True,
        dry_run: bool = False,
        clean_build: bool = False,
    ) -> io.NodeOutput:
        app_name = app_name or DEFAULT_APP_NAME

        workflow_file = Path(workflow_path).expanduser().resolve()
        if not workflow_file.exists():
            raise ModalDeploymentError(f"Workflow file '{workflow_file}' does not exist.")

        pip_packages = cls._parse_package_list(extra_pip_requirements)
        system_packages = cls._parse_package_list(extra_system_packages)
        gpu_choice = gpu_type.strip() if gpu_type else "H100"

        execution_api = get_execution_interface(api)

        await execution_api.set_progress(0.05, 1.0, node_id=cls.hidden.unique_id)
        cls._send_ui_update("Preparing Modal project scaffold...")
        project_paths = cls._prepare_modal_project(
            workflow_file=workflow_file,
            app_name=app_name,
            working_directory=working_directory,
            pip_packages=pip_packages,
            system_packages=system_packages,
            gpu_type=gpu_choice,
            clean_build=bool(clean_build),
        )

        await execution_api.set_progress(0.25, 1.0, node_id=cls.hidden.unique_id)
        cls._send_ui_update(f"Modal project files generated at {project_paths.root}")

        if not deploy or dry_run:
            placeholder_url = f"https://{app_name}.generated.modal"
            logger.info(
                "Dry run or deployment disabled; returning placeholder URL %s", placeholder_url
            )
            cls._send_ui_update(
                "Dry run selected; returning placeholder URL without deploying to Modal."
            )
            return io.NodeOutput(placeholder_url)

        await execution_api.set_progress(0.4, 1.0, node_id=cls.hidden.unique_id)
        cls._send_ui_update(
            "Running modal deploy – this can take a few minutes while dependencies install..."
        )
        service_url = await cls._run_modal_deploy(project_paths.workflow)
        await execution_api.set_progress(1.0, 1.0, node_id=cls.hidden.unique_id)
        cls._send_ui_update(f"Modal deployment completed: {service_url}")
        return io.NodeOutput(service_url)

    @classmethod
    def _prepare_modal_project(
        cls,
        workflow_file: Path,
        app_name: str,
        working_directory: str | None,
        pip_packages: list[str],
        system_packages: list[str],
        gpu_type: str | None,
        clean_build: bool,
    ) -> ModalProjectPaths:
        target_root = cls._resolve_target_directory(workflow_file, working_directory, app_name)
        if clean_build and target_root.exists():
            logger.info("Removing existing Modal project directory %s", target_root)
            shutil.rmtree(target_root)
        target_root.mkdir(parents=True, exist_ok=True)

        prompt_path = target_root / "prompt.json"
        config_path = target_root / "modal_config.json"
        workflow_py = target_root / "workflow.py"

        shutil.copyfile(workflow_file, prompt_path)
        logger.debug("Copied workflow JSON to %s", prompt_path)

        config_payload = {
            "app_name": app_name,
            "extra_pip_packages": pip_packages,
            "extra_system_packages": system_packages,
            "gpu_type": gpu_type or "H100",
        }
        config_path.write_text(json.dumps(config_payload, indent=2))
        logger.debug("Wrote Modal config to %s", config_path)

        workflow_py.write_text(
            cls._render_workflow_template(
                app_name=app_name,
                pip_packages=pip_packages,
                system_packages=system_packages,
                gpu_type=gpu_type,
            )
        )
        logger.debug("Generated workflow.py template at %s", workflow_py)

        return ModalProjectPaths(
            root=target_root,
            prompt=prompt_path,
            workflow=workflow_py,
            config=config_path,
        )

    @classmethod
    def _resolve_target_directory(
        cls, workflow_file: Path, working_directory: str | None, app_name: str
    ) -> Path:
        if working_directory:
            return Path(working_directory).expanduser().resolve() / app_name
        return workflow_file.parent / f"modal_{app_name}"

    @classmethod
    def _parse_package_list(cls, raw: str | None) -> list[str]:
        if not raw:
            return []
        entries: list[str] = []
        normalized = raw.replace("\r", "\n")
        for line in normalized.split("\n"):
            for token in line.split(","):
                value = token.strip()
                if value:
                    entries.append(value)
        return entries

    @classmethod
    def _render_workflow_template(
        cls,
        app_name: str,
        pip_packages: list[str],
        system_packages: list[str],
        gpu_type: str | None,
    ) -> str:
        template_path = Path(__file__).resolve().parent / "templates" / "modal_workflow.py.tpl"
        if not template_path.exists():
            raise ModalDeploymentError(f"Modal workflow template missing at {template_path}")

        from string import Template

        template = Template(template_path.read_text())
        substitutions = {
            "APP_NAME": repr(app_name),
            "PIP_PACKAGES": repr(pip_packages or []),
            "SYSTEM_PACKAGES": repr(system_packages or []),
            "GPU_LITERAL": repr(gpu_type),
        }
        return template.substitute(substitutions)

    @classmethod
    async def _run_modal_deploy(cls, workflow_py: Path) -> str:
        command = ["modal", "deploy", str(workflow_py)]
        logger.info("Running Modal deploy: %s", " ".join(command))

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout_task = asyncio.create_task(
            cls._stream_subprocess_output(process.stdout, "stdout")
        )
        stderr_task = asyncio.create_task(
            cls._stream_subprocess_output(process.stderr, "stderr")
        )

        returncode = await process.wait()
        stdout = await stdout_task
        stderr = await stderr_task

        if returncode != 0:
            logger.error("Modal deploy failed with code %s: %s", returncode, stderr)
            raise ModalDeploymentError(
                f"Modal deploy failed with exit code {returncode}: {stderr.strip() or stdout.strip()}"
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

    @classmethod
    def _send_ui_update(cls, message: str) -> None:
        logger.info(message)
        if PromptServer is None:
            return
        try:
            if PromptServer.instance is not None:
                PromptServer.instance.send_progress_text(message, cls.hidden.unique_id)
        except Exception:  # pragma: no cover - best effort only.
            logger.debug("Failed to push progress message to UI", exc_info=True)

    @classmethod
    async def _stream_subprocess_output(
        cls, stream: asyncio.StreamReader | None, label: str
    ) -> str:
        if stream is None:
            return ""
        collected: list[str] = []
        while True:
            line = await stream.readline()
            if not line:
                break
            decoded = line.decode("utf-8", "ignore")
            collected.append(decoded)
            stripped = decoded.strip()
            if stripped:
                cls._send_ui_update(f"[modal {label}] {stripped}")
        return "".join(collected)


__all__ = [
    "ModalDeploymentNode",
    "ModalDeploymentError",
]
