"""ComfyUI Remote custom node extension."""

from __future__ import annotations

import logging
logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised when running inside ComfyUI.
    from typing import Type

    from typing_extensions import override

    from comfy_api.latest import ComfyExtension, io

    from .remote_node import RemoteAPINode

    class ComfyRemoteExtension(ComfyExtension):
        """Registers the remote REST bridge node with ComfyUI."""

        @override
        async def get_node_list(self) -> list[Type[io.ComfyNode]]:
            logger.debug("Registering ComfyUI Remote nodes")
            return [RemoteAPINode]


    async def comfy_entrypoint() -> ComfyRemoteExtension:
        """Entry point executed by ComfyUI when loading the extension."""
        logger.info("Loading ComfyUI Remote extension")
        return ComfyRemoteExtension()

except ModuleNotFoundError:  # pragma: no cover - triggered only in isolated test envs.
    logger.debug("Comfy API not available; comfy_entrypoint stub will raise when called.")

    async def comfy_entrypoint():
        raise RuntimeError(
            "comfy_api is not available outside a ComfyUI runtime; cannot load ComfyRemoteExtension."
        )
