"""Shared ComfyUI API compatibility layer for optional imports."""

from __future__ import annotations

from typing import Any

ExecutionType = Any

try:  # pragma: no cover - exercised when running inside ComfyUI.
    from comfy_api.latest import ComfyAPI, io  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - enables running unit tests without ComfyUI.
    class _DummyExecution:
        async def set_progress(self, *args, **kwargs) -> None:
            return None

    class _DummyAPI:
        def __init__(self) -> None:
            self.execution = _DummyExecution()

    class _DummySchema:
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - simple passthrough
            self.args = args
            self.kwargs = kwargs

    class _DummyIO:
        Schema = _DummySchema

        class ComfyNode:  # type: ignore[assignment]
            class hidden:  # type: ignore[assignment]
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


def get_execution_interface(api: Any) -> Any:
    """Return the execution helper for progress reporting.

    ComfyUI 0.3.68 exposes the execution helper as a lazily-populated singleton on the
    API instance. Older snapshots (and the unit-test stubs above) still expose the
    attribute eagerly. This helper smooths over both behaviours by instantiating the
    singleton when needed and caching it on *api* for subsequent calls.
    """

    if hasattr(api, "execution") and getattr(api, "execution") is not None:
        return api.execution

    execution_cls: Any = None
    # Newer releases keep the Execution class on the instance type; older ones may
    # expose it directly on the instance.
    for candidate in (getattr(type(api), "Execution", None), getattr(api, "Execution", None)):
        if candidate is not None:
            execution_cls = candidate
            break

    if execution_cls is None:
        raise AttributeError("ComfyAPI instance does not expose an Execution helper")

    execution: Any
    if hasattr(execution_cls, "get_instance"):
        execution = execution_cls.get_instance()
    else:
        execution = execution_cls()

    setattr(api, "execution", execution)
    return execution

__all__ = ["ComfyAPI", "io", "get_execution_interface"]
