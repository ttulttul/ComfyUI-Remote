# ComfyUI Remote Nodes

ComfyUI Remote is a collection of custom nodes that make it easier to run ComfyUI workflows on remote infrastructure such as Modal and call them through a RESTful bridge.

## Nodes

| Node | Purpose |
| --- | --- |
| `Remote Input` (`ComfyRemoteInput`) | Marks entry points where JSON payload data from the REST caller flows into the graph. |
| `Remote Output` (`ComfyRemoteOutput`) | Marks exit points that capture values to return to the caller. |
| `Modal Deployment` (`ComfyRemoteModalDeployment`) | Converts a local ComfyUI workflow JSON into a Modal project, optionally invoking `modal deploy` and returning the serving URL. |
| `Remote API` (`ComfyRemoteNode`) | Sends inputs to a REST endpoint, streams progress updates, and deserialises the response for use inside ComfyUI. |

## Typical Flow

1. Add `Remote Input` nodes where you expect data from the remote request.
2. Add `Remote Output` nodes for the values you want to return.
3. Export the workflow JSON and run the `Modal Deployment` node, pointing it at the JSON file. This generates a Modal project scaffold (`prompt.json`, `modal_config.json`, `workflow.py`) and, when `deploy=True`, calls `modal deploy` to produce a public endpoint URL.
4. Feed the returned URL into the `Remote API` node. Connect its inputs to the data you want to send to the remote workflow and its output to the downstream nodes in ComfyUI.

Once connected, execution looks like:

```
[Local Graph Inputs] -> Remote API --HTTP--> Modal Service -> Remote workflow -> Remote Output -> Remote API (response) -> [Local Graph]
```

Progress coming back from the remote service is forwarded to ComfyUI's progress bar and console.

## Modal Prerequisites

- Install the Modal CLI (`pip install modal-client`) and authenticate (`modal token new`).
- The `Modal Deployment` node now emits a self-contained Modal app that clones the upstream [ComfyUI](https://github.com/comfyanonymous/ComfyUI) repository inside the container—no external worker template required.
- Optional pip dependencies and Debian packages can be declared through the node inputs; they are recorded in `modal_config.json` alongside the generated `workflow.py`.
- GPU builds automatically install CUDA-enabled PyTorch wheels (`torch==2.3.0`, `torchvision==0.18.0`, `torchaudio==2.3.0` from the cu121 index) with `--force-reinstall`, add `xformers==0.0.26.post1` via `pip install --no-deps` so Torch stays pinned, and export `/workspace/ComfyUI` on `PYTHONPATH` so ComfyUI’s `utils` package resolves correctly. Leaving the `Modal GPU` input blank defaults to an `H100`; set it explicitly if you need a different accelerator.

## Remote Protocol

The `Remote API` node serialises each connected input by:

- Pickling the Python object and base64 encoding it.
- Including metadata about the inferred ComfyUI type (`IMAGE`, `MASK`, etc.).

The remote endpoint should respond with JSON that contains either:

```json
{
  "status": "completed",
  "progress": {"value": 1, "max": 1},
  "result": {
    "type": "IMAGE",
    "encoding": "base64+pickle",
    "data": "..."
  }
}
```

- `status`: Optional but recommended; recognised values include `pending`, `running`, `completed`, and `failed`.
- `progress`: Can be a dict (`value/max`, `percent`, `ratio`) or a single number. Progress updates are forwarded to the ComfyUI progress bar.
- `poll_url`: Optional string for asynchronous operations. When present alongside a non-completed status, the node polls this URL until completion.

## Remote Input/Output Helpers

When the remote workflow starts it should:

1. Call `RemoteInputNode.inject_payload(node_id, payload_dict)` for each registered input node before the graph executes.
2. After the graph runs, call `RemoteOutputNode.pop_output(node_id)` to gather the response payload and return it via the REST interface.

These helpers are designed to be used inside the Modal worker's request handler.

## Running Tests

```
venv/bin/python -m unittest discover tests
```

The tests cover protocol serialization, Modal project scaffolding, and the input/output payload helpers.

## Contributing

- Keep new files ASCII unless the project already uses other encodings.
- Update `docs/LEARNINGS.md` when you discover important behaviours or caveats.
- Prefer logging via `logging.getLogger(__name__)` in new modules.
