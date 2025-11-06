# Learnings

- Implemented the first iteration of a "Remote API" node using the V3 schema, which exposes an autogrowing list of `ANY` inputs and a single `ANY` output.
- Established a JSON protocol that pickles inputs into base64 strings alongside their inferred Comfy types, enabling arbitrary payloads to be shipped to remote workers.
- Added a polling loop that understands `status`, `progress`, and `poll_url` fields to surface remote progress in the ComfyUI progress bar via the latest API helpers.
- Captured the new helper utilities in `comfy_remote.protocol` to keep serialization logic testable; see `tests/test_protocol.py` for coverage of edge cases.
- Introduced Remote Input/Output nodes that act as explicit ingress/egress points for remote payloads, complete with injection helpers for the Modal build pipeline.
- Added a Modal Deployment node that scaffolds a Modal Modal project using Modal's Python API, optionally shells out to `modal deploy`, and returns the service URL so the Remote API node can target the deployed workflow.
- Reworked the Modal scaffold to drop the external `modal-comfy-worker` dependency; we now generate `workflow.py` from an internal template and capture additional package requirements in `modal_config.json`, defaulting to a GPU-backed (`H100`) deployment unless overridden, force-reinstalling the CUDA PyTorch stack (`torch==2.3.0`, `torchvision==0.18.0`, `torchaudio==2.3.0`) plus `xformers`, and exporting `/workspace/ComfyUI` on `PYTHONPATH` so the bundled `utils` package resolves correctly.
- Modal GPU builds were downgrading `torchvision` after `xformers` installed its own dependencies, which surfaced remotely as `operator torchvision::nms does not exist`; the scaffold now pins the torch/torchvision/torchaudio stack after every dependency install and uses `pip install --no-deps` for the CUDA `xformers==0.0.26.post1` wheel so the detection ops stay registered inside the Modal container.
