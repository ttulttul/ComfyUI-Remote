# Learnings

- Implemented the first iteration of a "Remote API" node using the V3 schema, which exposes an autogrowing list of `ANY` inputs and a single `ANY` output.
- Established a JSON protocol that pickles inputs into base64 strings alongside their inferred Comfy types, enabling arbitrary payloads to be shipped to remote workers.
- Added a polling loop that understands `status`, `progress`, and `poll_url` fields to surface remote progress in the ComfyUI progress bar via the latest API helpers.
- Captured the new helper utilities in `comfy_remote.protocol` to keep serialization logic testable; see `tests/test_protocol.py` for coverage of edge cases.
