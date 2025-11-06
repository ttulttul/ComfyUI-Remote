import json
import tempfile
from pathlib import Path
import unittest

from comfy_remote.modal_builder import (
    ModalDeploymentNode,
    RemoteInputNode,
    RemoteOutputNode,
)


class RemoteNodesTest(unittest.TestCase):
    def test_remote_input_payload_resolution(self):
        node_id = "node-123"
        payload = {"payload": 42, "other": "ignored"}
        RemoteInputNode.inject_payload(node_id, payload)

        resolved = RemoteInputNode._resolve_payload(node_id, "payload", None)
        self.assertEqual(resolved, 42)

        missing_key = RemoteInputNode._resolve_payload(node_id, "missing", "fallback")
        self.assertEqual(missing_key, "fallback")

        RemoteInputNode.pop_payload(node_id)
        empty = RemoteInputNode._resolve_payload(node_id, "payload", "default")
        self.assertEqual(empty, "default")

    def test_remote_output_store(self):
        node_id = "node-output"
        RemoteOutputNode._register_output(node_id, "result", 99)
        RemoteOutputNode._register_output(node_id, "extra", {"status": "ok"})

        stored = RemoteOutputNode.pop_output(node_id)
        self.assertEqual(stored, {"result": 99, "extra": {"status": "ok"}})

        self.assertIsNone(RemoteOutputNode.pop_output(node_id))


class ModalDeploymentTest(unittest.TestCase):
    def test_prepare_modal_project_generates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            workflow_path = tmpdir_path / "workflow.json"
            workflow_path.write_text(json.dumps({"nodes": []}))

            project = ModalDeploymentNode._prepare_modal_project(
                workflow_file=workflow_path,
                repository_url="https://example.com/repo.git",
                app_name="sample-app",
                working_directory=tmpdir,
            )

            self.assertTrue(project.prompt.exists())
            self.assertTrue(project.snapshot.exists())
            self.assertTrue(project.workflow.exists())

            snapshot = json.loads(project.snapshot.read_text())
            self.assertIn("repositories", snapshot)
            self.assertEqual(snapshot["repositories"][0]["url"], "https://example.com/repo.git")

            workflow_source = project.workflow.read_text()
            self.assertIn("modal.App", workflow_source)
            self.assertIn("PROMPT_PATH", workflow_source)


if __name__ == "__main__":
    unittest.main()
