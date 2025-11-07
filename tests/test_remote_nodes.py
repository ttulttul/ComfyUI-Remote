import json
import tempfile
from pathlib import Path
import unittest

from comfy_remote.modal_builder import ModalDeploymentNode
from comfy_remote.remote_io import RemoteInputNode, RemoteOutputNode


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
                app_name="sample-app",
                working_directory=tmpdir,
                pip_packages=["numpy"],
                system_packages=["libgl1"],
                gpu_type="A10G",
                clean_build=False,
            )

            self.assertTrue(project.prompt.exists())
            self.assertTrue(project.workflow.exists())
            self.assertTrue(project.config.exists())

            config = json.loads(project.config.read_text())
            self.assertEqual(config["app_name"], "sample-app")
            self.assertEqual(config["extra_pip_packages"], ["numpy"])
            self.assertEqual(config["extra_system_packages"], ["libgl1"])
            self.assertEqual(config["gpu_type"], "A10G")

            workflow_source = project.workflow.read_text()
            self.assertIn("modal.App", workflow_source)
            self.assertIn("PROMPT_PATH", workflow_source)
            self.assertIn("import os", workflow_source)
            self.assertIn('pip install fastapi', workflow_source)
            self.assertIn('_ensure_comfy_module_resolution()', workflow_source)
            self.assertIn('TORCH_INSTALL_COMMAND = (', workflow_source)
            self.assertIn('f"{TORCH_INDEX_URL} torch=={TORCH_VERSION} torchvision=={TORCHVISION_VERSION} torchaudio=={TORCHAUDIO_VERSION}"', workflow_source)
            self.assertIn('"pip install --no-cache-dir --no-deps --index-url "', workflow_source)
            self.assertIn('f"{CUDA_WHEEL_INDEX} xformers=={XFORMERS_VERSION}"', workflow_source)
            self.assertIn('sys.modules.pop("utils", None)', workflow_source)
            self.assertIn('image = image.env({"PYTHONPATH": "/workspace/ComfyUI"})', workflow_source)
            self.assertIn('"min_containers": 1', workflow_source)
            self.assertIn('modal.fastapi_endpoint', workflow_source)
            self.assertIn("EXTRA_PIP_PACKAGES = ['numpy']", workflow_source)
            self.assertIn("EXTRA_SYSTEM_PACKAGES = ['libgl1']", workflow_source)
            self.assertIn("GPU_TYPE = 'A10G'", workflow_source)

    def test_prepare_modal_project_defaults_to_h100_gpu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            workflow_path = tmpdir_path / "workflow.json"
            workflow_path.write_text(json.dumps({"nodes": []}))

            project = ModalDeploymentNode._prepare_modal_project(
                workflow_file=workflow_path,
                app_name="default-gpu",
                working_directory=tmpdir,
                pip_packages=[],
                system_packages=[],
                gpu_type=None,
                clean_build=False,
            )

            config = json.loads(project.config.read_text())
            self.assertEqual(config["gpu_type"], "H100")

    def test_clean_build_removes_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            workflow_path = tmpdir_path / "workflow.json"
            workflow_path.write_text(json.dumps({"nodes": []}))

            initial = ModalDeploymentNode._prepare_modal_project(
                workflow_file=workflow_path,
                app_name="clean-me",
                working_directory=tmpdir,
                pip_packages=[],
                system_packages=[],
                gpu_type=None,
                clean_build=False,
            )

            stale_file = initial.root / "stale.txt"
            stale_file.write_text("keep me?")
            self.assertTrue(stale_file.exists())

            rebuilt = ModalDeploymentNode._prepare_modal_project(
                workflow_file=workflow_path,
                app_name="clean-me",
                working_directory=tmpdir,
                pip_packages=[],
                system_packages=[],
                gpu_type=None,
                clean_build=True,
            )

            self.assertFalse((rebuilt.root / "stale.txt").exists())


if __name__ == "__main__":
    unittest.main()
