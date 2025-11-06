from dataclasses import asdict
import importlib
import unittest

protocol = importlib.import_module("comfy_remote.protocol")


class ProtocolHelpersTest(unittest.TestCase):
    def test_infer_comfy_type_basic(self):
        cases = [
            (123, "INT"),
            (1.5, "FLOAT"),
            (True, "BOOLEAN"),
            ("text", "STRING"),
            ([1, 2, 3], "LIST"),
            ([(("dummy", {}))], "CONDITIONING"),
            ({"a": 1}, "DICT"),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(protocol.infer_comfy_type(value), expected)

    def test_extract_progress_variants(self):
        cases = [
            ({"value": 10, "max": 100}, (10.0, 100.0)),
            ({"percent": 0.5}, (0.5, 1.0)),
            ({"percent": 75}, (75.0, 100.0)),
            ({"ratio": 0.25}, (0.25, 1.0)),
            (0.2, (0.2, 1.0)),
            (50, (50.0, protocol.DEFAULT_PROGRESS_MAX)),
        ]
        for progress, expected in cases:
            with self.subTest(progress=progress):
                response = {"progress": progress}
                self.assertEqual(protocol.extract_progress(response), expected)

    def test_serialize_roundtrip_dict(self):
        original = {"alpha": 1, "beta": [1, 2, 3]}
        serialized = protocol.serialize_inputs([("input_1", original)])
        self.assertEqual(len(serialized), 1)
        payload = asdict(serialized[0])
        restored = protocol.deserialize_value(payload)
        self.assertEqual(restored, original)

    def test_extract_poll_url_relative(self):
        response = {"poll": "/status/123"}
        poll_url = protocol.extract_poll_url(response, base_url="https://api.example.com/task")
        self.assertEqual(poll_url, "https://api.example.com/status/123")

    def test_extract_progress_message(self):
        response = {"progress": {"message": "working"}}
        self.assertEqual(protocol.extract_progress_message(response), "working")


if __name__ == "__main__":
    unittest.main()
