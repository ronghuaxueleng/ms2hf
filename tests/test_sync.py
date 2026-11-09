import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import yaml

import ms2hf


class ModelScopeListingTests(unittest.TestCase):
    def test_model_and_dataset_listings_exclude_directories(self):
        backend = object.__new__(ms2hf.ModelScopeBackend)
        backend.token = None
        backend.api = Mock()
        files = [
            {"Path": "quant", "Type": "tree", "Size": 0},
            {"Path": "quant/int4_packed.pt", "Type": "blob", "Size": 123},
            {"Path": "empty.txt", "Type": "blob", "Size": 0},
        ]
        backend.api.get_model_files.return_value = files
        backend.api.list_repo_tree.return_value = {"Data": {"Files": files}}
        for repo_type in ("model", "dataset"):
            with self.subTest(repo_type=repo_type):
                result = backend.list_files("owner/repo", repo_type, "master")
                self.assertEqual([f.path for f in result], ["quant/int4_packed.pt", "empty.txt"])


class ModelCardTests(unittest.TestCase):
    def test_removes_empty_references_and_preserves_body_and_other_metadata(self):
        body = "\r\n\r\n# 中文模型\r\n\r\n---\r\n正文\r\n"
        for value, expected in (
            ('[""]', None), ('null', None), ('""', None),
            ('[null, "", owner/model]', ["owner/model"]),
        ):
            with self.subTest(value=value):
                original = ("---\r\nbase_model: " + value + "\r\nlicense: apache-2.0\r\n---" + body).encode("utf-8")
                result = ms2hf._clean_hf_model_card(original).decode("utf-8")
                header, actual_body = result[4:].split("---", 1)
                metadata = yaml.safe_load(header)
                self.assertEqual(metadata.get("base_model"), expected)
                self.assertEqual(metadata["license"], "apache-2.0")
                self.assertEqual(actual_body, body)

    def test_valid_cards_remain_byte_identical(self):
        for content in (
            b"# No metadata\n---\nbase_model: ['']\n---\n",
            b"---\nlicense: apache-2.0\n---\n# Body\n",
            b"---\nbase_model: owner/model\n---\n# Body\n",
            b"---\nbase_model: [owner/model]\n---\n# Body\n",
        ):
            with self.subTest(content=content):
                self.assertEqual(ms2hf._clean_hf_model_card(content), content)

    def test_hf_upload_cleans_model_card_without_modifying_download(self):
        backend = object.__new__(ms2hf.HuggingFaceBackend)
        backend.token = None
        backend.api = Mock()
        with tempfile.TemporaryDirectory() as directory:
            local_path = Path(directory) / "README.md"
            original = b'---\nbase_model:\n- ""\nlicense: apache-2.0\n---\n# Body\n'
            local_path.write_bytes(original)
            backend.upload_file("owner/repo", "model", "README.md", local_path, "sync", "branch")
            uploaded = backend.api.upload_file.call_args.kwargs
            self.assertNotIn(b"base_model", uploaded["path_or_fileobj"])
            self.assertEqual(uploaded["revision"], "branch")
            self.assertEqual(local_path.read_bytes(), original)
            backend.upload_file("owner/repo", "model", "nested/README.md", local_path, "sync")
            self.assertEqual(backend.api.upload_file.call_args.kwargs["path_or_fileobj"], str(local_path))


if __name__ == "__main__":
    unittest.main()
