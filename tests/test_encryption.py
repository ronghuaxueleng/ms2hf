import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cryptography.exceptions import InvalidTag

import ms2hf
import ms2hf_crypto as crypto


class MemoryBackend:
    """Exercise the real sync flow without SDKs, credentials or remote writes."""

    def __init__(self, files=None):
        self.files = dict(files or {})
        self.uploads = []
        self.downloads = []
        self.creations = 0
        self.fail_paths = set()
        self.fail_payloads = False
        self.missing = False

    def create_repo(self, *args, **kwargs):
        self.creations += 1
        self.missing = False

    def list_files(self, *args):
        if self.missing:
            raise RuntimeError("Repository not found")
        return [ms2hf.FileItem(path, len(data)) for path, data in self.files.items()]

    def download_file(self, repo_id, repo_type, revision, file_path, local_dir, force):
        self.downloads.append(file_path)
        destination = local_dir / file_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.files[file_path])
        return destination

    def upload_file(self, repo_id, repo_type, file_path, local_path, commit_message,
                    target_revision=None):
        if file_path in self.fail_paths or (
            self.fail_payloads and file_path.startswith(crypto.PAYLOAD_PREFIX)
        ):
            raise RuntimeError("Simulated upload failure")
        self.files[file_path] = local_path.read_bytes()
        self.uploads.append((file_path, commit_message))


class EncryptionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.context = contextlib.ExitStack()
        self.addCleanup(self.context.close)
        self.context.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.context.enter_context(contextlib.redirect_stderr(io.StringIO()))

    def test_round_trip_empty_and_multiple_chunks(self):
        key = bytes(range(32))
        with patch.object(crypto, "CHUNK_SIZE", 32):
            for size in (0, 1, 32, 65):
                with self.subTest(size=size):
                    original = self.root / "original"
                    encrypted = self.root / "encrypted"
                    restored = self.root / "restored"
                    data = bytes(range(size))
                    original.write_bytes(data)
                    crypto.encrypt_file(original, encrypted, key)
                    crypto.decrypt_file(encrypted, restored, key)
                    self.assertEqual(restored.read_bytes(), data)

    def test_wrong_key_tampering_truncation_and_trailing_data(self):
        key = bytes(range(32))
        original = self.root / "original"
        encrypted = self.root / "encrypted"
        output = self.root / "output"
        original.write_bytes(b"contents")
        crypto.encrypt_file(original, encrypted, key)
        data = encrypted.read_bytes()
        with self.assertRaises(InvalidTag):
            crypto.decrypt_file(encrypted, output, b"x" * 32)
        damaged = bytearray(data)
        damaged[len(crypto.MAGIC) + 16 + 4 + 12] ^= 1
        for content in (damaged, data[:-1], data[:-32], data + b"extra"):
            with self.subTest(length=len(content)):
                encrypted.write_bytes(content)
                with self.assertRaises((InvalidTag, ValueError)):
                    crypto.decrypt_file(encrypted, output, key)


class SyncTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.context = contextlib.ExitStack()
        self.addCleanup(self.context.close)
        self.context.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.context.enter_context(contextlib.redirect_stderr(io.StringIO()))
        self.source = MemoryBackend({
            "README.md": b"source readme: keywordsecretfilename",
            "folder/keywordsecretfilename.json": b'{"text": "keywordsecretfilename"}',
            "data/中文.bin": b"binary\x00\xff" * 70,
            "empty": b"",
        })
        self.target = MemoryBackend()
        self.context.enter_context(patch.object(ms2hf, "HuggingFaceBackend", lambda token: self.source))
        self.context.enter_context(patch.object(ms2hf, "ModelScopeBackend", lambda token: self.target))
        self.context.enter_context(patch.object(crypto, "CHUNK_SIZE", 128))

    def sync(self, **overrides):
        arguments = dict(
            direction="hf2ms", source_repo="owner/source", target_repo="owner/target",
            repo_type="model", source_revision=None, target_revision=None,
            work_dir=self.root / "work", include_regex=None, exclude_regex=None,
            force=False, workers=1, max_retries=1, retry_sleep=0,
            hf_token=None, ms_token=None, create_target=True, private=False,
            dry_run=False, keep_local=False, commit_message_template="{direction}: sync {path}",
            encrypt=True,
        )
        arguments.update(overrides)
        return ms2hf.sync_repositories(**arguments)

    def snapshot(self):
        archive = self.root / "archive"
        archive.mkdir(exist_ok=True)
        for relative, content in self.target.files.items():
            destination = archive / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        return archive

    def test_encrypted_parallel_sync_restores_names_and_contents(self):
        results = self.sync(workers=2)
        self.assertEqual([r.status for r in results], ["uploaded"] * len(self.source.files))
        self.assertIn(crypto.KEY_PATH, self.target.files)
        self.assertIn(crypto.MANIFEST_PATH, self.target.files)
        for remote, commit in self.target.uploads:
            self.assertNotIn("keywordsecretfilename", remote)
            self.assertNotIn("keywordsecretfilename", commit)
            self.assertNotIn(b"keywordsecretfilename", self.target.files[remote])
        self.assertEqual(list((self.root / "work").iterdir()), [])
        output = self.root / "restored"
        self.assertEqual(crypto.restore_repository(self.snapshot(), output), 0)
        for path, data in self.source.files.items():
            self.assertEqual((output / path).read_bytes(), data)

    def test_rerun_skips_and_force_reuses_key_and_payload_paths(self):
        self.sync()
        key = self.target.files[crypto.KEY_PATH]
        payload_paths = {p for p in self.target.files if p.startswith(crypto.PAYLOAD_PREFIX)}
        self.source.downloads.clear()
        results = self.sync()
        self.assertTrue(all(r.status == "skipped" for r in results))
        self.assertEqual(self.source.downloads, [])
        self.source.files["empty"] = b"updated"
        self.sync(force=True)
        self.assertEqual(self.target.files[crypto.KEY_PATH], key)
        self.assertEqual({p for p in self.target.files if p.startswith(crypto.PAYLOAD_PREFIX)}, payload_paths)
        output = self.root / "restored"
        self.assertEqual(crypto.restore_repository(self.snapshot(), output), 0)
        self.assertEqual((output / "empty").read_bytes(), b"updated")

    def test_dataset_sync_and_keep_local(self):
        self.assertTrue(all(r.status == "uploaded" for r in self.sync(
            repo_type="dataset", keep_local=True)))
        self.assertEqual(len(list((self.root / "work").glob("ms2hf-file-*"))), len(self.source.files))
        self.assertEqual(crypto.restore_repository(self.snapshot(), self.root / "restored"), 0)

    def test_failed_payloads_resume_without_losing_key_or_mapping(self):
        self.target.fail_payloads = True
        self.assertTrue(all(r.status == "failed" for r in self.sync()))
        key = self.target.files[crypto.KEY_PATH]
        self.assertEqual(list((self.root / "work").iterdir()), [])
        self.assertEqual(crypto.restore_repository(self.snapshot(), self.root / "incomplete"), 1)
        self.target.fail_payloads = False
        self.assertTrue(all(r.status == "uploaded" for r in self.sync()))
        self.assertEqual(self.target.files[crypto.KEY_PATH], key)
        self.assertEqual(crypto.restore_repository(self.snapshot(), self.root / "restored"), 0)

    def test_key_only_interruption_can_resume(self):
        self.target.fail_paths.add(crypto.MANIFEST_PATH)
        with self.assertRaisesRegex(RuntimeError, "Simulated"):
            self.sync()
        self.assertEqual(set(self.target.files), {crypto.KEY_PATH})
        key = self.target.files[crypto.KEY_PATH]
        self.target.fail_paths.clear()
        self.assertTrue(all(r.status == "uploaded" for r in self.sync()))
        self.assertEqual(self.target.files[crypto.KEY_PATH], key)

    def test_preview_has_no_remote_writes_or_source_downloads(self):
        self.assertTrue(all(r.status == "dry-run" for r in self.sync(dry_run=True)))
        self.assertEqual(self.target.creations, 0)
        self.assertEqual(self.target.files, {})
        self.assertEqual(self.source.downloads, [])
        self.sync()
        self.target.uploads.clear()
        self.target.downloads.clear()
        self.source.downloads.clear()
        self.assertTrue(all(r.status == "skipped" for r in self.sync(dry_run=True)))
        self.assertEqual(self.target.uploads, [])
        self.assertEqual(set(self.target.downloads), {crypto.KEY_PATH, crypto.MANIFEST_PATH})
        self.assertEqual(self.source.downloads, [])

    def test_preview_does_not_create_missing_target(self):
        self.target.missing = True
        with self.assertRaisesRegex(RuntimeError, "Dry runs do not create"):
            self.sync(dry_run=True)
        self.assertEqual(self.target.creations, 0)

    def test_missing_or_corrupt_metadata_stops_before_writing(self):
        self.sync()
        saved = dict(self.target.files)
        cases = []
        for path in (crypto.KEY_PATH, crypto.MANIFEST_PATH):
            files = dict(saved)
            del files[path]
            cases.append(files)
        files = dict(saved)
        files[crypto.MANIFEST_PATH] = b"corrupt"
        cases.append(files)
        for files in cases:
            with self.subTest(paths=list(files)):
                self.target.files = files
                self.target.uploads.clear()
                with self.assertRaises(ValueError):
                    self.sync(force=True)
                self.assertEqual(self.target.uploads, [])

    def test_filtering_and_plaintext_sync(self):
        results = self.sync(include_regex=[r"\.json$"], exclude_regex=[r"^data/"])
        self.assertEqual([r.path for r in results], ["folder/keywordsecretfilename.json"])
        with self.assertRaisesRegex(ValueError, "encrypted archive"):
            self.sync(encrypt=False)
        self.target.files.clear()
        self.assertTrue(all(r.status == "uploaded" for r in self.sync(encrypt=False)))
        self.assertEqual(self.target.files, self.source.files)

    def test_reject_plaintext_target_and_unsupported_options(self):
        self.target.files["plain.txt"] = b"existing plaintext"
        with self.assertRaisesRegex(ValueError, "new/empty"):
            self.sync()
        with self.assertRaisesRegex(ValueError, "only by hf2ms"):
            self.sync(direction="ms2hf")
        with self.assertRaisesRegex(ValueError, "default branch"):
            self.sync(target_revision="other")
        self.assertEqual(self.target.uploads, [])

    def test_restore_preserves_existing_files_on_authentication_failure(self):
        self.sync()
        archive = self.snapshot()
        output = self.root / "restored"
        self.assertEqual(crypto.restore_repository(archive, output), 0)
        kept = output / "empty"
        kept.write_bytes(b"local change")
        self.assertEqual(crypto.restore_repository(archive, output), 1)
        self.assertEqual(kept.read_bytes(), b"local change")
        key = crypto.read_key(archive / crypto.KEY_PATH)
        mapping = crypto.read_manifest(archive / crypto.MANIFEST_PATH, key)
        payload = archive / mapping["empty"]
        payload.write_bytes(payload.read_bytes()[:-1])
        self.assertEqual(crypto.restore_repository(archive, output, overwrite=True), 1)
        self.assertEqual(kept.read_bytes(), b"local change")
        self.assertEqual(list(output.rglob(".ms2hf-*")), [])

    def test_manifest_paths_cannot_escape_output(self):
        self.sync()
        archive = self.snapshot()
        key = crypto.read_key(archive / crypto.KEY_PATH)
        mapping = crypto.read_manifest(archive / crypto.MANIFEST_PATH, key)
        malicious = self.root / "malicious.json"
        malicious.write_text(json.dumps({"version": 1, "files": {
            "../escape": next(iter(mapping.values()))
        }}), encoding="utf-8")
        crypto.encrypt_file(malicious, archive / crypto.MANIFEST_PATH, key)
        with self.assertRaisesRegex(ValueError, "Invalid repository path"):
            crypto.restore_repository(archive, self.root / "restored")
        self.assertFalse((self.root / "escape").exists())


if __name__ == "__main__":
    unittest.main()
