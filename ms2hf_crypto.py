"""Encrypted repository storage and local restoration for hf2ms."""

import argparse
import base64
import json
import os
import struct
import sys
import tempfile
import uuid
from pathlib import Path, PurePosixPath

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


KEY_PATH = "ms2hf/key.json"
MANIFEST_PATH = "ms2hf/manifest.enc"
PAYLOAD_PREFIX = "ms2hf/data/"
MAGIC = b"MS2HFENC\x01"
CHUNK_SIZE = 4 * 1024 * 1024
REPOSITORY_README = """# Encrypted archive

本仓库由 ms2hf 加密上传。文件内容、原始文件名和目录结构保存在密文中，
不能直接用作模型或数据集仓库，需要先下载并解密。

密钥位于 `ms2hf/key.json`，所有有仓库读取权限的人均可解密。
此布局不提供访问保密性。请保留密钥、清单和所有数据文件。

## 恢复文件

1. 下载整个仓库（含实际的大文件内容；Git LFS 指针不能用于解密）。
2. 从包含加密功能的 ms2hf 源码安装工具：

```bash
pip install /path/to/ms2hf
```

项目主页：https://github.com/pkufool/ms2hf
使用 Fork 时请从运行同步任务的同一代码版本安装。

3. 执行（替换两个本地目录）：

```bash
ms2hf-decrypt ./downloaded-repo ./restored-files
```

恢复后使用 `restored-files` 中的原始模型或数据集。默认不覆盖已有文件；
确需覆盖时添加 `--overwrite`。同步未完成时会报告缺失文件，
请先重新运行加密同步补齐文件。不要同时向本仓库运行多个同步任务。
"""


def encrypt_file(source: Path, destination: Path, key: bytes) -> None:
    """Encrypt bounded chunks, including an authenticated end marker."""
    cipher = AESGCM(key)
    header = MAGIC + os.urandom(16)
    with source.open("rb") as reader, destination.open("wb") as writer:
        writer.write(header)
        index = 0
        while True:
            chunk = reader.read(CHUNK_SIZE)
            nonce = os.urandom(12)
            encrypted = cipher.encrypt(nonce, chunk, header + struct.pack(">Q", index))
            writer.write(struct.pack(">I", len(encrypted)))
            writer.write(nonce)
            writer.write(encrypted)
            if not chunk:
                return
            index += 1


def decrypt_file(source: Path, destination: Path, key: bytes) -> None:
    cipher = AESGCM(key)
    with source.open("rb") as reader, destination.open("wb") as writer:
        header = reader.read(len(MAGIC) + 16)
        if len(header) != len(MAGIC) + 16 or not header.startswith(MAGIC):
            raise ValueError("Invalid encrypted file header")
        index = 0
        while True:
            length_bytes = reader.read(4)
            if len(length_bytes) != 4:
                raise ValueError("Truncated encrypted file: missing end marker")
            length = struct.unpack(">I", length_bytes)[0]
            if not 16 <= length <= CHUNK_SIZE + 16:
                raise ValueError("Invalid encrypted chunk length")
            nonce = reader.read(12)
            encrypted = reader.read(length)
            if len(nonce) != 12 or len(encrypted) != length:
                raise ValueError("Truncated encrypted chunk")
            chunk = cipher.decrypt(nonce, encrypted, header + struct.pack(">Q", index))
            if not chunk:
                if reader.read(1):
                    raise ValueError("Unexpected data after end marker")
                return
            writer.write(chunk)
            index += 1


def read_key(path: Path) -> bytes:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("version") != 1:
        raise ValueError("Unsupported encryption key version")
    key = base64.b64decode(document["key"], validate=True)
    if len(key) != 32:
        raise ValueError("Expected a 256-bit encryption key")
    return key


def safe_path(root: Path, relative: str) -> Path:
    """Keep downloaded manifest paths inside their requested directory."""
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError(f"Invalid repository path: {relative!r}")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or not parts.parts:
        raise ValueError(f"Invalid repository path: {relative!r}")
    root = root.resolve()
    result = root.joinpath(*parts.parts).resolve()
    if root not in result.parents:
        raise ValueError(f"Path escapes destination: {relative!r}")
    return result


def read_manifest(path: Path, key: bytes) -> dict:
    with tempfile.TemporaryDirectory(prefix="ms2hf-manifest-") as directory:
        plain = Path(directory) / "manifest.json"
        decrypt_file(path, plain, key)
        document = json.loads(plain.read_text(encoding="utf-8"))
    if document.get("version") != 1 or not isinstance(document.get("files"), dict):
        raise ValueError("Unsupported encryption manifest")
    files = document["files"]
    for original, stored in files.items():
        safe_path(Path.cwd(), original)
        if not isinstance(stored, str) or not stored.startswith(PAYLOAD_PREFIX):
            raise ValueError("Invalid encrypted payload path")
        safe_path(Path.cwd(), stored)
    if len(set(files.values())) != len(files):
        raise ValueError("Duplicate encrypted payload paths")
    return files


class EncryptedTarget:
    """Map source paths to persistent random filenames on the target platform."""

    def __init__(self, backend, repo_id, repo_type, existing, selected, *, dry_run,
                 work_dir, retry):
        self.files = {}
        has_key = KEY_PATH in existing
        has_manifest = MANIFEST_PATH in existing
        payloads = {path for path in existing if path.startswith(PAYLOAD_PREFIX)}
        if (has_manifest or payloads) and not has_key:
            raise ValueError("Encrypted archive is missing its key; refusing to replace it")
        if payloads and not has_manifest:
            raise ValueError("Encrypted archive is missing its manifest")
        if existing - {KEY_PATH, MANIFEST_PATH, "README.md", ".gitattributes"} - payloads:
            raise ValueError("Use a new/empty target repository for encrypted uploads; existing plaintext files would remain")
        # Metadata is needed even for preview to identify existing original paths.
        with tempfile.TemporaryDirectory(prefix="ms2hf-metadata-", dir=str(work_dir)) as directory:
            folder = Path(directory)
            if has_key:
                local_key = []
                retry(lambda: local_key.append(backend.download_file(
                    repo_id, repo_type, None, KEY_PATH, folder, False)))
                self.key = read_key(local_key[-1])
            else:
                self.key = AESGCM.generate_key(bit_length=256)
            if has_manifest:
                local_manifest = []
                retry(lambda: local_manifest.append(backend.download_file(
                    repo_id, repo_type, None, MANIFEST_PATH, folder, False)))
                self.files = read_manifest(local_manifest[-1], self.key)
            if payloads - set(self.files.values()):
                raise ValueError("Encrypted payloads are missing from the manifest")
            self.existing = {path for path, stored in self.files.items() if stored in existing}
            for item in selected:
                safe_path(folder, item.path)
                if item.path not in self.files:
                    self.files[item.path] = PAYLOAD_PREFIX + uuid.uuid4().hex + ".enc"
            if dry_run:
                return
            key_file = folder / "key.json"
            key_file.write_text(json.dumps({
                "version": 1, "key": base64.b64encode(self.key).decode("ascii")
            }), encoding="utf-8")
            manifest = folder / "manifest.json"
            manifest.write_text(json.dumps({"version": 1, "files": self.files},
                                           ensure_ascii=False), encoding="utf-8")
            encrypted_manifest = folder / "manifest.enc"
            encrypt_file(manifest, encrypted_manifest, self.key)
            readme = folder / "README.md"
            readme.write_text(REPOSITORY_README, encoding="utf-8")
            # Persist the key and full plan before any payload so interrupted runs resume.
            metadata = [(MANIFEST_PATH, encrypted_manifest), ("README.md", readme)]
            if not has_key:
                metadata.insert(0, (KEY_PATH, key_file))
            for remote, local in metadata:
                retry(lambda remote=remote, local=local: backend.upload_file(
                    repo_id, repo_type, remote, local, "Update encrypted archive metadata"))

    def encrypt(self, original: str, local_path: Path, work_dir: Path):
        # Isolate the output from SDK downloads, including colliding source basenames.
        with tempfile.NamedTemporaryFile(prefix="ms2hf-encrypted-", suffix=".enc",
                                         dir=str(work_dir), delete=False) as temporary:
            destination = Path(temporary.name)
        try:
            encrypt_file(local_path, destination, self.key)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        return self.files[original], destination


def restore_repository(repository: Path, output: Path, overwrite: bool = False) -> int:
    repository = repository.resolve()
    output = output.resolve()
    if output == repository or repository in output.parents or output in repository.parents:
        raise ValueError("Restore into a separate directory that does not overlap the encrypted repository")
    key = read_key(repository / KEY_PATH)
    files = read_manifest(repository / MANIFEST_PATH, key)
    failures = 0
    for original, stored in files.items():
        temporary = None
        try:
            destination = safe_path(output, original)
            encrypted = safe_path(repository, stored)
            if destination.exists() and not overwrite:
                raise FileExistsError(f"Destination exists: {destination}; use --overwrite")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix=".ms2hf-", dir=str(destination.parent),
                                             delete=False) as handle:
                temporary = Path(handle.name)
            decrypt_file(encrypted, temporary, key)
            # Publish only after authentication, preserving existing files on failure.
            temporary.replace(destination)
            print(f"Restored: {original}")
        except Exception as exc:
            failures += 1
            print(f"FAILED {original}: {type(exc).__name__}: {exc}", file=sys.stderr)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print(f"Restored: {len(files) - failures}; failed: {failures}")
    return 1 if failures else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Restore a locally downloaded ms2hf encrypted repository.")
    parser.add_argument("repository", type=Path, help="Downloaded encrypted repository directory")
    parser.add_argument("output", type=Path, help="Separate directory for restored files")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files")
    args = parser.parse_args(argv)
    try:
        return restore_repository(args.repository, args.output, args.overwrite)
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
