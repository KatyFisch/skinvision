from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from demo.model_assets import CHECKPOINT_FILES, CHECKPOINT_TAG, download_checkpoints


class DemoModelAssetsTest(TestCase):
    def test_download_checks_commit_and_reuses_files_offline(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            remote = root / "remote"
            checkout = root / "checkout"
            data = root / "data"
            for repo in (remote, checkout):
                run(["git", "init", "-q", str(repo)], check=True)
            for entry in CHECKPOINT_FILES:
                path = remote / entry
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(entry.as_posix())
            run(["git", "add", "."], cwd=remote, check=True)
            run(
                ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "Checkpoints"], cwd=remote, check=True
            )
            run(["git", "tag", CHECKPOINT_TAG], cwd=remote, check=True)
            commit = run(["git", "rev-parse", "HEAD"], cwd=remote, check=True, capture_output=True, text=True).stdout.strip()
            run(["git", "remote", "add", "origin", remote.as_uri()], cwd=checkout, check=True)

            with self.assertRaisesRegex(RuntimeError, "did not match"):
                download_checkpoints(checkout, data)
            self.assertFalse((data / "checkpoints" / CHECKPOINT_TAG).exists())

            with patch("demo.model_assets.CHECKPOINT_COMMIT", commit):
                cache = download_checkpoints(checkout, data)
                run(["git", "fetch", "-q", "origin", "tag", CHECKPOINT_TAG], cwd=checkout, check=True)
                run(["git", "remote", "remove", "origin"], cwd=checkout, check=True)
                self.assertEqual(download_checkpoints(checkout, data), cache)
                local_cache = download_checkpoints(checkout, root / "local-data")
            for entry in CHECKPOINT_FILES:
                self.assertEqual((cache / entry).read_text(), entry.as_posix())
                self.assertEqual((local_cache / entry).read_text(), entry.as_posix())
            self.assertFalse((cache / ".git").exists())
