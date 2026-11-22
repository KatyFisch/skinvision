"""Fetch the demo's private inference checkpoints using the checkout's Git remote."""

from pathlib import Path
from shutil import rmtree
from subprocess import SubprocessError, run
from tempfile import TemporaryDirectory

CHECKPOINT_TAG = "demo-checkpoints-v1"
CHECKPOINT_COMMIT = "97bd1fe4e986e910c9126bc8423a6e3031d17b76"
ROUTER_MANIFEST = Path("router/model.json")
# 95% melanoma routing sensitivity on the development folds for these weights.
DEFAULT_ROUTER_CUTOFF = 9.013080125441775e-05
SEGMENTATION_CHECKPOINT = Path("unet_segmentation_full.pth")
CHECKPOINT_FILES = (SEGMENTATION_CHECKPOINT, ROUTER_MANIFEST, *(Path(f"router/fold-{fold}-refit.pt") for fold in range(1, 4)))


def complete(directory: Path) -> bool:
    return all((directory / entry).is_file() for entry in CHECKPOINT_FILES)


def download_checkpoints(repo: Path, data_dir: Path) -> Path:
    destination = data_dir / "checkpoints" / CHECKPOINT_TAG
    if complete(destination):
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as staging:
        bundle = Path(staging) / "bundle"
        try:
            local = run(
                ["git", "rev-parse", "--verify", f"refs/tags/{CHECKPOINT_TAG}^{{commit}}"],
                cwd=repo,
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if local.returncode == 0 and local.stdout.strip() == CHECKPOINT_COMMIT:
                source = repo.resolve().as_uri()
                print(f"Preparing SkinVision checkpoints from local tag {CHECKPOINT_TAG}…", flush=True)
            else:
                origin = run(["git", "remote", "get-url", "origin"], cwd=repo, check=True, capture_output=True, text=True, timeout=15)
                source = origin.stdout.strip()
                print(f"Downloading SkinVision checkpoints ({CHECKPOINT_TAG}) from origin…", flush=True)
            run(
                ["git", "clone", "--quiet", "--depth=1", "--single-branch", "--branch", CHECKPOINT_TAG, "--", source, str(bundle)],
                cwd=repo,
                check=True,
                capture_output=True,
                timeout=300,
            )
            revision = run(["git", "rev-parse", "HEAD"], cwd=bundle, check=True, capture_output=True, text=True, timeout=15)
        except (OSError, SubprocessError):
            raise RuntimeError("Could not download the demo checkpoints. Check that Git can access origin and try again.") from None

        if revision.stdout.strip() != CHECKPOINT_COMMIT or not complete(bundle):
            raise RuntimeError(f"The {CHECKPOINT_TAG} checkpoint bundle did not match this version of the demo.")

        rmtree(bundle / ".git")
        try:
            bundle.rename(destination)
        except OSError:
            if not complete(destination):
                raise

    return destination
