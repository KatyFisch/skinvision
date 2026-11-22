"""Download and crop freely reusable photos for the local demo gallery."""

import argparse
import os
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from subprocess import CalledProcessError, run
from tempfile import TemporaryDirectory

from PIL import Image, ImageOps

COMMONS = "https://commons.wikimedia.org/wiki/File:"
UPLOADS = "https://upload.wikimedia.org/wikipedia/commons/"
SAMPLE_FOLDER = "demo-samples"
BUNDLED_SAMPLE_DIR = Path(__file__).resolve().parent / "assets" / SAMPLE_FOLDER


@dataclass(frozen=True)
class Sample:
    name: str
    title: str
    credit: str
    license: str
    source: str
    download: str
    checksum: str
    crop: tuple[int, int, int, int]

    @property
    def filename(self) -> str:
        return f"{self.name}.jpg"


SAMPLES = (
    Sample(
        "shoulder-mole",
        "Shoulder mole",
        "Lava.wiki",
        "Public domain",
        "Nevus.jpg",
        "b/b1/Nevus.jpg",
        "4b8dcbe1702c2498fca600160159eea34a0f4dedceae2ff036228bd63a8ac3fb",
        (218, 329, 346, 425),
    ),
    Sample(
        "halo-nevus",
        "Halo nevus",
        "JeffyP",
        "Public domain",
        "Halo_Nevus.jpg",
        "c/cb/Halo_Nevus.jpg",
        "c0bc5aa6415f0a658bfc40a72126d37295fb4938162cab13a0817b2535c8e1a4",
        (154, 174, 322, 300),
    ),
    Sample(
        "blue-nevus",
        "Blue nevus",
        "Nictitate",
        "CC0 1.0",
        "Blue_nevus.png",
        "c/c4/Blue_nevus.png",
        "481bee4e76f330f2f5f5fdd364ed17870eb52377aa73128fc3d6b8508f7bf352",
        (660, 1030, 1428, 1606),
    ),
    Sample(
        "skin-melanoma",
        "Skin melanoma",
        "National Cancer Institute (9186)",
        "Public domain in the U.S.",
        "Melanoma.jpg",
        "6/6c/Melanoma.jpg",
        "426662261a6ff73f4122cad0604df95019b14395b74600eafc8d80019e45f14d",
        (4, 0, 2276, 1704),
    ),
    Sample(
        "foot-melanoma",
        "Foot melanoma",
        "Kelly Nelson / National Cancer Institute (9246)",
        "Public domain",
        "Melanoma_(1).jpg",
        "d/df/Melanoma_%281%29.jpg",
        "081a11358014caf6b9da259c63cd6e4044fc1b4d9a95ac05a10d62452f08e63c",
        (285, 470, 441, 587),
    ),
    Sample(
        "melanoma-close-up",
        "Melanoma close-up",
        "National Cancer Institute (9189)",
        "Public domain in the U.S.",
        "Melanoma_(4).jpg",
        "b/bc/Melanoma_%284%29.jpg",
        "90e8892253571fee5d48e267760a558a6d893a1f578fa2f2803c81111b277fd6",
        (15, 108, 1395, 1143),
    ),
    Sample(
        "basal-cell-carcinoma",
        "Basal cell carcinoma",
        "National Cancer Institute (2164)",
        "Public domain in the U.S.",
        "Basal_cell_carcinoma_(2).jpg",
        "c/cd/Basal_cell_carcinoma_%282%29.jpg",
        "ee16e014de19516d0fd724b97ef13879aa3747bd5e908e03f047b03ba6bcff62",
        (1116, 690, 2180, 1488),
    ),
    Sample(
        "cherry-angioma",
        "Cherry angioma",
        "JOAN",
        "CC BY 4.0",
        "Angioma_cereza_en_hombro.jpg",
        "4/47/Angioma_cereza_en_hombro.jpg",
        "2b7d7197506417f4bb66c636e9aa66ae1afd4c5408d76e96ee41f83428b7937d",
        (1610, 3092, 3310, 4367),
    ),
    Sample(
        "bcc-on-back",
        "BCC on back",
        "John Hendrix, MD",
        "Public domain",
        "Basal_cell_carcinoma.jpg",
        "4/4f/Basal_cell_carcinoma.jpg",
        "93c59b87181fb68d46f507d29da7f8c8e4f565fd91aba05892657bc4969a9ee8",
        (21, 1, 531, 383),
    ),
)
SAMPLE_ASSETS = {f"/samples/{sample.filename}": sample for sample in SAMPLES}


def available_samples(directory: Path) -> list[dict[str, str]]:
    return [
        {"url": url, "title": sample.title, "credit": sample.credit, "license": sample.license, "source": COMMONS + sample.source}
        for url, sample in SAMPLE_ASSETS.items()
        if (directory / sample.filename).is_file()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data_dir = Path(os.environ.get("MILK10K_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    args = parser.parse_args()
    destination = args.data_dir.expanduser() / SAMPLE_FOLDER
    destination.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=destination) as staging:
        for sample in SAMPLES:
            try:
                download = run(["curl", "-fLsS", "--max-time", "30", UPLOADS + sample.download], capture_output=True, check=True)
            except CalledProcessError as error:
                raise SystemExit(f"Could not download {sample.title}. {error.stderr.decode().strip()}") from error
            original = download.stdout
            if sha256(original).hexdigest() != sample.checksum:
                raise SystemExit(f"The published source for {sample.title} has changed. Review it before updating the sample.")

            with Image.open(BytesIO(original)) as image:
                upright = ImageOps.exif_transpose(image).convert("RGB")
                cropped = upright.crop(sample.crop).resize((600, 450), Image.Resampling.LANCZOS)
                cropped.save(Path(staging) / sample.filename, quality=93, optimize=True)

        for sample in SAMPLES:
            (Path(staging) / sample.filename).replace(destination / sample.filename)
            print(f"Prepared {sample.title}")


if __name__ == "__main__":
    main()
