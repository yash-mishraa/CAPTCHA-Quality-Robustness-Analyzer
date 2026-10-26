from __future__ import annotations

import hashlib
import os
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from app.lib import contracts, data
from app.lib.contracts import IMAGE_HASH_MESSAGE, IMAGE_UNAVAILABLE_MESSAGE, ImageIntegrityError


def write_image(path: Path, *, size: tuple[int, int], mode: str, format: str = "PNG") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size).save(path, format=format)


def test_manifest_image_hash_is_a_pixel_hash(
    specimen: pd.Series,
) -> None:
    verified = data.load_verified_image(specimen, "image")
    assert verified.image.size == (contracts.EXPECTED_WIDTH, contracts.EXPECTED_HEIGHT)
    assert verified.image.mode == "L"
    assert hashlib.sha256(verified.image.tobytes()).hexdigest() == verified.sha256
    assert verified.sha256 == specimen["image_sha256"]
    assert (
        hashlib.sha256(Path(contracts.DATA_ROOT / specimen["image_path"]).read_bytes()).hexdigest()
        != verified.sha256
    )


def test_reference_image_hash_is_verified(specimen: pd.Series) -> None:
    verified = data.load_verified_image(specimen, "reference")
    assert verified.image.size == (contracts.EXPECTED_WIDTH, contracts.EXPECTED_HEIGHT)
    assert verified.sha256 == specimen["reference_sha256"]


def test_selected_image_is_loaded_by_record_not_path(specimen: pd.Series) -> None:
    payload = data.load_selected_image(str(specimen["sample_id"]), "image")
    assert len(payload) > 0
    with pytest.raises(ValueError, match="role must be"):
        data.load_selected_image(str(specimen["sample_id"]), "preview")


def test_tampered_pixels_fail_closed(
    image_root: Path,
    specimen: pd.Series,
) -> None:
    path = image_root / str(specimen["image_path"])
    with Image.open(path) as image:
        tampered = image.copy()
    tampered.putpixel((0, 0), (int(tampered.getpixel((0, 0))) + 1) % 256)
    tampered.save(path, format="PNG")

    with pytest.raises(ImageIntegrityError) as caught:
        data.load_verified_image(specimen, "image", data_root=image_root)
    assert str(caught.value) == IMAGE_HASH_MESSAGE
    assert "pixel hash" in caught.value.detail


def test_missing_image_fails_closed(
    image_root: Path,
    specimen: pd.Series,
) -> None:
    (image_root / str(specimen["image_path"])).unlink()

    with pytest.raises(ImageIntegrityError) as caught:
        data.load_verified_image(specimen, "image", data_root=image_root)
    assert str(caught.value) == IMAGE_UNAVAILABLE_MESSAGE


def test_wrong_container_format_fails_closed(
    image_root: Path,
    specimen: pd.Series,
) -> None:
    buffer = BytesIO()
    Image.new("RGB", (contracts.EXPECTED_WIDTH, contracts.EXPECTED_HEIGHT)).save(
        buffer, format="JPEG"
    )
    path = image_root / str(specimen["image_path"])
    path.write_bytes(buffer.getvalue())

    with pytest.raises(ImageIntegrityError) as caught:
        data.load_verified_image(specimen, "image", data_root=image_root)
    assert str(caught.value) == IMAGE_HASH_MESSAGE


def test_wrong_dimensions_fail_closed(
    image_root: Path,
    specimen: pd.Series,
) -> None:
    write_image(
        image_root / str(specimen["image_path"]),
        size=(contracts.EXPECTED_WIDTH, contracts.EXPECTED_HEIGHT - 1),
        mode="L",
    )

    with pytest.raises(ImageIntegrityError) as caught:
        data.load_verified_image(specimen, "image", data_root=image_root)
    assert "dimensions" in caught.value.detail


def test_wrong_mode_fails_closed(
    image_root: Path,
    specimen: pd.Series,
) -> None:
    write_image(
        image_root / str(specimen["image_path"]),
        size=(contracts.EXPECTED_WIDTH, contracts.EXPECTED_HEIGHT),
        mode="RGB",
    )

    with pytest.raises(ImageIntegrityError) as caught:
        data.load_verified_image(specimen, "image", data_root=image_root)
    assert "mode" in caught.value.detail


def test_malformed_manifest_hash_fails_closed(
    image_root: Path,
    specimen: pd.Series,
) -> None:
    row = specimen.copy()
    row["image_sha256"] = "not-a-sha256"

    with pytest.raises(ImageIntegrityError) as caught:
        data.load_verified_image(row, "image", data_root=image_root)
    assert str(caught.value) == IMAGE_HASH_MESSAGE
    assert "invalid manifest hash" in caught.value.detail


@pytest.mark.parametrize(
    "relative",
    [
        "",
        " images/clean/x.png",
        "../ground_truth.csv",
        "/etc/passwd",
        "images/../ground_truth.csv",
        "images/clean/../../ground_truth.csv",
        "images//clean/x.png",
        "images/clean/x.png/",
        "images/clean/x.jpg",
        "images\\clean\\x.png",
        "images/clean/x.png\x00",
        "images",
        "clean/x.png",
        None,
        42,
    ],
)
def test_path_traversal_and_malformed_paths_are_rejected(relative: object) -> None:
    with pytest.raises(ImageIntegrityError):
        data.resolve_image_path(relative)


def test_valid_dataset_relative_path_is_accepted() -> None:
    resolved = data.resolve_image_path("images/clean/example.png")
    assert resolved == (contracts.DATA_ROOT / "images" / "clean" / "example.png").resolve()


def test_symlinked_image_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    clean = root / "images" / "clean"
    clean.mkdir(parents=True)
    target = clean / "target.png"
    write_image(target, size=(contracts.EXPECTED_WIDTH, contracts.EXPECTED_HEIGHT), mode="L")
    link = clean / "link.png"
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(ImageIntegrityError) as caught:
        data.resolve_image_path("images/clean/link.png", data_root=root)
    assert "symlink" in caught.value.detail


def test_image_path_does_not_escape_data_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    try:
        os.symlink(outside, root / "images" / "escaped")
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(ImageIntegrityError) as caught:
        data.resolve_image_path("images/escaped/x.png", data_root=root)
    assert "escapes" in caught.value.detail
