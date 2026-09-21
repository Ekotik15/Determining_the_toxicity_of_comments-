from pathlib import Path

import pandas as pd
import pytest

from toxic_comments import data


def test_load_splits_removes_overlap_and_preserves_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    train = pd.DataFrame(
        {
            "text": [f"train {index}" for index in range(100)] + ["shared"],
            "label": [index % 2 for index in range(100)] + [1],
        }
    )
    test = pd.DataFrame(
        {
            "text": [f"test {index}" for index in range(100)] + ["shared"],
            "label": [index % 2 for index in range(100)] + [0],
        }
    )

    def fake_read_split(url: str, cache_path: Path, expected_sha256: str) -> pd.DataFrame:
        del cache_path, expected_sha256
        return train.copy() if url == data.TRAIN_URL else test.copy()

    monkeypatch.setattr(data, "_read_split", fake_read_split)
    splits = data.load_splits(tmp_path, validation_size=0.2)

    assert "shared" not in set(splits.train["text"])
    assert "shared" not in set(splits.validation["text"])
    assert set(splits.train["label"]) == {0, 1}
    assert set(splits.validation["label"]) == {0, 1}
    assert set(splits.test["label"]) == {0, 1}


def test_load_splits_rejects_tiny_sample(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 100"):
        data.load_splits(tmp_path, sample_size=50)


def test_read_split_rejects_cache_with_wrong_checksum(tmp_path: Path) -> None:
    cache_path = tmp_path / "train.parquet"
    cache_path.write_bytes(b"not a parquet file")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        data._read_split(data.TRAIN_URL, cache_path, data.TRAIN_SHA256)


def test_read_split_downloads_and_verifies_source(tmp_path: Path) -> None:
    source_path = tmp_path / "source.parquet"
    cache_path = tmp_path / "cache" / "train.parquet"
    source = pd.DataFrame({"text": ["example"], "label": [1]})
    source.to_parquet(source_path, index=False)

    loaded = data._read_split(source_path.as_uri(), cache_path, data._sha256(source_path))

    pd.testing.assert_frame_equal(loaded, source)
    assert cache_path.read_bytes() == source_path.read_bytes()
