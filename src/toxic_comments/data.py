import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_REVISION = "ab3aee44e1d13d6938102bef4c6af909e6a9799d"
TRAIN_URL = (
    "https://huggingface.co/datasets/mteb/toxic_conversations_50k/"
    f"resolve/{DATASET_REVISION}/default/train/0000.parquet"
)
TEST_URL = (
    "https://huggingface.co/datasets/mteb/toxic_conversations_50k/"
    f"resolve/{DATASET_REVISION}/default/test/0000.parquet"
)
TRAIN_SHA256 = "d27d62973dfc05d1931c5cde813a65792c960cae4e1de914b7650f2c8d00376a"
TEST_SHA256 = "0da0742843c8c94406779a18d6a9c575e4a592e7e650fee52593e9cf5c9475cd"


@dataclass(frozen=True)
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_split(url: str, cache_path: Path, expected_sha256: str) -> pd.DataFrame:
    if cache_path.exists():
        actual_sha256 = _sha256(cache_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Checksum mismatch for {cache_path}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        download_path = cache_path.with_suffix(f"{cache_path.suffix}.download")
        with urlopen(url) as response, download_path.open("wb") as destination:
            shutil.copyfileobj(response, destination)
        actual_sha256 = _sha256(download_path)
        if actual_sha256 != expected_sha256:
            download_path.unlink()
            raise ValueError(
                f"Checksum mismatch for downloaded split: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        download_path.replace(cache_path)

    frame = pd.read_parquet(cache_path)
    required_columns = {"text", "label"}
    if not required_columns.issubset(frame.columns):
        raise ValueError(f"Dataset must contain columns: {sorted(required_columns)}")

    frame = frame[["text", "label"]].dropna().copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["label"] = frame["label"].astype(int)
    frame = frame[frame["text"].str.len() > 0]
    if set(frame["label"].unique()) - {0, 1}:
        raise ValueError("Labels must be binary values 0 and 1")
    return frame.drop_duplicates(subset="text").reset_index(drop=True)


def _stratified_sample(
    frame: pd.DataFrame,
    sample_size: int | None,
    random_state: int,
) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(frame):
        return frame
    sampled, _ = train_test_split(
        frame,
        train_size=sample_size,
        random_state=random_state,
        stratify=frame["label"],
    )
    return sampled.reset_index(drop=True)


def load_splits(
    cache_dir: Path,
    validation_size: float = 0.2,
    random_state: int = 42,
    sample_size: int | None = None,
) -> DatasetSplits:
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")
    if sample_size is not None and sample_size < 100:
        raise ValueError("sample_size must be at least 100")

    source_train = _read_split(TRAIN_URL, cache_dir / "train.parquet", TRAIN_SHA256)
    source_test = _read_split(TEST_URL, cache_dir / "test.parquet", TEST_SHA256)

    test_texts = set(source_test["text"])
    source_train = source_train[~source_train["text"].isin(test_texts)].reset_index(drop=True)

    source_train = _stratified_sample(source_train, sample_size, random_state)
    source_test = _stratified_sample(source_test, sample_size, random_state + 1)

    train, validation = train_test_split(
        source_train,
        test_size=validation_size,
        random_state=random_state,
        stratify=source_train["label"],
    )
    return DatasetSplits(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=source_test.reset_index(drop=True),
    )
