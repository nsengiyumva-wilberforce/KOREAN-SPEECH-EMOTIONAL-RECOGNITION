"""Build SpeechBrain manifests for the KITE speech-emotion subset.

Keeps only anger, happiness, neutral, sadness, and surprise.
Uses the 80/10/10 train/dev/test assignment already stored in the metadata.
"""

import json
import os

import pandas as pd

from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)

KEEP_EMOTIONS = ("anger", "happiness", "neutral", "sadness", "surprise")
SPLIT_TO_JSON = {"train": "train", "dev": "valid", "valid": "valid", "test": "test"}


def prepare_data(
    data_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    metadata_csv=None,
    emotions=KEEP_EMOTIONS,
    train_wav_folder=None,
):
    """Write train, valid, and test JSON manifests for KITE.

    Arguments
    ---------
    data_folder : str
        Directory that contains the KITE wav files and, by default,
        ``1_metadata_ser.csv``.
    save_json_train : str
        Path of the training manifest.
    save_json_valid : str
        Path of the validation manifest. The CSV calls this split ``dev``.
    save_json_test : str
        Path of the test manifest.
    metadata_csv : str
        CSV with columns ``audio_path``, ``emotion_label``, ``duration``,
        and ``split``. Defaults to ``1_metadata_ser.csv`` inside
        ``data_folder``.
    emotions : list
        Emotion names to keep. Comparison is case-insensitive.
    train_wav_folder : str
        Directory of voice-only training wavs, same file names as the
        originals. Validation and test keep the drama wavs in
        ``data_folder``. Empty or None leaves training on the originals.
    """
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation already completed, skipping.")
        return

    if metadata_csv is None:
        metadata_csv = os.path.join(data_folder, "1_metadata_ser.csv")

    keep = {name.strip().lower() for name in emotions}
    df = pd.read_csv(metadata_csv)
    required = {"audio_path", "emotion_label", "duration", "split"}
    missing_cols = required.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Metadata is missing columns: {sorted(missing_cols)}")

    df["emotion_label"] = df["emotion_label"].astype(str).str.strip().str.lower()
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    before = len(df)
    df = df[df["emotion_label"].isin(keep)].copy()
    logger.info(
        "Kept %d of %d utterances for emotions %s",
        len(df),
        before,
        ", ".join(sorted(keep)),
    )

    if train_wav_folder:
        train_wav_folder = os.path.abspath(train_wav_folder)

    manifests = {"train": {}, "valid": {}, "test": {}}
    missing_wavs = 0
    unknown_split = 0
    missing_separated = []

    for row in df.itertuples(index=False):
        split_name = SPLIT_TO_JSON.get(row.split)
        if split_name is None:
            unknown_split += 1
            continue

        wav_name = os.path.basename(row.audio_path)
        wav_path = os.path.join(data_folder, wav_name)
        if not os.path.isfile(wav_path):
            missing_wavs += 1
            continue
        if split_name == "train" and train_wav_folder:
            separated = os.path.join(train_wav_folder, wav_name)
            if not os.path.isfile(separated):
                missing_separated.append(wav_name)
                continue
            wav_path = separated

        uttid = os.path.splitext(wav_name)[0]
        manifests[split_name][uttid] = {
            "wav": wav_path,
            "length": float(row.duration),
            "emo": row.emotion_label,
        }

    if unknown_split:
        logger.warning("Dropped %d rows with an unknown split.", unknown_split)
    if missing_wavs:
        logger.warning("Dropped %d rows whose wav file is missing.", missing_wavs)
    if missing_separated:
        raise ValueError(
            f"{len(missing_separated)} training clips have no voice-only wav "
            f"in {train_wav_folder}. Finish separate_train.py before training."
        )
    if train_wav_folder:
        logger.info("Training wavs read from %s", train_wav_folder)

    outputs = {
        "train": save_json_train,
        "valid": save_json_valid,
        "test": save_json_test,
    }
    for split_name, path in outputs.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, mode="w", encoding="utf-8") as json_f:
            json.dump(manifests[split_name], json_f, indent=2)
        logger.info(
            "%s: %d utterances -> %s",
            split_name,
            len(manifests[split_name]),
            path,
        )

    if not manifests["train"]:
        raise ValueError("Training manifest is empty. Check data_folder and the CSV.")


def skip(*filenames):
    """Return True when every manifest file already exists."""
    return all(os.path.isfile(filename) for filename in filenames)
