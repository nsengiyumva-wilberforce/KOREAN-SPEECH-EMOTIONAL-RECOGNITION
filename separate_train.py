#!/usr/bin/env python3
"""Write voice-only copies of the KITE training clips.

Demucs keeps the vocal stem and drops the music and effects bed.
Validation and test clips are not read. The original wavs are not replaced.
Re-running skips files that are already written.
"""

import argparse
import os

import pandas as pd
import torch
import torchaudio
from demucs.apply import apply_model
from demucs.audio import convert_audio
from demucs.pretrained import get_model

RECIPE = os.path.dirname(os.path.abspath(__file__))
CSV = (
    "/media/computergeek/SER-datasets/korean-audio-sentiment-analysis/"
    "data/KITE/1_metadata_ser.csv"
)
OUT = os.path.join(RECIPE, "separated_train")
EMOTIONS = {"anger", "happiness", "neutral", "sadness", "surprise"}
SAMPLE_RATE = 16000


def train_rows(csv_path):
    frame = pd.read_csv(csv_path)
    frame["emotion_label"] = frame["emotion_label"].astype(str).str.strip().str.lower()
    frame["split"] = frame["split"].astype(str).str.strip().str.lower()
    frame = frame[
        (frame["split"] == "train") & frame["emotion_label"].isin(EMOTIONS)
    ]
    rows = []
    for row in frame.itertuples(index=False):
        name = os.path.basename(row.audio_path)
        src = row.audio_path
        if not os.path.isfile(src):
            src = os.path.join(os.path.dirname(csv_path), name)
        rows.append((src, name))
    return rows


def separate_one(model, src, dest, device):
    wav, sr = torchaudio.load(src)
    wav = convert_audio(wav, sr, model.samplerate, model.audio_channels)
    ref = wav.mean(0)
    std = ref.std().clamp_min(1e-8)
    wav = (wav - ref.mean()) / std
    with torch.no_grad():
        sources = apply_model(
            model,
            wav[None],
            device=device,
            shifts=0,
            split=True,
            overlap=0.25,
            progress=False,
        )[0]
    sources = sources * std + ref.mean()
    vocals = sources[model.sources.index("vocals")]
    vocals = convert_audio(vocals, model.samplerate, SAMPLE_RATE, 1)
    vocals = vocals.clamp(-1.0, 1.0).cpu()
    temporary = dest + ".partial.wav"
    torchaudio.save(
        temporary, vocals, SAMPLE_RATE, encoding="PCM_S", bits_per_sample=16
    )
    os.replace(temporary, dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default=CSV)
    parser.add_argument("--out_dir", default=OUT)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = train_rows(args.metadata)
    if args.limit:
        rows = rows[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Separating {len(rows)} training clips on {device} -> {args.out_dir}")
    model = get_model("htdemucs")
    model.eval()
    written = 0
    skipped = 0
    for index, (src, name) in enumerate(rows, start=1):
        dest = os.path.join(args.out_dir, name)
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            skipped += 1
            continue
        if not os.path.isfile(src):
            raise FileNotFoundError(src)
        separate_one(model, src, dest, device)
        written += 1
        if written == 1 or written % 50 == 0:
            print(f"{index}/{len(rows)}  wrote {written}  skipped {skipped}  {name}")
    print(f"Done. wrote {written}, skipped {skipped}, folder {args.out_dir}")


if __name__ == "__main__":
    main()
