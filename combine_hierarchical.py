"""Combine the neutral-versus-rest model with the four-emotion model.

If the first model predicts neutral, that is the final label. Otherwise the
second model's prediction is used. Both prediction files store the original
five-class emotion in the true column.
"""

import csv
import os
import sys

import torch

from train_with_wav2vec2 import _format_confusion, _wa_ua


EMOTIONS = ["anger", "happiness", "neutral", "sadness", "surprise"]


def load_predictions(path):
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row["id"]] = row
    return rows


def main():
    recipe = os.path.dirname(os.path.abspath(__file__))
    binary_path = os.path.join(
        recipe, "results", "kite_binary", "1993", "predictions_test.csv"
    )
    emotional_path = os.path.join(
        recipe, "results", "kite_emotional", "1993", "predictions_test.csv"
    )
    out_dir = os.path.join(recipe, "results", "kite_hierarchical")
    os.makedirs(out_dir, exist_ok=True)

    binary = load_predictions(binary_path)
    emotional = load_predictions(emotional_path)
    index = {name: i for i, name in enumerate(EMOTIONS)}
    confusion = torch.zeros(len(EMOTIONS), len(EMOTIONS))
    for utt_id, row in binary.items():
        true_name = row["true"]
        if row["prediction"] == "neutral":
            pred_name = "neutral"
        else:
            pred_name = emotional[utt_id]["prediction"]
        confusion[index[true_name], index[pred_name]] += 1

    wa, ua = _wa_ua(confusion)
    text = _format_confusion(confusion, EMOTIONS, wa, ua)
    report = os.path.join(out_dir, "confusion_test.txt")
    with open(report, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")
    print(text)


if __name__ == "__main__":
    sys.exit(main())
