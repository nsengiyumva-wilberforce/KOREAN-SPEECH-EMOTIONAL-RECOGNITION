"""Override HuBERT with the balanced-batch model on anger and sadness only.

HuBERT is the epoch selected by validation macro-F1, then logit-adjusted
with tau 0.75. When that system predicts neutral and the balanced model
predicts anger or sadness by a gap chosen on validation, the balanced
label is used. The test set is scored once.
"""

import os
import sys

import torch

from combine_margin import (
    collect,
    load_brain,
    score,
    train_log_prior,
)
from train_with_wav2vec2 import _format_confusion


RECIPE = os.path.dirname(os.path.abspath(__file__))
HUBERT = os.path.join(
    RECIPE, "results", "kite_hubert_large_korean", "1993", "hyperparams.yaml"
)
BALANCED = os.path.join(
    RECIPE, "results", "kite_xlsr_balanced", "1993", "hyperparams.yaml"
)
TAU = 0.75
MARGINS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
MIN_WA = 0.50
OVERRIDE = ("anger", "sadness")


def fuse(hubert_log, balanced_log, class_names, margin):
    """Use the balanced label when it is clearly anger or sadness."""
    names = {name: index for index, name in enumerate(class_names)}
    neutral = names["neutral"]
    hubert_pred = hubert_log.argmax(dim=-1)
    balanced_prob = balanced_log.exp()
    balanced_pred = balanced_prob.argmax(dim=-1)
    chosen_prob = balanced_prob.gather(1, balanced_pred.unsqueeze(1)).squeeze(1)
    gap = chosen_prob - balanced_prob[:, neutral]
    rare = torch.zeros(len(balanced_pred), dtype=torch.bool)
    for name in OVERRIDE:
        rare |= balanced_pred == names[name]
    use_balanced = (hubert_pred == neutral) & rare & (gap >= margin)
    return torch.where(use_balanced, balanced_pred, hubert_pred), use_balanced


def main():
    os.chdir(RECIPE)
    print("Loading HuBERT large Korean", flush=True)
    hubert_brain, datasets, hparams = load_brain(HUBERT)
    loader_kwargs = hparams["dataloader_options"]
    names = list(hparams["class_names"])
    log_prior = train_log_prior(datasets, hparams)
    hubert = {}
    labels = {}
    ids = {}
    for split in ("valid", "test"):
        split_ids, log_probs, split_labels = collect(
            hubert_brain, datasets[split], loader_kwargs
        )
        hubert[split] = log_probs - TAU * log_prior
        labels[split] = split_labels
        ids[split] = split_ids
        print(f"  {split}: {len(split_ids)} utterances", flush=True)
    del hubert_brain
    torch.cuda.empty_cache()

    print("Loading balanced-batch XLS-R", flush=True)
    balanced_brain, balanced_datasets, balanced_hparams = load_brain(BALANCED)
    if list(balanced_hparams["class_names"]) != names:
        raise RuntimeError("The two models do not use the same class order")
    balanced = {}
    for split in ("valid", "test"):
        split_ids, log_probs, split_labels = collect(
            balanced_brain, balanced_datasets[split], loader_kwargs
        )
        if split_ids != ids[split]:
            raise RuntimeError(f"{split} utterance ids differ between the two models")
        if not torch.equal(split_labels, labels[split]):
            raise RuntimeError(f"{split} labels differ between the two models")
        balanced[split] = log_probs
        print(f"  {split}: {len(split_ids)} utterances", flush=True)
    del balanced_brain
    torch.cuda.empty_cache()

    print("margin  valid_WA  valid_UA  overrides", flush=True)
    rows = []
    for margin in MARGINS:
        predictions, overrides = fuse(
            hubert["valid"], balanced["valid"], names, margin
        )
        _, wa, ua = score(predictions, labels["valid"], names)
        n_overrides = int(overrides.sum())
        print(
            f"{margin:5.2f}  {wa * 100:7.2f}%  {ua * 100:7.2f}%  {n_overrides}",
            flush=True,
        )
        rows.append(
            {
                "margin": margin,
                "wa": wa,
                "ua": ua,
                "overrides": n_overrides,
            }
        )

    eligible = [
        row for row in rows if row["wa"] >= MIN_WA and row["overrides"] > 0
    ]
    if not eligible:
        eligible = [row for row in rows if row["wa"] >= MIN_WA]
    chosen = max(eligible, key=lambda row: (row["ua"], row["wa"], row["margin"]))
    predictions, overrides = fuse(
        hubert["test"], balanced["test"], names, chosen["margin"]
    )
    confusion, wa, ua = score(predictions, labels["test"], names)
    anger_index = names.index("anger")
    sadness_index = names.index("sadness")
    anger_overrides = int((overrides & (predictions == anger_index)).sum())
    sadness_overrides = int((overrides & (predictions == sadness_index)).sum())
    print(
        f"chosen margin {chosen['margin']:.2f}  "
        f"test WA {wa * 100:.2f}%  test UA {ua * 100:.2f}%  "
        f"overrides {int(overrides.sum())}  "
        f"anger {anger_overrides}  sadness {sadness_overrides}",
        flush=True,
    )
    text = _format_confusion(confusion, names, wa, ua)
    report = os.path.join(
        RECIPE,
        "results",
        "kite_hubert_large_korean",
        "1993",
        "confusion_test_balanced.txt",
    )
    with open(report, "w", encoding="utf-8") as handle:
        handle.write(
            "Rule: HuBERT large Korean, logit-adjusted with "
            f"tau {TAU}, overridden when it predicts neutral and the "
            "balanced-batch XLS-R predicts anger or sadness with "
            f"P(class) - P(neutral) >= {chosen['margin']}.\n"
        )
        handle.write(
            "Margin chosen on validation among settings with "
            f"valid WA >= {MIN_WA:.2f}, maximizing valid UA.\n"
        )
        handle.write(
            f"valid WA: {chosen['wa'] * 100:.2f}%  "
            f"valid UA: {chosen['ua'] * 100:.2f}%\n"
        )
        handle.write(
            f"test overrides: {int(overrides.sum())}  "
            f"anger: {anger_overrides}  sadness: {sadness_overrides}\n\n"
        )
        handle.write("margin  valid_WA  valid_UA  overrides\n")
        for row in rows:
            handle.write(
                f"{row['margin']:5.2f}  {row['wa'] * 100:7.2f}%  "
                f"{row['ua'] * 100:7.2f}%  {row['overrides']}\n"
            )
        handle.write("\n")
        handle.write(text)
        handle.write("\n")
    print(text, flush=True)

    table = os.path.join(RECIPE, "results", "comparison_table.txt")
    row_name = "HuBERT plus balanced anger sadness"
    kept = []
    if os.path.exists(table):
        with open(table, encoding="utf-8") as handle:
            kept = [
                line
                for line in handle
                if not line.startswith(row_name + "\t")
            ]
    kept.append(f"{row_name}\t{wa * 100:.2f}%\t{ua * 100:.2f}%\n")
    with open(table, "w", encoding="utf-8") as handle:
        handle.writelines(kept)
    return 0


if __name__ == "__main__":
    sys.exit(main())
