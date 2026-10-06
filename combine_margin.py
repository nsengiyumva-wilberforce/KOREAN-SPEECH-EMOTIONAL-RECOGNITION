"""Combine the class-weighted XLS-R model with the balanced-batch model.

The class-weighted model is the logit-adjusted checkpoint (tau 0.75).
When it predicts neutral and the balanced model predicts anger or sadness
by a gap of 0.8, the balanced prediction is used. Surprise uses its own
gap, chosen on validation, and the test set is scored once.
"""

import os
import sys

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from train_with_wav2vec2 import (
    EmoIdBrain,
    _confusion_from_preds,
    _format_confusion,
    _map_emotion,
    _wa_ua,
    dataio_prep,
)


RECIPE = os.path.dirname(os.path.abspath(__file__))
WEIGHTED = os.path.join(RECIPE, "results", "kite_xlsr", "1993", "hyperparams.yaml")
BALANCED = os.path.join(
    RECIPE, "results", "kite_xlsr_balanced", "1993", "hyperparams.yaml"
)
TAU = 0.75
RARE_MARGIN = 0.8
SURPRISE_MARGINS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
MIN_WA = 0.50


def load_brain(hparams_path):
    """Load one finished experiment and the checkpoint selected in training."""
    with open(hparams_path, encoding="utf-8") as handle:
        hparams = load_hyperpyyaml(handle)
    hparams["skip_prep"] = True
    datasets = dataio_prep(hparams)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hparams["wav2vec2"] = hparams["wav2vec2"].to(device)
    if not hparams["freeze_wav2vec2"] and hparams["freeze_wav2vec2_conv"]:
        hparams["wav2vec2"].model.feature_extractor._freeze_parameters()
    brain = EmoIdBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts={"device": device},
        checkpointer=hparams["checkpointer"],
    )
    brain.on_evaluate_start(max_key=hparams["select_metric"])
    brain.modules.eval()
    return brain, datasets, hparams


def collect(brain, dataset, loader_kwargs):
    """Return utterance ids, log-probabilities, and labels in id order."""
    loader_kwargs = dict(loader_kwargs)
    loader_kwargs.pop("shuffle", None)
    loader = brain.make_dataloader(
        dataset, sb.Stage.TEST, ckpt_prefix=None, **loader_kwargs
    )
    ids = []
    log_probs = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            log_probs.append(brain.compute_forward(batch, sb.Stage.TEST).cpu())
            emoid, _ = batch.emo_encoded
            labels.append(emoid.squeeze(1).detach().cpu())
            ids.extend(list(batch.id))
    order = sorted(range(len(ids)), key=lambda index: ids[index])
    ids = [ids[index] for index in order]
    log_probs = torch.cat(log_probs)[order]
    labels = torch.cat(labels)[order]
    return ids, log_probs, labels


def train_log_prior(datasets, hparams):
    """Log of the training-set class frequencies, in encoder order."""
    encoder = hparams["label_encoder"]
    counts = torch.zeros(len(hparams["class_names"]))
    for utt_id in datasets["train"].data_ids:
        emo = _map_emotion(datasets["train"].data[utt_id]["emo"], hparams)
        counts[encoder.encode_label(emo)] += 1
    return (counts / counts.sum()).clamp(min=1e-8).log()


def fuse(weighted_log, balanced_log, class_names, surprise_margin):
    """Override neutral when the balanced model is clearly a rare class.

    Anger and sadness keep the gap already chosen for the shared rule.
    Surprise uses surprise_margin.
    """
    names = {name: index for index, name in enumerate(class_names)}
    neutral = names["neutral"]
    weighted_pred = weighted_log.argmax(dim=-1)
    balanced_prob = balanced_log.exp()
    balanced_pred = balanced_prob.argmax(dim=-1)
    rare_prob = balanced_prob.gather(1, balanced_pred.unsqueeze(1)).squeeze(1)
    gap = rare_prob - balanced_prob[:, neutral]
    anger_sad = (balanced_pred == names["anger"]) | (
        balanced_pred == names["sadness"]
    )
    clear = (anger_sad & (gap >= RARE_MARGIN)) | (
        (balanced_pred == names["surprise"]) & (gap >= surprise_margin)
    )
    use_balanced = (weighted_pred == neutral) & clear
    return torch.where(use_balanced, balanced_pred, weighted_pred), use_balanced


def score(predictions, labels, class_names):
    confusion = _confusion_from_preds(predictions, labels, len(class_names))
    wa, ua = _wa_ua(confusion)
    return confusion, wa, ua


def main():
    os.chdir(RECIPE)
    print("Loading class-weighted XLS-R", flush=True)
    weighted_brain, datasets, hparams = load_brain(WEIGHTED)
    loader_kwargs = hparams["dataloader_options"]
    names = hparams["class_names"]
    log_prior = train_log_prior(datasets, hparams)
    weighted = {}
    labels = {}
    for split in ("valid", "test"):
        ids, log_probs, split_labels = collect(
            weighted_brain, datasets[split], loader_kwargs
        )
        weighted[split] = log_probs - TAU * log_prior
        labels[split] = split_labels
        print(f"  {split}: {len(ids)} utterances", flush=True)
    del weighted_brain
    torch.cuda.empty_cache()

    print("Loading balanced-batch XLS-R", flush=True)
    balanced_brain, balanced_datasets, balanced_hparams = load_brain(BALANCED)
    balanced = {}
    for split in ("valid", "test"):
        ids, log_probs, split_labels = collect(
            balanced_brain, balanced_datasets[split], loader_kwargs
        )
        if not torch.equal(split_labels, labels[split]):
            raise RuntimeError(f"{split} labels differ between the two models")
        balanced[split] = log_probs
        print(f"  {split}: {len(ids)} utterances", flush=True)
    del balanced_brain
    torch.cuda.empty_cache()

    print(
        "surprise_margin  valid_WA  valid_UA  overrides  "
        "(anger and sadness stay at %.2f)" % RARE_MARGIN,
        flush=True,
    )
    rows = []
    for margin in SURPRISE_MARGINS:
        predictions, overrides = fuse(
            weighted["valid"], balanced["valid"], names, margin
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

    # The surprise gap is locked on validation before the test set is read.
    eligible = [
        row
        for row in rows
        if row["wa"] >= MIN_WA and row["overrides"] > 0
    ]
    if not eligible:
        eligible = [row for row in rows if row["wa"] >= MIN_WA]
    chosen = max(eligible, key=lambda row: (row["ua"], row["wa"], -row["margin"]))
    predictions, overrides = fuse(
        weighted["test"], balanced["test"], names, chosen["margin"]
    )
    confusion, wa, ua = score(predictions, labels["test"], names)
    surprise_index = names.index("surprise")
    surprise_overrides = int(
        (overrides & (predictions == surprise_index)).sum()
    )
    print(
        f"chosen surprise margin {chosen['margin']:.2f}  "
        f"test WA {wa * 100:.2f}%  test UA {ua * 100:.2f}%  "
        f"overrides {int(overrides.sum())}  "
        f"surprise overrides {surprise_overrides}",
        flush=True,
    )
    text = _format_confusion(confusion, names, wa, ua)
    out_dir = os.path.join(RECIPE, "results", "kite_fusion")
    os.makedirs(out_dir, exist_ok=True)
    report = os.path.join(out_dir, "confusion_test_surprise.txt")
    with open(report, "w", encoding="utf-8") as handle:
        handle.write(
            "Rule: logit-adjusted class-weighted XLS-R "
            f"(tau {TAU}), overridden when it predicts neutral and the "
            "balanced model predicts anger or sadness with "
            f"P(class) - P(neutral) >= {RARE_MARGIN}, or surprise with "
            f"P(surprise) - P(neutral) >= {chosen['margin']}.\n"
        )
        handle.write(
            "Surprise margin chosen on validation among settings with "
            f"valid WA >= {MIN_WA:.2f}, maximizing valid UA.\n"
        )
        handle.write(
            f"valid WA: {chosen['wa'] * 100:.2f}%  "
            f"valid UA: {chosen['ua'] * 100:.2f}%\n"
        )
        handle.write(
            f"test overrides: {int(overrides.sum())}  "
            f"surprise overrides: {surprise_overrides}\n\n"
        )
        handle.write("surprise_margin  valid_WA  valid_UA  overrides\n")
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
    row_name = "Weighted plus surprise margin"
    kept = []
    if os.path.exists(table):
        with open(table, encoding="utf-8") as handle:
            kept = [
                line
                for line in handle
                if not line.startswith(row_name + "\t")
            ]
    kept.append(
        f"{row_name}\t{wa * 100:.2f}%\t{ua * 100:.2f}%\n"
    )
    with open(table, "w", encoding="utf-8") as handle:
        handle.writelines(kept)
    return 0


if __name__ == "__main__":
    sys.exit(main())
