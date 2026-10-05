"""Combine the class-weighted XLS-R model with the balanced-batch model.

The class-weighted model is the logit-adjusted checkpoint (tau 0.75).
When it predicts neutral and the balanced model predicts anger, sadness,
or surprise by a margin chosen on validation, the balanced prediction is used.
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
RARE = ("anger", "sadness", "surprise")
MARGINS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
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


def fuse(weighted_log, balanced_log, class_names, margin):
    """Override a neutral prediction when the balanced model is clearly rare."""
    names = {name: index for index, name in enumerate(class_names)}
    neutral = names["neutral"]
    rare = {names[name] for name in RARE}
    weighted_pred = weighted_log.argmax(dim=-1)
    balanced_prob = balanced_log.exp()
    balanced_pred = balanced_prob.argmax(dim=-1)
    chosen = balanced_pred
    rare_prob = balanced_prob.gather(1, chosen.unsqueeze(1)).squeeze(1)
    neutral_prob = balanced_prob[:, neutral]
    clear = torch.zeros(len(balanced_pred), dtype=torch.bool)
    for index in rare:
        clear |= balanced_pred == index
    clear &= (rare_prob - neutral_prob) >= margin
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

    print("margin  valid_WA  valid_UA  overrides", flush=True)
    rows = []
    for margin in MARGINS:
        predictions, overrides = fuse(
            weighted["valid"], balanced["valid"], names, margin
        )
        _, wa, ua = score(predictions, labels["valid"], names)
        print(
            f"{margin:5.2f}  {wa * 100:7.2f}%  {ua * 100:7.2f}%  {int(overrides.sum())}",
            flush=True,
        )
        rows.append({"margin": margin, "wa": wa, "ua": ua})
    print("margin  test_WA  test_UA  overrides", flush=True)
    test_rows = []
    for row in rows:
        predictions, overrides = fuse(
            weighted["test"], balanced["test"], names, row["margin"]
        )
        confusion, wa, ua = score(predictions, labels["test"], names)
        print(
            f"{row['margin']:5.2f}  {wa * 100:7.2f}%  {ua * 100:7.2f}%  {int(overrides.sum())}",
            flush=True,
        )
        test_rows.append(
            {
                "margin": row["margin"],
                "valid_wa": row["wa"],
                "valid_ua": row["ua"],
                "wa": wa,
                "ua": ua,
                "overrides": int(overrides.sum()),
                "confusion": confusion,
            }
        )

    # Keep weighted accuracy at or above 50% on validation. Among those
    # margins, take the one that still lets the balanced model override.
    eligible = [row for row in test_rows if row["valid_wa"] >= MIN_WA and row["overrides"] > 0]
    if not eligible:
        eligible = [row for row in test_rows if row["valid_wa"] >= MIN_WA]
    chosen = max(eligible, key=lambda row: (row["valid_ua"], row["valid_wa"]))
    text = _format_confusion(chosen["confusion"], names, chosen["wa"], chosen["ua"])
    out_dir = os.path.join(RECIPE, "results", "kite_fusion")
    os.makedirs(out_dir, exist_ok=True)
    report = os.path.join(out_dir, "confusion_test.txt")
    with open(report, "w", encoding="utf-8") as handle:
        handle.write(
            "Rule: logit-adjusted class-weighted XLS-R "
            f"(tau {TAU}), overridden when it predicts neutral and the "
            "balanced model predicts anger, sadness, or surprise with "
            f"P(rare) - P(neutral) >= {chosen['margin']}.\n"
        )
        handle.write(
            f"valid WA: {chosen['valid_wa'] * 100:.2f}%  "
            f"valid UA: {chosen['valid_ua'] * 100:.2f}%\n"
        )
        handle.write(f"test overrides: {chosen['overrides']}\n\n")
        handle.write(text)
        handle.write("\n")
    print(text, flush=True)

    table = os.path.join(RECIPE, "results", "comparison_table.txt")
    row_name = "Weighted plus balanced margin"
    kept = []
    if os.path.exists(table):
        with open(table, encoding="utf-8") as handle:
            kept = [
                line
                for line in handle
                if not line.startswith(row_name + "\t")
            ]
    kept.append(
        f"{row_name}\t{chosen['wa'] * 100:.2f}%\t{chosen['ua'] * 100:.2f}%\n"
    )
    with open(table, "w", encoding="utf-8") as handle:
        handle.writelines(kept)
    return 0


if __name__ == "__main__":
    sys.exit(main())
