#!/usr/bin/env python3
"""Recipe for training emotion recognition on KITE speech segments with wav2vec2.

The system classifies 5 emotions (anger, happiness, neutral, sadness, surprise).
Other emotions in the metadata are dropped. The CSV split is train/dev/test,
already 80/10/10. The dev split is used as validation.

To run this recipe, do the following:
> python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml
"""

import os
import random
import sys

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class EmoIdBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + emotion classifier."""
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        outputs = self.modules.wav2vec2(wavs, lens)

        # last dim will be used for AdaptiveAVG pool
        outputs = self.hparams.avg_pool(outputs, lens)
        outputs = outputs.view(outputs.shape[0], -1)

        outputs = self.modules.output_mlp(outputs)
        outputs = self.hparams.log_softmax(outputs)
        return outputs

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label."""
        emoid, _ = batch.emo_encoded

        """to meet the input form of nll loss"""
        emoid = emoid.squeeze(1)
        weight = self.hparams.class_weight.to(predictions.device)
        loss = self.hparams.compute_cost(predictions, emoid, weight=weight)
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, emoid)
            predicted = predictions.argmax(dim=-1)
            n_classes = self.hparams.class_weight.numel()
            indices = emoid * n_classes + predicted
            counts = torch.bincount(
                indices.detach().cpu(), minlength=n_classes * n_classes
            )
            self.confusion += counts.reshape(n_classes, n_classes)
            if self.hparams.save_predictions:
                names = self.hparams.class_names
                raw = batch.raw_emo
                if isinstance(raw, (tuple, list)) and raw and not isinstance(raw[0], str):
                    raw = raw[0]
                for utt_id, pred_i, true_name in zip(
                    batch.id, predicted.detach().cpu().tolist(), raw
                ):
                    self.pred_rows.append((utt_id, names[pred_i], true_name))

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()
            n_classes = self.hparams.class_weight.numel()
            self.confusion = torch.zeros(n_classes, n_classes)
            self.pred_rows = []

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            macro_f1 = _macro_f1(self.confusion)
            wa, ua = _wa_ua(self.confusion)
            stats = {
                "loss": stage_loss,
                "error_rate": self.error_metrics.summarize("average"),
                "macro_f1": macro_f1,
                # Lower is better, so NewBobScheduler can anneal on it.
                "balanced_error": 1.0 - macro_f1,
                "wa": wa,
                "ua": ua,
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:
            if self.hparams.select_metric == "ua":
                anneal_value = 1.0 - stats["ua"]
            else:
                anneal_value = stats["balanced_error"]
            old_lr, new_lr = self.hparams.lr_annealing(anneal_value)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            (
                old_lr_wav2vec2,
                new_lr_wav2vec2,
            ) = self.hparams.lr_annealing_wav2vec2(anneal_value)
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec2_optimizer, new_lr_wav2vec2
            )

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr, "wave2vec_lr": old_lr_wav2vec2},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta=stats,
                max_keys=[self.hparams.select_metric],
                keep_recent=False,
            )

        if stage != sb.Stage.TRAIN:
            split = "valid" if stage == sb.Stage.VALID else "test"
            split = getattr(self, "report_split", split)
            self._write_confusion(split)

        # We also write statistics about test data to stdout and to logfile.
        if stage == sb.Stage.TEST:
            split = getattr(self, "report_split", "test")
            stat_name = "valid_stats" if split == "valid" else "test_stats"
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                **{stat_name: stats},
            )

    def _write_confusion(self, split):
        """Write WA, UA, and the confusion matrix for this evaluation split."""
        names = self.hparams.class_names
        wa, ua = _wa_ua(self.confusion)
        text = _format_confusion(self.confusion, names, wa, ua)
        path = os.path.join(
            self.hparams.output_folder, f"confusion_{split}.txt"
        )
        with open(path, "w", encoding="utf-8") as report:
            report.write(text)
            report.write("\n")
        if self.hparams.save_predictions and self.pred_rows:
            pred_path = os.path.join(
                self.hparams.output_folder, f"predictions_{split}.csv"
            )
            with open(pred_path, "w", encoding="utf-8") as pred_file:
                pred_file.write("id,prediction,true\n")
                for utt_id, pred_name, true_name in self.pred_rows:
                    pred_file.write(f"{utt_id},{pred_name},{true_name}\n")
        logger.info("Wrote %s\n%s", path, text)

    def on_fit_start(self):
        """Freeze the lowest transformer layers before the optimizer is built."""
        n_freeze = int(self.hparams.freeze_transformer_layers)
        if n_freeze > 0:
            layers = self.modules.wav2vec2.model.encoder.layers
            for layer in list(layers)[:n_freeze]:
                for param in layer.parameters():
                    param.requires_grad = False
            logger.info("Froze the first %d wav2vec transformer layers", n_freeze)
        super().on_fit_start()

    def make_dataloader(self, dataset, stage, ckpt_prefix="dataloader-", **loader_kwargs):
        """Use class-balanced draws for training when that switch is on."""
        if stage == sb.Stage.TRAIN and self.hparams.balanced_sampling:
            loader_kwargs = dict(loader_kwargs)
            loader_kwargs.pop("shuffle", None)
            loader_kwargs["sampler"] = _balanced_sampler(
                dataset, self.hparams.label_encoder, self.hparams
            )
        return super().make_dataloader(
            dataset, stage, ckpt_prefix=ckpt_prefix, **loader_kwargs
        )

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        wav2vec_params = [
            param
            for param in self.modules.wav2vec2.parameters()
            if param.requires_grad
        ]
        self.wav2vec2_optimizer = self.hparams.wav2vec2_opt_class(wav2vec_params)
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        # Optimizer moments are left out of the checkpoint. Adam state for
        # XLS-R is about 2 GB, and the disk cannot hold that for every run.
        self.optimizers_dict = {
            "model_optimizer": self.optimizer,
            "wav2vec2_optimizer": self.wav2vec2_optimizer,
        }


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined
    functions. We expect `prepare_mini_librispeech` to have been called before
    this, so that the `train.json`, `valid.json`,  and `valid.json` manifest
    files are available.
    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.
    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """

    # Long KITE turns (up to ~90s) are padded to the longest item in the
    # batch. That exceeds 8 GB on wav2vec2. Crop to max_wav_seconds.
    max_len = int(hparams["sample_rate"] * hparams["max_wav_seconds"])
    augment_rare = bool(hparams.get("augment_rare", False))
    augment_prob = float(hparams.get("augment_prob", 0.8))
    rare_emotions = {"anger", "sadness", "surprise"}

    def make_audio_pipeline(random_crop):
        @sb.utils.data_pipeline.takes("wav", "emo")
        @sb.utils.data_pipeline.provides("sig")
        def audio_pipeline(wav, emo):
            """Load the waveform and crop it when it is longer than the cap."""
            sig = sb.dataio.dataio.read_audio(wav)
            if sig.shape[0] > max_len:
                if random_crop:
                    start = random.randint(0, sig.shape[0] - max_len)
                else:
                    start = (sig.shape[0] - max_len) // 2
                sig = sig[start : start + max_len]
            if (
                random_crop
                and augment_rare
                and emo in rare_emotions
                and random.random() < augment_prob
            ):
                sig = _augment_waveform(sig, hparams["sample_rate"], max_len)
            return sig

        return audio_pipeline

    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, 'spk01': 0, 'spk02': 1, ..)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    encoder_ready = {"done": False}

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("emo")
    @sb.utils.data_pipeline.provides("raw_emo", "emo", "emo_encoded")
    def label_pipeline(emo):
        yield emo
        mapped = _map_emotion(emo, hparams)
        if encoder_ready["done"] and mapped not in label_encoder.lab2ind:
            mapped = next(iter(label_encoder.ind2lab.values()))
        yield mapped
        emo_encoded = label_encoder.encode_label_torch(mapped)
        yield emo_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[
                make_audio_pipeline(random_crop=(dataset == "train")),
                label_pipeline,
            ],
            output_keys=["id", "sig", "raw_emo", "emo_encoded"],
        )
    if hparams["label_mode"] == "emotional":
        # Train and validation drop neutral. Test stays complete so a
        # hierarchical model can score clips the first stage calls emotional.
        for split in ("train", "valid"):
            dataset = datasets[split]
            dataset.data_ids = [
                utt_id
                for utt_id in dataset.data_ids
                if dataset.data[utt_id]["emo"] != hparams["neutral_name"]
            ]
    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mapping.

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="emo",
    )
    label_encoder.expect_len(hparams["out_n_neurons"])
    encoder_ready["done"] = True
    hparams["class_weight"] = _class_weights(
        label_encoder, datasets["train"], hparams
    )
    hparams["class_names"] = [
        label_encoder.ind2lab[i] for i in range(len(label_encoder))
    ]
    hparams["label_encoder"] = label_encoder
    logger.info(
        "Class weights (index order %s): %s",
        hparams["class_names"],
        [round(weight, 3) for weight in hparams["class_weight"].tolist()],
    )
    if augment_rare:
        logger.info(
            "Training augmentation on %s with probability %.2f",
            sorted(rare_emotions),
            augment_prob,
        )

    return datasets


def _augment_waveform(sig, sample_rate, max_len):
    """Vary speed and loudness, and sometimes mute a short span.

    Speed 90 and 110 are percentages of the original rate. The result is
    cropped again so a slowed clip still fits in the 16 second cap.
    """
    speed = random.choice((90, 100, 110))
    if speed != 100:
        new_freq = int(round(sample_rate * 100 / speed))
        sig = torchaudio.functional.resample(sig, sample_rate, new_freq)
    sig = sig * random.uniform(0.8, 1.2)
    if sig.shape[0] > int(0.4 * sample_rate) and random.random() < 0.5:
        width = max(1, int(sig.shape[0] * random.uniform(0.05, 0.15)))
        width = min(width, sig.shape[0] - 1)
        start = random.randint(0, sig.shape[0] - width)
        sig = sig.clone()
        sig[start : start + width] = 0
    if sig.shape[0] > max_len:
        start = random.randint(0, sig.shape[0] - max_len)
        sig = sig[start : start + max_len]
    return sig


def _hparam(hparams, key):
    """Read a hyperparameter from the YAML dict or the Brain namespace."""
    if isinstance(hparams, dict):
        return hparams[key]
    return getattr(hparams, key)


def _map_emotion(emo, hparams):
    """Collapse the five emotions when training the neutral-versus-rest model."""
    if _hparam(hparams, "label_mode") == "binary":
        if emo == _hparam(hparams, "neutral_name"):
            return "neutral"
        return "emotional"
    return emo


def _balanced_sampler(dataset, label_encoder, hparams):
    """Draw each emotion about equally often within an epoch."""
    labels = []
    for utt_id in dataset.data_ids:
        emo = _map_emotion(dataset.data[utt_id]["emo"], hparams)
        labels.append(label_encoder.encode_label(emo))
    labels = torch.tensor(labels, dtype=torch.long)
    counts = torch.bincount(labels).float().clamp(min=1)
    weights = 1.0 / counts[labels]
    return torch.utils.data.WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True
    )


def _class_weights(label_encoder, train_dataset, hparams):
    """Class weights for the training loss.

    ``inverse`` uses total / (n_classes * count). ``inverse_sqrt`` uses the
    square root of that weight, which upweights rare emotions less strongly.
    """
    counts = torch.zeros(len(label_encoder))
    for utt_id in train_dataset.data_ids:
        emo = _map_emotion(train_dataset.data[utt_id]["emo"], hparams)
        counts[label_encoder.encode_label(emo)] += 1
    if torch.any(counts == 0):
        missing = [
            label_encoder.ind2lab[i]
            for i, count in enumerate(counts.tolist())
            if count == 0
        ]
        raise ValueError(f"Training set has no examples of {missing}")
    weights = counts.sum() / (len(counts) * counts)
    if isinstance(hparams, dict):
        mode = hparams.get("class_weighting", "inverse")
    else:
        mode = getattr(hparams, "class_weighting", "inverse")
    if mode == "inverse_sqrt":
        return weights.sqrt()
    return weights


def _wa_ua(confusion):
    """Weighted accuracy and unweighted accuracy.

    WA is overall accuracy. UA is the mean of per-class recall.
    """
    confusion = confusion.float()
    total = confusion.sum()
    wa = (confusion.diag().sum() / total).item() if total > 0 else 0.0
    support = confusion.sum(dim=1)
    recall = confusion.diag() / support.clamp(min=1)
    present = support > 0
    ua = recall[present].mean().item() if present.any() else 0.0
    return wa, ua


def _format_confusion(confusion, class_names, wa, ua):
    """Counts, with a row-normalized percentage under each count."""
    confusion = confusion.int()
    width = 14
    header = "true\\pred".ljust(width) + "".join(
        name.rjust(width) for name in class_names
    )
    support = confusion.sum(dim=1).clamp(min=1)
    lines = [
        f"WA (weighted accuracy): {wa * 100:.2f}%",
        f"UA (unweighted accuracy): {ua * 100:.2f}%",
        "",
        "Rows are the true emotion. Columns are the prediction.",
        header,
    ]
    for i, name in enumerate(class_names):
        counts = "".join(
            str(int(confusion[i, j])).rjust(width)
            for j in range(len(class_names))
        )
        percents = "".join(
            f"{100.0 * float(confusion[i, j]) / float(support[i]):.1f}%".rjust(
                width
            )
            for j in range(len(class_names))
        )
        lines.append(name.ljust(width) + counts)
        lines.append("".ljust(width) + percents)
    return "\n".join(lines)


def _macro_f1(confusion):
    """Unweighted mean of per-class F1. Classes with no examples are skipped."""
    confusion = confusion.float()
    true_positive = confusion.diag()
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    precision = true_positive / predicted.clamp(min=1)
    recall = true_positive / support.clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    present = support > 0
    return f1[present].mean().item()


def _confusion_from_preds(predictions, targets, n_classes):
    indices = targets * n_classes + predictions
    counts = torch.bincount(indices, minlength=n_classes * n_classes)
    return counts.reshape(n_classes, n_classes).float()


def _collect_log_probs(brain, dataset, loader_kwargs):
    """Run the loaded model and keep log-probabilities for logit adjustment."""
    loader_kwargs = dict(loader_kwargs)
    loader = brain.make_dataloader(
        dataset, sb.Stage.TEST, ckpt_prefix=None, **loader_kwargs
    )
    brain.modules.eval()
    log_probs = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            log_probs.append(brain.compute_forward(batch, sb.Stage.TEST).cpu())
            emoid, _ = batch.emo_encoded
            labels.append(emoid.squeeze(1).detach().cpu())
    return torch.cat(log_probs), torch.cat(labels)


def _run_logit_adjustment(brain, datasets, hparams):
    """Pick a prior-subtraction strength on validation, then score test."""
    brain.on_evaluate_start(max_key=hparams["select_metric"])
    valid_log, valid_y = _collect_log_probs(
        brain, datasets["valid"], hparams["dataloader_options"]
    )
    test_log, test_y = _collect_log_probs(
        brain, datasets["test"], hparams["dataloader_options"]
    )
    counts = torch.zeros(len(hparams["class_names"]))
    for utt_id in datasets["train"].data_ids:
        emo = _map_emotion(datasets["train"].data[utt_id]["emo"], hparams)
        counts[hparams["label_encoder"].encode_label(emo)] += 1
    log_prior = (counts / counts.sum()).clamp(min=1e-8).log()

    best = None
    for tau in hparams["logit_taus"]:
        adjusted = valid_log - float(tau) * log_prior
        confusion = _confusion_from_preds(
            adjusted.argmax(dim=-1), valid_y, len(hparams["class_names"])
        )
        wa, ua = _wa_ua(confusion)
        logger.info("logit tau %.2f  valid WA %.2f%%  UA %.2f%%", tau, wa * 100, ua * 100)
        if best is None or ua > best["ua"]:
            best = {"tau": float(tau), "wa": wa, "ua": ua}

    adjusted = test_log - best["tau"] * log_prior
    confusion = _confusion_from_preds(
        adjusted.argmax(dim=-1), test_y, len(hparams["class_names"])
    )
    wa, ua = _wa_ua(confusion)
    text = _format_confusion(confusion, hparams["class_names"], wa, ua)
    report = os.path.join(hparams["output_folder"], "confusion_test_logit.txt")
    with open(report, "w", encoding="utf-8") as handle:
        handle.write(f"tau: {best['tau']}\n")
        handle.write(
            f"valid WA: {best['wa'] * 100:.2f}%  valid UA: {best['ua'] * 100:.2f}%\n\n"
        )
        handle.write(text)
        handle.write("\n")
    logger.info(
        "Logit adjustment tau %.2f  test WA %.2f%%  UA %.2f%%\n%s",
        best["tau"],
        wa * 100,
        ua * 100,
        text,
    )


def _build_hidden_classifier(hparams):
    """Replace the linear head with a two-layer classifier when requested."""
    hidden = int(hparams["classifier_hidden"])
    if hidden <= 0:
        return
    mlp = torch.nn.Sequential(
        sb.nnet.linear.Linear(
            input_size=hparams["encoder_dim"], n_neurons=hidden
        ),
        torch.nn.ReLU(),
        sb.nnet.linear.Linear(
            input_size=hidden,
            n_neurons=hparams["out_n_neurons"],
            bias=False,
        ),
    )
    hparams["modules"]["output_mlp"] = mlp
    hparams["model"] = torch.nn.ModuleList([mlp])
    hparams["checkpointer"].recoverables["model"] = hparams["model"]


# RECIPE BEGINS!
if __name__ == "__main__":
    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    from kite_prepare import prepare_data  # noqa E402

    # Data preparation, to be run on only one process.
    if not hparams["skip_prep"]:
        sb.utils.distributed.run_on_main(
            prepare_data,
            kwargs={
                "data_folder": hparams["data_folder"],
                "metadata_csv": hparams["metadata_csv"],
                "save_json_train": hparams["train_annotation"],
                "save_json_valid": hparams["valid_annotation"],
                "save_json_test": hparams["test_annotation"],
                "emotions": hparams["emotions"],
            },
        )

    # Create dataset objects "train", "valid", and "test".
    datasets = dataio_prep(hparams)

    hparams["wav2vec2"] = hparams["wav2vec2"].to(device=run_opts["device"])
    # freeze the feature extractor part when unfreezing
    if not hparams["freeze_wav2vec2"] and hparams["freeze_wav2vec2_conv"]:
        hparams["wav2vec2"].model.feature_extractor._freeze_parameters()
    _build_hidden_classifier(hparams)

    # Initialize the Brain object to prepare for mask training.
    emo_id_brain = EmoIdBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    if hparams["logit_adjust"]:
        _run_logit_adjustment(emo_id_brain, datasets, hparams)
    else:
        if not run_opts["test_only"]:
            emo_id_brain.fit(
                epoch_counter=emo_id_brain.hparams.epoch_counter,
                train_set=datasets["train"],
                valid_set=datasets["valid"],
                train_loader_kwargs=hparams["dataloader_options"],
                valid_loader_kwargs=hparams["dataloader_options"],
            )

        for split in ("valid", "test"):
            emo_id_brain.report_split = split
            emo_id_brain.evaluate(
                datasets[split],
                max_key=hparams["select_metric"],
                test_loader_kwargs=hparams["dataloader_options"],
            )
