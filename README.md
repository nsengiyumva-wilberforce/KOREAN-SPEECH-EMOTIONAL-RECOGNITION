# KITE speech emotion recognition

Five-class emotion recognition on the KITE Korean speech segments: anger, happiness, neutral, sadness, and surprise. The model is Wav2Vec2 XLS-R 300M with a linear head, trained in SpeechBrain.

Run every command from this directory (`kite/`). Result paths in the hyperparameters are relative to it.

## Setup

Python 3.10 and a CUDA GPU. XLS-R 300M fits an 8 GB GPU at batch size 2. Longer clips are cropped to 16 seconds.

```bash
pip install -r requirements.txt
```

That installs SpeechBrain 1.1.1. The recipe does not use a local SpeechBrain checkout.

Audio and labels are read from:

`/media/computergeek/SER-datasets/korean-audio-sentiment-analysis/data/KITE`

The metadata file is `1_metadata_ser.csv` in that folder. The official train / dev / test split is kept. The dev split is written as the validation manifest. Point the recipe at another copy with:

```bash
python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml \
  --data_folder /path/to/KITE \
  --metadata_csv /path/to/KITE/1_metadata_ser.csv
```

## Finished model

The class-weighted XLS-R run is already in `results/kite_xlsr/1993/`. The best checkpoint is epoch 15, selected by macro-F1. On the test set that checkpoint scores weighted accuracy 61.99% and unweighted accuracy 39.87%.

Logit adjustment (tau 0.75) on the same checkpoint scores test weighted accuracy 57.34% and unweighted accuracy 43.74%. The matrix is `results/kite_xlsr/1993/confusion_test_logit.txt`.

Training again into `results/kite_xlsr/1993` replaces that checkpoint. Pass a new `--output_folder` if you want to keep it.

## Train

```bash
python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml
```

Defaults are in `hparams/train_with_wav2vec2.yaml`: 30 epochs, batch size 2, class-weighted loss, checkpoint by macro-F1. The first run downloads `facebook/wav2vec2-xls-r-300m` into `results/kite_xlsr/1993/save/wav2vec2_checkpoint`.

Override any hyperparameter from the command line as `--key value`. Example, a shorter run in a new folder that reuses the downloaded encoder:

```bash
python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml \
  --output_folder results/kite_xlsr_new/1993 \
  --wav2vec2_folder results/kite_xlsr/1993/save/wav2vec2_checkpoint \
  --number_of_epochs 20
```

Other switches in the same file: `--balanced_sampling True`, `--select_metric ua`, `--freeze_transformer_layers 18`, `--classifier_hidden 256`.

Inverse-square-root class weights use the square root of the inverse-frequency weight. Add `--augment_rare True` to also vary anger, sadness, and surprise during training (speed 90 or 110, gain 0.8–1.2, and sometimes a short mute). Neutral, happiness, validation, and test are unchanged. Run that only after the other XLS-R job has finished; both need the whole 8 GB GPU.

```bash
python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml \
  --output_folder results/kite_xlsr_invsqrt_aug/1993 \
  --wav2vec2_folder results/kite_xlsr/1993/save/wav2vec2_checkpoint \
  --class_weighting inverse_sqrt \
  --augment_rare True \
  --number_of_epochs 30
```

## Evaluate a saved checkpoint

`--test_only` is a flag. Do not write `--test_only True`.

```bash
python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml --test_only
```

This loads the best checkpoint in `results/kite_xlsr/1993` and writes `confusion_valid.txt` and `confusion_test.txt` there. Each file contains weighted accuracy, unweighted accuracy, and the confusion matrix. Rows are the true emotion and columns are the prediction.

## Logit adjustment

This does not train. It sweeps the tau values in the hyperparameters on the validation set, then scores the test set at the tau with the highest validation unweighted accuracy.

```bash
python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml --logit_adjust True
```

The report is `results/kite_xlsr/1993/confusion_test_logit.txt`. The selected tau is 0.75.

## Live web app

The app loads the class-weighted checkpoint and applies logit adjustment at tau 0.75. It expects that checkpoint and the wav2vec2 weights under `results/kite_xlsr/1993/`.

```bash
python webapp/app.py
```

The server listens on every network interface and uses HTTPS, so a phone or another computer on the same Wi-Fi can open it. The startup log prints that address, for example `https://192.168.0.10:7861`. Accept the certificate warning, then allow the microphone. Browsers block the microphone on plain HTTP except on this computer.

On this computer, open `https://127.0.0.1:7861`. Options: `--port`, `--cpu`, and `--http` (this computer only, no certificate).

## Other saved runs

`results/comparison_table.txt` lists the finished experiments. Each run folder has its own `confusion_test.txt`. `run_comparisons.sh` is the script that trained them one after another on one GPU. `combine_margin.py` scores the two-model fusion and needs both `results/kite_xlsr/1993` and `results/kite_xlsr_balanced/1993`.
