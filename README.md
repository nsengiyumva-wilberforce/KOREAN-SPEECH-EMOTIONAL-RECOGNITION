# KITE speech emotion recognition

Five-class emotion recognition on the KITE Korean drama speech segments: anger, happiness, neutral, sadness, and surprise. A fresh clone contains the training code only. Checkpoints, logs, and audio are not in the repository. `results/` is created on the first run.

The encoder is `team-lucid/hubert-large-korean` (317M, hidden size 1024) with a bias-free linear head, trained in SpeechBrain. Run every command from the repository root, the directory that contains `train_with_wav2vec2.py`.

On the finished run the best checkpoint was epoch 8, chosen by validation macro-F1 (weighted accuracy 65.34%, unweighted accuracy 45.84%, macro-F1 0.460). Test scores for that checkpoint:

| | Weighted accuracy | Unweighted accuracy |
|---|---|---|
| Plain argmax | 65.00% | 43.31% |
| Logit adjustment, tau 0.75 | 54.06% | 48.39% |

Tau 0.75 is chosen on the validation set (validation unweighted accuracy 47.12%, weighted 51.18%). The tau-0.75 test score is the number to report. The same seed, split, and recipe should land near these figures. Do not pick a different tau from the test table.

## 1. Install

Python 3.10 and an NVIDIA GPU. This recipe was trained on an 8 GB GPU. HuBERT large fits batch size 2 in fp32. Batch size 4 does not. Run one training job at a time. Clips longer than 16 seconds are cropped: a random crop in training, the center crop in validation and test.

Install the CUDA build of PyTorch first, then the recipe requirements. SpeechBrain 1.1.1 is pinned in `requirements.txt`.

```bash
pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## 2. Data

Put the KITE wav files and `1_metadata_ser.csv` in one directory. The CSV needs `audio_path`, `emotion_label`, `duration`, and `split`. Splits are `train`, `dev`, and `test`. The recipe writes the dev split as the validation manifest and keeps only anger, happiness, neutral, sadness, and surprise. Each wav is loaded by its file name from that directory, so the file name in `audio_path` has to match a wav sitting next to the CSV. Leave the CSV and the official split unchanged.

The path inside `hparams/train_with_wav2vec2.yaml` is a local default. A new checkout has to pass its own directory on the command line.

## 3. Train

The yaml defaults still name an XLS-R encoder. This command switches to Korean HuBERT and writes a new run under `results/kite_hubert_large_korean/1993`. The first launch downloads `team-lucid/hubert-large-korean` into `results/kite_hubert_large_korean/1993/save/wav2vec2_checkpoint`.

Training runs 30 epochs. The head learning rate is 1e-4 and the encoder learning rate is 1e-5. The convolutional feature extractor stays frozen. Class weights are inverse frequency, total / (5 × class count). On this dataset those weights are anger 3.002, happiness 1.197, neutral 0.305, sadness 5.298, surprise 2.731. The learning-rate schedule watches `1 - macro_f1`. After each epoch the recipe keeps the checkpoint with the best validation macro-F1 and deletes the rest, so let all 30 epochs finish. On the finished run the kept checkpoint was epoch 8.

Replace the two paths with the directory from step 2.

```bash
python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml \
  --data_folder /path/to/KITE \
  --metadata_csv /path/to/KITE/1_metadata_ser.csv \
  --output_folder results/kite_hubert_large_korean/1993 \
  --wav2vec2_hub team-lucid/hubert-large-korean \
  --number_of_epochs 30
```

When epoch 30 ends, the script reloads the best macro-F1 checkpoint and writes `confusion_valid.txt` and `confusion_test.txt` in the output folder. Rows are the true emotion. Columns are the prediction. Report both weighted and unweighted accuracy. The plain test matrix is the first of the two scores in the table above.

## 4. Logit adjustment

Run this after training, in the same output folder. It does not train. It subtracts `tau * log(train prior)` from the log-probabilities, sweeps the tau list in the hyperparameters on the validation set, and scores the test set once at the tau with the highest validation unweighted accuracy.

```bash
python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml \
  --data_folder /path/to/KITE \
  --metadata_csv /path/to/KITE/1_metadata_ser.csv \
  --output_folder results/kite_hubert_large_korean/1993 \
  --wav2vec2_hub team-lucid/hubert-large-korean \
  --logit_adjust True
```

The report is `results/kite_hubert_large_korean/1993/confusion_test_logit.txt`. On the finished run the selected tau was 0.75, with test weighted accuracy 54.06% and unweighted accuracy 48.39%.

## 5. Score the checkpoint again

`--test_only` is a flag. Do not write `--test_only True`. This loads the saved macro-F1 checkpoint and rewrites the plain validation and test matrices. It does not apply logit adjustment.

```bash
python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml \
  --data_folder /path/to/KITE \
  --metadata_csv /path/to/KITE/1_metadata_ser.csv \
  --output_folder results/kite_hubert_large_korean/1993 \
  --wav2vec2_hub team-lucid/hubert-large-korean \
  --test_only
```
