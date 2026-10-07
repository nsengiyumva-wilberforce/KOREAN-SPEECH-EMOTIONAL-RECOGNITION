# KITE speech emotion recognition

Five-class emotion recognition on Korean drama speech: anger, happiness, neutral, sadness, and surprise. A fresh clone contains the training code and `extractor.py`. Checkpoints, logs, and audio are not in the repository. `results/` is created on the first training run.

Run every command from the repository root, the directory that contains `train_with_wav2vec2.py` and `extractor.py`.

The model to train is `team-lucid/hubert-large-korean` (317M parameters, hidden size 1024) with a bias-free linear head, in SpeechBrain. The checkpoint is chosen by validation macro-F1. On the finished run that checkpoint was epoch 8 (validation weighted accuracy 65.34%, unweighted accuracy 45.84%, macro-F1 0.460). Weighted accuracy is the overall share of correct clips. Unweighted accuracy is the average of the five per-class recalls. Report both.

Test scores for that checkpoint:

| | Weighted accuracy | Unweighted accuracy |
|---|---|---|
| Plain argmax | 65.00% | 43.31% |
| Logit adjustment, tau 0.75 | 54.06% | 48.39% |

Tau 0.75 is chosen on the validation set (validation unweighted accuracy 47.12%, weighted accuracy 51.18%). The tau-0.75 test score is the number to report. The same seed, split, and recipe should land near these figures. Keep the tau that won on validation. The test table is only for the final report.

Follow the sections in order: install, extract, then train. Section 3 is the single-GPU HuBERT large run. Section 4 is HuBERT xlarge on 3 GPUs. Logit adjustment comes after training.

## 1. Install

Python 3.10 and an NVIDIA GPU. This recipe was trained on an 8 GB GPU. HuBERT large fits batch size 2 in fp32. Batch size 4 does not. Run one training job at a time. Clips longer than 16 seconds are cropped: a random crop in training, the center crop in validation and test.

Install the CUDA build of PyTorch first, then the recipe requirements. SpeechBrain 1.1.1 is pinned in `requirements.txt`. Those requirements also cover segment extraction.

```bash
pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## 2. Extract the segments

`extractor.py` cuts one dialogue turn out of each KETI video and writes a mono 16 kHz float32 wav. A high-pass filter at 80 Hz is applied, near-silent cuts are dropped, and each kept clip is peak-normalized. Turns shorter than 0.3 seconds are dropped. The original videos are not modified.

The `--root_dir` folder contains the part directories (`멀티모달_Part_01` through `멀티모달_Part_11`). Inside each part, every clip has its own folder with `{video_id}.mp4` and `{video_id}_interpolation.json`. Pass `--skip_feature_extraction` so the script stops once the wavs and the metadata CSV exist.

```bash
python extractor.py \
  --root_dir /path/to/멀티모달_분야 \
  --output_audio_dir /path/to/KITE \
  --skip_feature_extraction
```

When it finishes, `/path/to/KITE` contains the wavs and `1_metadata_ser.csv`. The CSV columns used by training are `audio_path`, `emotion_label`, `duration`, and `split`. The split is stratified 80/10/10 train/dev/test with seed 42. Leave the CSV and the wavs unchanged after this step. Training loads each wav by file name from `--output_audio_dir`, so the file name in `audio_path` has to match a wav in that directory.

The CSV still contains emotions outside the five training classes. Training keeps only anger, happiness, neutral, sadness, and surprise, and writes the dev split as the validation manifest. On the finished extraction that kept set is 8795 train, 1102 validation, and 1097 test. The preparation log prints these three counts. If they differ, the videos or the CSV are not the same release.

The path inside `hparams/train_with_wav2vec2.yaml` is a local default. Pass your own directory on the training command.

## 3. Train

The yaml defaults still name an XLS-R encoder. This command switches to Korean HuBERT and writes the run under `results/kite_hubert_large_korean/1993`. The first launch downloads `team-lucid/hubert-large-korean` into `results/kite_hubert_large_korean/1993/save/wav2vec2_checkpoint`. Use the same `/path/to/KITE` directory from step 2.

```bash
python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml \
  --data_folder /path/to/KITE \
  --metadata_csv /path/to/KITE/1_metadata_ser.csv \
  --output_folder results/kite_hubert_large_korean/1993 \
  --wav2vec2_hub team-lucid/hubert-large-korean \
  --number_of_epochs 30
```

Training runs 30 epochs. The head learning rate is 1e-4 and the encoder learning rate is 1e-5. The convolutional feature extractor stays frozen. Class weights are inverse frequency, total / (5 × class count). On this dataset those weights are anger 3.002, happiness 1.197, neutral 0.305, sadness 5.298, surprise 2.731. The learning-rate schedule watches `1 - macro_f1`. After each epoch the recipe keeps the checkpoint with the best validation macro-F1 and deletes the rest, so let all 30 epochs finish. On the finished run the kept checkpoint was epoch 8.

At the end the script reloads that checkpoint and writes `confusion_valid.txt` and `confusion_test.txt` in the output folder. Rows are the true emotion. Columns are the prediction. Read both weighted and unweighted accuracy from `confusion_test.txt`. That plain test matrix is the first row of the table at the top.

## 4. Train HuBERT xlarge on 3 GPUs

`hparams/train_hubert_xlarge.yaml` loads `team-lucid/hubert-xlarge-korean` (1B parameters, 48 transformer layers, hidden size 1280). The linear head is sized to 1280. Batch size 4 is the batch on each GPU. Launch three processes so each update sees 12 clips. Learning rates stay the same as the large run: 1e-4 for the head and 1e-5 for the encoder. The run is written to `results/kite_hubert_xlarge_korean/1993` and does not touch the large checkpoint.

```bash
torchrun --nproc_per_node=3 train_with_wav2vec2.py hparams/train_hubert_xlarge.yaml \
  --data_folder /path/to/KITE \
  --metadata_csv /path/to/KITE/1_metadata_ser.csv
```

Validation and test still score the full split, and one process writes the log and the confusion files. After training, score and sweep tau with one process. Do not use `torchrun` for these two commands.

```bash
python train_with_wav2vec2.py hparams/train_hubert_xlarge.yaml \
  --data_folder /path/to/KITE \
  --metadata_csv /path/to/KITE/1_metadata_ser.csv \
  --test_only

python train_with_wav2vec2.py hparams/train_hubert_xlarge.yaml \
  --data_folder /path/to/KITE \
  --metadata_csv /path/to/KITE/1_metadata_ser.csv \
  --logit_adjust True
```

## 5. Logit adjustment

Run this after training has finished, in the same output folder. It does not train. It subtracts `tau * log(train prior)` from the log-probabilities, sweeps the tau list in the hyperparameters on the validation set, and scores the test set once at the tau with the highest validation unweighted accuracy.

```bash
python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml \
  --data_folder /path/to/KITE \
  --metadata_csv /path/to/KITE/1_metadata_ser.csv \
  --output_folder results/kite_hubert_large_korean/1993 \
  --wav2vec2_hub team-lucid/hubert-large-korean \
  --logit_adjust True
```

The report is `results/kite_hubert_large_korean/1993/confusion_test_logit.txt`. On the finished run the selected tau was 0.75, with test weighted accuracy 54.06% and unweighted accuracy 48.39%.

## 6. Score the checkpoint again

`--test_only` is a flag. Do not write `--test_only True`. This loads the saved macro-F1 checkpoint and rewrites the plain validation and test matrices. It does not apply logit adjustment.

```bash
python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml \
  --data_folder /path/to/KITE \
  --metadata_csv /path/to/KITE/1_metadata_ser.csv \
  --output_folder results/kite_hubert_large_korean/1993 \
  --wav2vec2_hub team-lucid/hubert-large-korean \
  --test_only
```

## 7. Ablations

These are the finished comparisons on the same split. Each row is the checkpoint with the best validation macro-F1. Validation F1 is macro-F1 shown as a percentage. Runs that never produced a test score are omitted. The last row is the score to report.

| Run | Epoch | Val WA | Val UA | Val F1 | Test WA | Test UA | Note |
|---|---:|---:|---:|---:|---:|---:|---|
| XLS-R, balanced sampling | 3 | 29.30 | 46.10 | 28.20 | 29.54 | 45.79 | test |
| Korean XLS-R, balanced sampling | 5 | 31.60 | 36.40 | 26.20 | 33.73 | 37.53 | test |
| XLS-R, freeze 18 layers | 15 | 58.20 | 42.10 | 39.60 | 58.80 | 43.01 | test |
| XLS-R, class weights | 15 | 62.90 | 41.60 | 42.20 | 61.99 | 39.87 | test |
| XLS-R, class weights + tau 0.75 | | | | | 57.34 | 43.74 | logit |
| XLS-R fusion override | | 51.54 | 42.73 | | 51.32 | 46.31 | test |
| XLS-R, latent mixup | 9 | 63.10 | 43.70 | 42.90 | 60.89 | 40.02 | test |
| XLS-R, latent mixup + tau 0.25 | | | | | 58.98 | 42.36 | logit |
| HuBERT, freeze first 12 | 11 | 65.00 | 42.80 | 44.10 | 65.63 | 40.84 | test |
| HuBERT, SAM concat | 4 | 61.80 | 47.00 | 45.70 | 60.16 | 45.33 | test |
| HuBERT, SAM concat + tau 1.00 | | | | | 29.44 | 46.65 | logit |
| HuBERT large Korean | 8 | 65.30 | 45.80 | 46.00 | 65.00 | 43.31 | test |
| HuBERT large Korean + tau 0.75 | | | | | 54.06 | 48.39 | logit, better |
