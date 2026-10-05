#!/bin/bash
# Sequential comparison on one GPU. Each new training run uses 20 epochs.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
LOG=results/comparison_runs.log
TABLE=results/comparison_table.txt
XLSR_WEIGHTS=results/kite_xlsr/1993/save/wav2vec2_checkpoint
: > "$LOG"

record() {
  local name="$1"
  local file="$2"
  local wa ua
  wa=$(grep "WA (weighted accuracy):" "$file" | tail -n 1 | sed 's/.*: //')
  ua=$(grep "UA (unweighted accuracy):" "$file" | tail -n 1 | sed 's/.*: //')
  printf "%s\t%s\t%s\n" "$name" "$wa" "$ua" >> "$TABLE"
  echo "recorded $name  WA $wa  UA $ua"
}

printf "method\ttest_WA\ttest_UA\n" > "$TABLE"
record "XLS-R + class weights" results/kite_xlsr/1993/confusion_test.txt

echo "===== logit adjustment =====" | tee -a "$LOG"
python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml --logit_adjust True >> "$LOG" 2>&1
record "XLS-R + logit adjustment" results/kite_xlsr/1993/confusion_test_logit.txt

run_train() {
  local name="$1"
  shift
  echo "===== $name =====" | tee -a "$LOG"
  python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml "$@" >> "$LOG" 2>&1
}

run_train "XLS-R + balanced batches" \
  --output_folder results/kite_xlsr_balanced/1993 \
  --wav2vec2_folder "$XLSR_WEIGHTS" \
  --balanced_sampling True \
  --select_metric ua \
  --number_of_epochs 20
record "XLS-R + balanced batches" results/kite_xlsr_balanced/1993/confusion_test.txt

run_train "Korean wav2vec + balanced batches" \
  --output_folder results/kite_korean_balanced/1993 \
  --wav2vec2_hub kresnik/wav2vec2-large-xlsr-korean \
  --balanced_sampling True \
  --select_metric ua \
  --number_of_epochs 20
record "Korean wav2vec + balanced batches" results/kite_korean_balanced/1993/confusion_test.txt

run_train "XLS-R frozen lower layers + 2-layer head" \
  --output_folder results/kite_xlsr_frozen/1993 \
  --wav2vec2_folder "$XLSR_WEIGHTS" \
  --freeze_transformer_layers 18 \
  --classifier_hidden 256 \
  --select_metric ua \
  --number_of_epochs 20
record "XLS-R frozen lower layers + 2-layer head" results/kite_xlsr_frozen/1993/confusion_test.txt

run_train "Hierarchical stage 1, neutral versus rest" \
  --output_folder results/kite_binary/1993 \
  --wav2vec2_folder "$XLSR_WEIGHTS" \
  --label_mode binary \
  --out_n_neurons 2 \
  --balanced_sampling True \
  --save_predictions True \
  --select_metric ua \
  --number_of_epochs 20
run_train "Hierarchical stage 2, four emotions" \
  --output_folder results/kite_emotional/1993 \
  --wav2vec2_folder "$XLSR_WEIGHTS" \
  --label_mode emotional \
  --out_n_neurons 4 \
  --balanced_sampling True \
  --save_predictions True \
  --select_metric ua \
  --number_of_epochs 20
python combine_hierarchical.py >> "$LOG" 2>&1
record "Hierarchical neutral then emotion" results/kite_hierarchical/confusion_test.txt

echo "===== comparison table =====" | tee -a "$LOG"
column -t -s $'\t' "$TABLE" | tee -a "$LOG"
