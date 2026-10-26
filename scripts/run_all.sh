#!/usr/bin/env bash
# Full pipeline in the order I ran it. Each training run takes ~50-75 min on an RTX 5060.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
Q25=Qwen/Qwen2.5-1.5B-Instruct
Q3=Qwen/Qwen3-1.7B
PHI=microsoft/Phi-4-mini-instruct

# Pilot (English, Qwen2.5): penalties from step 1 vs a 200-step warm-up
$PY -m stego.train --model $Q25 --steps 600 --out runs/en_baseline
$PY -m stego.train --model $Q25 --steps 600 --w-comm 2 --penalty-warmup 200 --out runs/en_warmup
$PY -m stego.evaluate --model $Q25 --n 128 --out runs/eval_untrained.jsonl
$PY -m stego.evaluate --model $Q25 --adapters runs/en_baseline/final --n 128 --out runs/eval_trained.jsonl
$PY -m stego.evaluate --model $Q25 --adapters runs/en_warmup/final --n 128 --out runs/eval_warmup.jsonl

# Qwen3 runs, English and Arabic. Round 1: w_comm 2, 200-step warm-up. Round 2: w_comm 3, 300-step warm-up, 2 seeds.
train_eval() {  # name lang seed w_comm warmup
  $PY -m stego.train --model $Q3 --lang $2 --seed $3 --steps 600 --w-comm $4 --penalty-warmup $5 --out runs/$1
  $PY -m stego.evaluate --model $Q3 --lang $2 --adapters runs/$1/final --n 128 --out runs/eval_$1.jsonl
}
for L in en ar; do
  train_eval q3_${L}_warmup $L 0 2 200
  $PY -m stego.evaluate --model $Q3 --lang $L --n 128 --out runs/eval_q3_${L}_untrained.jsonl
done
for seed in 1 2; do for L in en ar; do train_eval r2_${L}_s$seed $L $seed 3 300; done; done

# How each code developed: rerun every checkpoint on the same prompts
for R in q3_en_warmup q3_ar_warmup r2_en_s1 r2_ar_s1 r2_en_s2 r2_ar_s2; do
  L=$($PY -c "import json; print(json.load(open('runs/$R/config.json'))['lang'])")
  $PY -m stego.snapshots --model $Q3 --lang $L --run runs/$R --out runs/snapshots_$R.json
done
$PY -m stego.snapshots --model $Q25 --lang en --run runs/en_warmup --out runs/snapshots_en_warmup.json

# Defenses: pilot (data/en, runs/defense_en) and round-1 Arabic
$PY -m stego.collect --model $Q25 --adapters runs/en_warmup/final --secret-test runs/eval_warmup.jsonl
$PY -m stego.collect --model $Q25 --adapters runs/en_warmup/final --secret-test runs/eval_warmup.jsonl --only-sender-clean
for c in "llm-monitor --name qwen_monitor" "llm-monitor --name phi_monitor --model $PHI --4bit" \
         "classifier" "classifier --negatives sendclean" "paraphrase" "probe"; do
  $PY -m stego.defend $c --lang en
done
R=q3_ar_warmup
ar=(--lang ar --model $Q3 --data data/$R --out runs/defense_$R --adapters runs/$R/final)
$PY -m stego.collect --model $Q3 --lang ar --adapters runs/$R/final --secret-test runs/eval_$R.jsonl --out data/$R
$PY -m stego.collect --model $Q3 --lang ar --adapters runs/$R/final --secret-test runs/eval_$R.jsonl --out data/$R --only-sender-clean
$PY -m stego.defend llm-monitor "${ar[@]}" --name samebase_monitor
$PY -m stego.defend llm-monitor "${ar[@]}" --name phi_monitor --model $PHI --4bit
$PY -m stego.defend classifier "${ar[@]}"
$PY -m stego.defend classifier "${ar[@]}" --negatives sendclean
$PY -m stego.defend paraphrase "${ar[@]}"
$PY -m stego.defend probe "${ar[@]}"
$PY -m stego.defend translate "${ar[@]}"
$PY -m stego.defend llm-monitor "${ar[@]}" --translated --name samebase_monitor_translated
$PY -m stego.defend llm-monitor "${ar[@]}" --translated --name phi_monitor_translated --model $PHI --4bit

# Do trained detectors transfer to other seeds and the other language?
$PY -m stego.generalize

# Web page
$PY -m stego.export_web
$PY scripts/build_web.py
