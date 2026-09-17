#!/usr/bin/env bash
# Sync scripts to eva02 and run train + eval there.
#   ./run.sh sync                      # just rsync
#   ./run.sh train Qwen/Qwen3.5-0.8B   # train (full FT); add --lora --grad-ckpt for 9B via EXTRA
#   ./run.sh gen                       # generate GSM8K candidates via llama-server (background-friendly)
#   ./run.sh eval ckpt/qwen3.5-0.8b-nli results/qwen0.8b.json
set -euo pipefail
# override for another box, e.g. azrtx:
#   HOST=azrtx REMOTE=/home/azureuser/qwen_nli PY=/home/azureuser/dense-trainer/.venv/bin/python ENVS="HF_HOME=/mnt/hf CUDA_VISIBLE_DEVICES=0" ./run.sh train ...
HOST=${HOST:-eva02}
REMOTE=${REMOTE:-/home/alexw/qwen_nli}
PY=${PY:-/home/alexw/miniconda3/envs/aisci/bin/python}
ENVS=${ENVS:-}
HERE="$(cd "$(dirname "$0")" && pwd)"

sync() { rsync -az --exclude results --exclude data "$HERE/"{train.py,eval.py,latent_mlp.py,summarize.py,flappy.py,flappy_video.py,sweep_flappy.sh,doom.py,doom_vision.py,hf_publish.py,radar.py,modeling_openjev.py,run.sh} "$HOST:$REMOTE/"; }

case "${1:-}" in
  sync) sync ;;
  train)
    sync
    MODEL=${2:-Qwen/Qwen3.5-0.8B}; NAME=$(basename "$MODEL" | tr 'A-Z' 'a-z')
    ssh "$HOST" "cd $REMOTE && mkdir -p logs && nohup env $ENVS $PY train.py --model $MODEL --out ckpt/${NAME}-nli ${EXTRA:-} > logs/train_${NAME}.log 2>&1 & echo started pid \$!"
    ;;
  gen)
    sync
    ssh "$HOST" "cd $REMOTE && mkdir -p logs data && nohup env $ENVS $PY eval.py --models x --out /dev/null --gen-only > logs/gen_gsm8k.log 2>&1 & echo started pid \$!"
    ;;
  eval)
    sync
    CKPT=${2:?ckpt}; OUT=${3:?out}
    ssh "$HOST" "cd $REMOTE && mkdir -p logs results && nohup env $ENVS $PY eval.py --models $CKPT dleemiller/ModernCE-large-nli --out $OUT ${EXTRA:-} > logs/eval_$(basename "$OUT" .json).log 2>&1 & echo started pid \$!"
    ;;
  *) echo "usage: $0 {sync|train MODEL|gen|eval CKPT OUT}"; exit 1 ;;
esac
