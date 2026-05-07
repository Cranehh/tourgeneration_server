#!/bin/bash
# 顺序训练所有 6 个 baseline，把日志各自存档
set -e
cd "$(dirname "$0")"

PY=/home/cranehh/anaconda3/envs/DiT/bin/python
LOG_DIR=logs
mkdir -p "$LOG_DIR"

EPOCHS=${EPOCHS:-100}
BS=${BS:-256}

declare -a SCRIPTS=(train_gan.py train_wgan.py train_vae.py train_cgan.py train_cwgan.py train_cvae.py)

for s in "${SCRIPTS[@]}"; do
  name=$(basename "$s" .py | sed 's/train_//')
  echo "================================================="
  echo "[$(date +%H:%M:%S)] Training $name (epochs=$EPOCHS bs=$BS)"
  echo "================================================="
  $PY "$s" --epochs "$EPOCHS" --batch_size "$BS" 2>&1 | grep -E "device|epoch [0-9]+ done|saved|Error|cond_dim|train flat|target_dim" \
    > "$LOG_DIR/${name}.log" || { echo "[$name] FAILED"; exit 1; }
  echo "[$(date +%H:%M:%S)] Done $name"
  tail -3 "$LOG_DIR/${name}.log"
  echo
done
echo "All baselines trained. Checkpoints in checkpoints/, logs in $LOG_DIR/"
