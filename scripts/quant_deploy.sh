#!/usr/bin/bash

PYTHONPATH=/scratch/ando/local/lib/python \
  python ./src/eval.py \
  model.quant_mode=deploy \
  data.batch_size=1 \
  +trainer.limit_test_batches=1 \
  "$@"

