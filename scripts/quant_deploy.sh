#!/usr/bin/bash

PRJ_DIR=$(dirname $(dirname $(readlink -f $0)))

if [[ ! -e $PRJ_DIR/.local/lib/python ]] ; then
  echo "ERROR: Build xir"
  exit 1
fi

PYTHONPATH=$PRJ_DIR/.local/lib/python \
  python ./src/eval.py \
  model.quant_mode=deploy \
  data.batch_size=1 \
  +trainer.limit_test_batches=1 \
  "$@"

