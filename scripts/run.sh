#!/usr/bin/bash

PRJ_DIR=$(dirname $(dirname $(readlink -f $0)))

if [[ ! -e $PRJ_DIR/.local/lib/python ]] ; then
  $PRJ_DIR/scripts/build_runtime.sh
fi

if [[ ! -e $PRJ_DIR/.local/lib/librt-engine.so ]] ; then
  $PRJ_DIR/scripts/build_runtime.sh
fi

if [[ -z $XCLBIN_PATH ]] ; then
  source $PRJ_DIR/docker/setup_vck5000.sh DPUCVDX8H_8pe_normal
fi

PYTHONPATH=$PRJ_DIR/.local/lib/python \
  LD_LIBRARY_PATH=$PRJ_DIR/.local/lib:$LD_LIBRARY_PATH \
  HYDRA_FULL_ERROR=1 \
  python ./src/run.py \
  "$@"

