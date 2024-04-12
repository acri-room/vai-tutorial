#!/usr/bin/bash

if [[ $# -ne 2 ]] ; then
  echo usage: $0 XMODEL NET_NAME
  exit
fi

./docker/run.sh \
  bash -c \
  "source /opt/vitis_ai/conda/etc/profile.d/conda.sh ; \
  conda activate vitis-ai-pytorch; \
  vai_c_xir \
  --xmodel $1 \
  --arch /opt/vitis_ai/compiler/arch/DPUCVDX8H/VCK50008PE/arch.json \
  --output_dir ./compiled \
  --net_name $2"

