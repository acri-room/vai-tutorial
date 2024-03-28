#!/usr/bin/bash

PRJ_DIR=$(dirname $(dirname $(readlink -f $0)))

$PRJ_DIR/scripts/build_pytorch_nndct.sh
$PRJ_DIR/scripts/build_unilog.sh
$PRJ_DIR/scripts/build_xir.sh
