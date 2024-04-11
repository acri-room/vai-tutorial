#!/usr/bin/bash

PRJ_DIR=$(dirname $(dirname $(readlink -f $0)))

$PRJ_DIR/scripts/build_unilog.sh
$PRJ_DIR/scripts/build_xir.sh
$PRJ_DIR/scripts/build_target_factory.sh
$PRJ_DIR/scripts/build_vart.sh
$PRJ_DIR/scripts/build_rt-engine.sh
