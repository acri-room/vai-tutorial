#!/usr/bin/bash

PRJ_DIR=$(dirname $(dirname $(readlink -f $0)))

VAI_DIR=/tools/repo/Xilinx/Vitis-AI
SRC_DIR=$VAI_DIR/src/vai_runtime/unilog
BUILD_DIR=$PRJ_DIR/.build/unilog
INSTALL_DIR=$PRJ_DIR/.local

rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR

cmake $SRC_DIR -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DCMAKE_CXX_FLAGS="-fPIC"
make -j
make install
