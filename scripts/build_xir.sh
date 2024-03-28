#!/usr/bin/bash

PRJ_DIR=$(dirname $(dirname $(readlink -f $0)))

VAI_DIR=/tools/repo/Xilinx/Vitis-AI
SRC_DIR=$VAI_DIR/src/vai_runtime/xir
BUILD_DIR=$PRJ_DIR/.build/xir
INSTALL_DIR=$PRJ_DIR/.local

rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR

git config --global --add safe.directory $VAI_DIR

cmake $SRC_DIR \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
  -DBUILD_PYTHON=ON \
  -DINSTALL_HOME=ON \
  -Dunilog_DIR=$INSTALL_DIR \
  -DCMAKE_CXX_FLAGS="-fPIC"
make -j
make install

