#!/usr/bin/bash

PRJ_DIR=$(dirname $(dirname $(readlink -f $0)))

RT_ENGINE_DIR=/tools/repo/Xilinx/rt-engine
SRC_DIR=$PRJ_DIR/.repo/rt-engine
BUILD_DIR=$PRJ_DIR/.build/rt-engine
INSTALL_DIR=$PRJ_DIR/.local

mkdir -p $PRJ_DIR/.repo
if [[ ! -e $PRJ_DIR/.repo/rt-engine ]] ; then
  cd $PRJ_DIR/.repo
  if [[ -e $RT_ENGINE_DIR ]] ; then
    cp -a $RT_ENGINE_DIR .
  else
    git clone https://github.com/Xilinx/rt-engine.git
  fi
fi

cd $PRJ_DIR/.repo/rt-engine
git checkout v3.5

rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR

cmake $SRC_DIR \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
  -DBUILD_TESTS=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DXRM_DIR=/opt/xilinx/xrm/share/cmake \
  -DCMAKE_CXX_FLAGS="-fPIC"
make -j
make install

