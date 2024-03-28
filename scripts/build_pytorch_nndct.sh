#!/usr/bin/bash

PRJ_DIR=$(dirname $(dirname $(readlink -f $0)))

VAI_DIR=/tools/repo/Xilinx/Vitis-AI
SRC_DIR=$VAI_DIR/src/vai_quantizer/vai_q_pytorch
BUILD_DIR=$PRJ_DIR/.build/pytorch_nndct
INSTALL_DIR=$PRJ_DIR/.local

rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR

cp -rp $SRC_DIR .
cd vai_q_pytorch/pytorch_binding

sed -i 's/-std=c++14/-std=c++17/' setup.py

pip install wheel onnx
pip install -r ../requirements.txt

ROCM_HOME=/opt/rocm python setup.py bdist_wheel -d ./

pip install ./pytorch_nndct*.whl
