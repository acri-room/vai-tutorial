# Vitis AI Tutorial

```
python -m venv .venv
source .venv/bin/activate

pip install --find-links=/tools/pip/pytorch-rocm torch==2.2.2+rocm5.7 torchvision==0.17.2+rocm5.7 torchaudio==2.2.2+rocm5.7
pip install -r requirements.txt

mkdir data
tar xf /tools/data/cifar-10-python.tar.gz -C data

./scripts/train.sh
OR
./scripts/train.sh trainer.max_epochs=8

./scripts/build_quantizer.sh

./scripts/quant_calib.sh  ckpt_path=./logs/train/runs/<DATE_TIME>/checkpoints/epoch_*.ckpt
./scripts/quant_test.sh   ckpt_path=./logs/train/runs/<DATE_TIME>/checkpoints/epoch_*.ckpt
./scripts/quant_deploy.sh ckpt_path=./logs/train/runs/<DATE_TIME>/checkpoints/epoch_*.ckpt

./scripts/compile.sh ./quantized/ResNet9_int.xmodel ResNet9_Cifar10

pip install --find-links=/tools/pip/pytorch torch==2.2.2+cpu torchvision==0.17.2+cpu torchaudio==2.2.2+cpu
```

