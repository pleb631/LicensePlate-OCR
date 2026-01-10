#!/bin/bash
ENV_NAME="lprnet"

if conda env list|grep -qE "^[^#]*\b$ENV_NAME\b";then
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME
else
conda create -n $ENV_NAME python==3.11.0 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME
pip install --upgrade pip
python -m pip install -r requirements.txt
fi
echo "$CONDA_DEFAULT_ENV"

export PYTHONPATH=.
# lprnet/lib/python3.11/site-packages/mxnet/numpy/utils.py 中第37行要改为bool = onp.bool_
python tools/train.py fit -c config/lpnet_c2f.yaml
# python tools/export/convert_onnx.py --weights lightning_logs/C2fNet-lp0603/checkpoints/last.ckpt
