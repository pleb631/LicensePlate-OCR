#!/bin/bash
python prepare-dataset/CBLPRD.py
python prepare-dataset/CCPD.py
python prepare-dataset/CRPD.py
CUDA_VISIBLE_DEVICES=0 venv/bin/python tools/train.py fit -c config/lpnet_c2fyaml
