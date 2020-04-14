#!/usr/bin/bash

set -x

python prepdataset.py
python unsupervised.py
python blidentification.py
python supervisedfit.py
python supervisedpred.py
python multidays.py

set +x
echo "-*- All good! -*-"
