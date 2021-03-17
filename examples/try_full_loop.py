# -*- coding: utf-8 -*-
"""
PERFORM UNSUPERVISED AND SUPERVISED BOUNDARY LAYER CLASSIFICATION STRAIGHTFULLY

 +-----------------------------------------+
 |  Date of creation: 17 Mar. 2021         |
 +-----------------------------------------+
 |  Meteo-France                           |
 |  CNRM/GMEI/LISA                         |
 +-----------------------------------------+

"""

import os
import numpy as np
import datetime as dt

# Local packages
from blusc import graphics
from blusc import prepdataset
from blusc import unsupervised
from blusc import supervisedfit
from blusc import supervisedpred

# Parametrisation
# -----------------

### Input
day = dt.date(2015, 2, 19)

### Output
graphics.storeImages = False
graphics.fmtImages = ".svg"
graphics.figureDir = "../tmpout/"

### Algo
unsup_algo = "hierarchical-average.cityblock"
target_nb_clusters = 4
sup_algo = "KNeighborsClassifier"


### Paths
CEI_dir = "../working-directories/0-original-data/CEILOMETER/"
MWR_dir = "../working-directories/0-original-data/MWR/"
CEI_file = os.path.join(
    CEI_dir, "PASSY_PASSY_CNRM_CEILOMETER_CT25K_" + day.strftime("%Y_%m%d") + "_V01.nc"
)
MWR_file = os.path.join(
    MWR_dir,
    "PASSY2015_SALLANCHES_CNRM_MWR_HATPRO_" + day.strftime("%Y_%m%d") + "_V01.nc",
)

datasetpath = (
    "../working-directories/1-unlabelled-datasets/DATASET_"
    + day.strftime("%Y_%m%d")
    + ".PASSY2015_BT-T_linear_dz40_dt30_zmax2000.nc"
)
idflabelspath = (
    "../working-directories/3-identified-labels/IDFLABELS_"
    + day.strftime("%Y_%m%d")
    + ".PASSY2015_BT-T_linear_dz40_dt30_zmax2000.nc"
)
classifierpath = (
    "../working-directories/4-pre-trained-classifiers/"
    + sup_algo
    + ".PASSY2015_BT-T_linear_dz40_dt30_zmax2000.pkl"
)

# Execution
# -----------

prepdataset.prepdataset(CEI_file, MWR_file, plot_on=True)

unsupervised.ublc(
    datasetpath,
    algo=unsup_algo,
    target_nb_clusters=target_nb_clusters,
    saveNetcdf=True,
    plot_on=True,
)

os.system("python ../blusc/blidentification.py")

supervisedfit.train_sblc(idflabelspath, algo=sup_algo, savePickle=True)

supervisedpred.predict_sblc(datasetpath, classifierpath, plot_on=True)
