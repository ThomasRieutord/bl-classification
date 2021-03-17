#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PERFORM UNSUPERVISED BOUNDARY LAYER CLASSIFICATION ON ALL AVAILABLE DAYS

 +-----------------------------------------+
 |  Date of creation: 14 Apr. 2020         |
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
from blusc import multidays


# Parametrisation
# -----------------

### Input
day = dt.date(2015, 2, 19)

### Output
graphics.storeImages = False
graphics.fmtImages = ".svg"
graphics.figureDir = "../tmpout/"

### Algo
algo = "hierarchical-average.euclidean"
target_nb_clusters = "auto"

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


# Execution
# -----------

multidays.unsupervised_path(
    CEI_file, MWR_file, algo=algo, target_nb_clusters=target_nb_clusters
)
