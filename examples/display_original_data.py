#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DISPLAY ORIGINAL DATA FOR BOUNDARY LAYER CLASSIFICATION

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
from blusc import multidays

# Parametrisation
# -----------------

### Input
day = dt.date(2015, 2, 19)

### Output
graphics.storeImages = False
graphics.fmtImages = ".svg"
graphics.figureDir = "../tmpout/"

### Paths
CEI_dir = "../working-directories/0-original-data/CEILOMETER/"
MWR_dir = "../working-directories/0-original-data/MWR/"
CEI_file = (
    CEI_dir + "PASSY_PASSY_CNRM_CEILOMETER_CT25K_" + day.strftime("%Y_%m%d") + "_V01.nc"
)
MWR_file = (
    MWR_dir
    + "PASSY2015_SALLANCHES_CNRM_MWR_HATPRO_"
    + day.strftime("%Y_%m%d")
    + "_V01.nc"
)


# Execution
# -----------

graphics.quicklook(CEI_file)
graphics.quicklook(MWR_file)
