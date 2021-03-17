#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
PERFORM SUPERVISED BOUNDARY LAYER CLASSIFICATION ON ALL AVAILABLE DAYS

 +-----------------------------------------+
 |  Date of creation: 14 Apr. 2020         |
 +-----------------------------------------+
 |  Meteo-France                           |
 |  CNRM/GMEI/LISA                         |
 +-----------------------------------------+
'''

import os
import numpy as np
import datetime as dt

# Local packages
from blusc import multidays
from blusc import graphics

CEI_dir = "../working-directories/0-original-data/CEILOMETER/"
MWR_dir = "../working-directories/0-original-data/MWR/"
CEI_file = CEI_dir+"PASSY_PASSY_CNRM_CEILOMETER_CT25K_2015_0219_V01.nc"
MWR_file = MWR_dir+"PASSY2015_SALLANCHES_CNRM_MWR_HATPRO_2015_0219_V01.nc"

outputDir = ""
algo="LabelSpreading"

graphics.storeImages=False
graphics.figureDir = outputDir

multidays.supervised_path(CEI_file, MWR_file, algo=algo, outputDir=outputDir)

