#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
PERFORM UNSUPERVISED BOUNDARY LAYER CLASSIFICATION ON ALL AVAILABLE DAYS

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
from blusc import graphics
from blusc import multidays

CEI_dir = "../working-directories/0-original-data/CEILOMETER/"
MWR_dir = "../working-directories/0-original-data/MWR/"
CEI_file = CEI_dir+"PASSY_PASSY_CNRM_CEILOMETER_CT25K_2015_0219_V01.nc"
MWR_file = MWR_dir+"PASSY2015_SALLANCHES_CNRM_MWR_HATPRO_2015_0219_V01.nc"

outputDir = ""
algo="hierarchical-average.euclidean"
target_nb_clusters='auto'

graphics.storeImages=False
graphics.figureDir = outputDir

multidays.unsupervised_path(CEI_file, MWR_file, algo=algo, target_nb_clusters=target_nb_clusters, outputDir=outputDir)


