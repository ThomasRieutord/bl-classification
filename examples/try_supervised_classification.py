#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
LAUNCHING SCRIPT FOR KABL PROGRAM.

 +-----------------------------------------+
 |  Date of creation: 4 Oct. 2019          |
 +-----------------------------------------+
 |  Meteo-France                           |
 |  DSO/DOA/IED and CNRM/GMEI/LISA         |
 +-----------------------------------------+
 
Copyright Meteo-France, 2019, [CeCILL-C](https://cecill.info/licences.en.html) license (open source)

This module is a computer program that is part of the KABL (K-means for 
Atmospheric Boundary Layer) program. This program performs boundary layer
height estimation for concentration profiles using K-means algorithm.

This software is governed by the CeCILL-C license under French law and
abiding by the rules of distribution of free software.  You can  use,
modify and/ or redistribute the software under the terms of the CeCILL-C 
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability.

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or 
data to be ensured and,  more generally, to use and operate it in the
same conditions as regards security.

The fact that you are presently reading this means that you have had
knowledge of the CeCILL-C license and that you accept its terms.
'''

import os
import numpy as np
import datetime as dt

# Local packages
from blcovid import multidays
from blcovid import graphics

CEI_dir = "../working-directories/0-original-data/CEILOMETER/"
MWR_dir = "../working-directories/0-original-data/MWR/"
CEI_file = CEI_dir+"PASSY_PASSY_CNRM_CEILOMETER_CT25K_2015_0219_V01.nc"
MWR_file = MWR_dir+"PASSY2015_SALLANCHES_CNRM_MWR_HATPRO_2015_0219_V01.nc"

outputDir = ""
algo="LabelSpreading"

graphics.storeImages=True
graphics.figureDir = outputDir

multidays.supervised_path(CEI_file, MWR_file, algo=algo, outputDir=outputDir)

