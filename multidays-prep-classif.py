#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Intented for Python 3.5

Prepare dataset from one day of data from Passy-2015 field campaign
and perform boundary layer classification on the entire day.
 +------------------------------------------------------+
 |	CNRM (UMR 3589) - Meteo-France, CNRS 				|	
 |	GMEI/LISA 											|	
 +------------------------------------------------------+
Copyright (C) 2019  CNRM/GMEI/LISA Thomas Rieutord

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from bl-classification import blclassification
from prep-dataset import prepdataset,check_availability


CEI_dir = "/cnrm/lisa/data1/MANIP/PASSY-DATA-MAIN/Telemetre/netcdfCT25K/"
MWR_dir = "/cnrm/lisa/data1/Rieutord/PASSY_vrac/Radiometre/"#"original_data/

daybothavail = check_availability(CEI_dir,MWR_dir)

for day in daybothavail:
	print("\n-- Day",day)
	CEI_file = scandate_inv(day)
	MWR_file = scandate_inv(day,campaign='PASSY2015',site='SALLANCHES',techno='MWR',instru='HATPRO')
	print("CEI_file",CEI_file)
	print("MWR_file",MWR_file)
