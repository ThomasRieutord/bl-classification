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
from blclassification import blclassification
from prepdataset import prepdataset,check_availability


CEI_dir = "original_data/"
MWR_dir = "original_data/"

#daybothavail = check_availability(CEI_dir,MWR_dir)
daybothavail = [dt.datetime(2015,2,18),dt.datetime(2015,2,19),dt.datetime(2015,2,20)]
n_K=6
CH_index=[]
S_index=[]

for day in daybothavail:
	print("\n-- Day",day)
	CEI_file = scandate_inv(day)
	MWR_file = scandate_inv(day,campaign='PASSY2015',site='SALLANCHES',techno='MWR',instru='HATPRO')
	
	dailydir=day.strftime('%Y%m%d')+"/"
	if not os.path.isdir(dailydir):
		os.mkdir(dailydir)
	
	datasetpath=prepdataset(CEI_dir+CEI_file,MWR_dir+MWR_file,saveNetcdf=True,storeImages=True,outputDir=dailydir)
	
	if os.path.isfile(datasetpath):
		CH_values,S_values = blclassification(datasetpath,day,figureDir=dailydir,storeImages=True)
	else:
		CH_values=np.full(n_K,np.nan)
		S_values=np.full(n_K,np.nan)
	CH_index.append(CH_values)
	S_index.append(S_index)

CH_index=np.array(CH_index)
S_index=np.array(S_index)
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
plt.figure()
plt.title("Evolution of CH index")
plt.plot(daybothavail,np.mean(CH_index,axis=1),'k--',linewidth=2,label="K="+str(k+2))
for k in range(n_K):
	plt.plot(daybothavail,CH_index[:,k],'-o',linewidth=2,label="K="+str(k+2))
plt.gcf().autofmt_xdate()
plt.grid()
plt.legend(loc='best')
plt.show(block=False)
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
plt.figure()
plt.title("Evolution of silouhette index")
plt.plot(daybothavail,np.mean(S_index,axis=1),'k--',linewidth=2,label="K="+str(k+2))
for k in range(n_K):
	plt.plot(daybothavail,S_index[:,k],'-o',linewidth=2,label="K="+str(k+2))
plt.gcf().autofmt_xdate()
plt.grid()
plt.legend(loc='best')
plt.show(block=False)
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
input("\n Press Enter to exit (close down all figures)\n")
