#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Intented for Python 3.5

Prepare dataset from the 2nd IOP of Passy-2015 data in order to apply
boudary layer classification.
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

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from utils import *

# SETTINGS
#==========

# Parameters of the interpolation
#---------------------------------
z_max=2000						# Maximum altitude (m agl)
predictors = ['BT','T']			# Variables to put in the dataset (['BT','T'] or ['BT','T','Z'])
dt_common = 'MWR'#30			# Time resolution common grid
dz_common = 'MWR'#40			# Vertical resolution common grid
interpMethod = 'linear'			# Interpolation method ('linear','cubic','nearestneighbors')

# Outputs
#---------
outputDir="datasets/"
saveNetcdf=True
figureDir = ""
storeImages=False
fmtImages='.png'

# Inputs
#--------
day=dt.datetime(2015,2,19)
dataDir = "original_data/"
CEI_file = "PASSY_PASSY_CNRM_CEILOMETER_CT25K_2015_0219_V01.nc"
MWR_file = "PASSY2015_SALLANCHES_CNRM_MWR_HATPRO_2015_0219_V01.nc"


# EXTRACTION
#============

# Raw data
#----------
t_cei,z_cei,backscatter=extractOneFile(dataDir+CEI_file,altmax=z_max)
t_mwr,z_mwr,temperature=extractOneFile(dataDir+MWR_file,altmax=z_max)
# NB: here the backscatter signal is raw, it is NOT in decibel.

# Sanity checks
#---------------
print('\n--- CEILOMETER ---')
print("Size=",np.size(backscatter))
print("Percentage of NaN=",100*np.sum(np.isnan(backscatter))/np.size(backscatter),"%")
with np.errstate(invalid='ignore'):
	print("Percentage of negative values=",100*np.sum(backscatter<0)/np.size(backscatter),"%")
print("VALUES : min=",np.nanmin(backscatter),"max=",np.nanmax(backscatter),"mean=",np.nanmean(backscatter),"median=",np.nanmedian(backscatter))
print("GRID : dt=",np.mean(np.diff(t_cei)),"dz=",np.mean(np.diff(z_cei)),"Nt=",len(t_cei),"Nz=",len(z_cei),"data shape=",np.shape(backscatter))
print('------')
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
plt.figure()
plt.title("Original backscatter")
with np.errstate(divide='ignore',invalid='ignore'):
	plt.pcolormesh(t_cei,z_cei,10*np.log10(backscatter).T,vmin=-11,vmax=20,cmap='plasma')
plt.colorbar(label="Aerosol Backscatter (dB)")
plt.gcf().autofmt_xdate()
plt.xlabel("Time (UTC)")
plt.ylabel("Alt (m agl)")
plt.show(block=False)
if storeImages:
	plt.savefig(figureDir+"BT_orig"+fmtImages)
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
print('\n--- RADIOMETER ---')
print("Size=",np.size(temperature))
print("Percentage of NaN=",100*np.sum(np.isnan(temperature))/np.size(temperature),"%")
print("VALUES : min=",np.nanmin(temperature),"max=",np.nanmax(temperature),"mean=",np.nanmean(temperature),"median=",np.nanmedian(temperature))
print("GRID : dt=",np.mean(np.diff(t_mwr)),"dz=",np.mean(np.diff(z_mwr)),"Nt=",len(t_mwr),"Nz=",len(z_mwr),"data shape=",np.shape(temperature))
print('------')
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
plt.figure()
plt.title("Original temperature")
plt.pcolormesh(t_mwr,z_mwr,temperature.T,vmin=260,vmax=285,cmap='jet')
plt.colorbar(label="Temperature (K)")
plt.gcf().autofmt_xdate()
plt.xlabel("Time (UTC)")
plt.ylabel("Alt (m agl)")
plt.show(block=False)
if storeImages:
	plt.savefig(figureDir+"T_orig"+fmtImages)
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# Negative backscatter are outliers
with np.errstate(invalid='ignore'):
	backscatter[backscatter<=0]=np.nan



# INTERPOLATION
#===============


# Arrival grid
#--------------

if dz_common=='MWR':
	# Autre alternative : prendre une grille existante.
	z_common,t_common=z_mwr,t_mwr
else:
	t_min=dt.datetime(2015,2,19)
	t_max=dt.datetime(2015,2,20)
	z_common,t_common=generategrid(t_min,t_max,z_max,dz_common,dt_common)


# Interpolation
#---------------

print("\nInvalid values after interpolation:")
# Radiometer
temperature[:,0]=np.nan		# First level is bad
if dz_common=='MWR' and np.sum(np.isnan(temperature[:,1:]))==0:
	T_query=temperature
	T_query[:,0]=np.nan		# First level is bad
else:
	T_query=estimateongrid(z_common,t_common,z_mwr,t_mwr,temperature,method=interpMethod,crossval=False)

with np.errstate(invalid='ignore'):
	print(" - T_query - #NaN=",np.sum(np.isnan(T_query)),"#Inf=",np.sum(np.isinf(T_query)),"#Neg=",np.sum(T_query<=0))

# Ceilometer
BT_query=estimateongrid(z_common,t_common,z_cei,t_cei,backscatter,method=interpMethod,crossval=False)

with np.errstate(invalid='ignore'):
	print(" - BT_query - #NaN=",np.sum(np.isnan(BT_query)),"#Inf=",np.sum(np.isinf(BT_query)),"#Neg=",np.sum(BT_query<=0))

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
plt.figure()
plt.title("Interpolated temperature")
plt.pcolormesh(t_common,z_common,T_query.T,vmin=260,vmax=285,cmap='jet')
plt.colorbar(label="Temperature (K)")
plt.gcf().autofmt_xdate()
plt.xlabel("Time (UTC)")
plt.ylabel("Alt (m agl)")
plt.show(block=False)
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
plt.figure()
plt.title("Interpolated backscatter")
with np.errstate(divide='ignore',invalid='ignore'):
	plt.pcolormesh(t_common,z_common,10*np.log10(BT_query).T,vmin=-11,vmax=20,cmap='plasma')
plt.colorbar(label="Aerosol Backscatter (dB)")
plt.gcf().autofmt_xdate()
plt.xlabel("Time (UTC)")
plt.ylabel("Alt (m agl)")
plt.show(block=False)
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# Shape the data
# ---------------
# Some data can still be missing even after the interpolation
#   * Radiometer : resolution coarsens with altitude => last gates missing
#   * Ceilometer : high lowest range => first gates missing

trash,mask_T=deletelines(T_query,nan_max=T_query.shape[0]-1,return_mask=True,transpose=True)

trash,mask_BT=deletelines(BT_query,nan_max=BT_query.shape[0]-1,return_mask=True,transpose=True)

z_common=z_common[~np.logical_or(mask_T,mask_BT)]
T=T_query[:,~np.logical_or(mask_T,mask_BT)]
BT=BT_query[:,~np.logical_or(mask_T,mask_BT)]

# Convert backscatter to decibel
BT=10*np.log10(BT)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
plt.figure()
plt.title("Cleaned backscatter")
plt.pcolormesh(t_common,z_common,BT.T,vmin=-11,vmax=18,cmap='plasma')
plt.colorbar(label="Aerosol Backscatter (dB)")
plt.gcf().autofmt_xdate()
plt.xlabel("Time (UTC)")
plt.ylabel("Alt (m agl)")
plt.show(block=False)
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
plt.figure()
plt.title("Cleaned temperature")
plt.pcolormesh(t_common,z_common,T.T,vmin=260,vmax=285,cmap='jet')
plt.colorbar(label="Temperature (K)")
plt.gcf().autofmt_xdate()
plt.xlabel("Time (UTC)")
plt.ylabel("Alt (m agl)")
plt.show(block=False)
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# Add altitude in predictors (optional)
if 'Z' in predictors:
	X_raw = np.array([BT.ravel(),T.ravel(),np.tile(z_common,(T.shape[0],1)).ravel()]).T
else:
	X_raw = np.array([BT.ravel(),T.ravel()]).T

# Sanity checks
print("\nInvalid values after removing bad boundaries:")
with np.errstate(invalid='ignore'):
	print(" - X_raw - #NaN=",np.sum(np.isnan(X_raw)),"#Inf=",np.sum(np.isinf(X_raw)),"#Neg=",np.sum(X_raw<=0))
print("shape(X_raw)=",X_raw.shape)



n_invalidValues = np.sum(np.isnan(X_raw))+np.sum(np.isinf(X_raw))


# Write dataset in netcdf file
#------------------------------
# The matrix X_raw is stored with the names of its columns and the grid on which is has been estimated.

if saveNetcdf:
	datasetname='DATASET_PASSY2015_'+'-'.join(predictors)+'_'+interpMethod+'_dz'+str(dz_common)+'_dt'+str(dt_common)+'_zmax'+str(z_max)+'.nc'
	msg=write_dataset(outputDir+datasetname,X_raw,t_common,z_common)
	print(msg)

input("\n Press Enter to exit (close down all figures)\n")
