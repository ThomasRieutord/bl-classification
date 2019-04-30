#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Intented for Python 3.5

Perform boundary layer classification on one of the dataset of Passy-2015 2nd IOP.
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
import datetime as dt
from sklearn import metrics
from utils import *

# Font change
import matplotlib as mpl
mpl.rc('font',family='Fira Sans')
mpl.rcParams.update({'font.size': 16})
mpl.rc('xtick',labelsize=13)
mpl.rc('ytick',labelsize=13)

storeImages=True
fmtImages='.png'
figureDir=""

dataDir = "datasets/"
datasetname="DATASET_PASSY2015_BT-T_linear_dz40_dt30_zmax2000.nc"

# LOADING AND CHECKING DATASET
#==============================

print("Loading dataset :",datasetname)
X_raw,predictors,interpMethod,z_common,t_common,z_max,Dz,Dt=load_dataset(dataDir+datasetname,t0=dt.datetime(2015,2,19))
print("Shape X_raw=",np.shape(X_raw))
print("Percentage of NaN=",100*np.sum(np.isnan(X_raw))/np.size(X_raw),"%")

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
plt.figure()
plt.title("Data for classification : temperature and backscatter")
if 'Z' in predictors:
    plt.scatter(X_raw[:,0],X_raw[:,1],c=X_raw[:,2],cmap='Blues')
    plt.colorbar(label="Altitude (m agl)")
else:
    plt.scatter(X_raw[:,0],X_raw[:,1],s=5)
plt.xlabel("Backscatter (dB)")
plt.ylabel("Temperature (K)")
plt.show(block=False)
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


X=normalization(X_raw,strategy='meanstd',return_toolbox=False)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
plt.figure()
plt.title("Data after normalisation")
if 'Z' in predictors:
    plt.scatter(X[:,0],X[:,1],c=X[:,2],cmap='Blues')
    plt.colorbar(label="Altitude (m agl)")
else:
    plt.scatter(X[:,0],X[:,1],s=5)
plt.xlabel("Backscatter")
plt.ylabel("Temperature")
plt.show(block=False)
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


# HIERARCHICAL CLUSTERING
#=========================

from scipy.cluster import hierarchy as cha

# Get hierarchy
#---------------

linkageStategy="average"
metricName="cityblock"

linkageMatrix=cha.linkage(X,method=linkageStategy,metric=metricName)


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
plt.figure()
plt.title(u"Dendrogramme")
cha.dendrogram(linkageMatrix,p=10,truncate_mode='level',distance_sort='ascending',color_threshold=2.25,no_labels=True)
plt.ylabel("Cophenetic distance")
plt.xlabel("Observations")
plt.show(block=False)
if storeImages:
	plt.savefig(figureDir+"dendrogramme"+fmtImages)
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


# Get flat clusters
#-------------------

K_values = np.arange(2,8)
CH_values = []
S_values = []
DB_values = []
zoneIDs = []

for target_nb_clusters in K_values:
	print(" - Looking for ",target_nb_clusters,"clusters")
	zoneID=cha.fcluster(linkageMatrix,t=target_nb_clusters,criterion='maxclust')
	zoneIDs.append(zoneID)
	
	clusterZTview(t_common,z_common,zoneID,storeImages=storeImages,fmtImages=fmtImages,figureDir=figureDir)
	
	cluster2Dview(X_raw[:,0],predictors[0],X_raw[:,1],predictors[1],zoneID,storeImages=storeImages,fmtImages=fmtImages,figureDir=figureDir)

print("Plot results of",len(zoneIDs),"classifications")
cluster2Dview_multi(X_raw[:,0],predictors[0],X_raw[:,1],predictors[1],zoneIDs,storeImages=storeImages,fmtImages=fmtImages,figureDir=figureDir)
clusterZTview_multi(t_common,z_common,zoneIDs,storeImages=storeImages,fmtImages=fmtImages,figureDir=figureDir)

# Quality scores
#----------------

# Calinski-Harabaz index
ch_score=metrics.calinski_harabaz_score(X,zoneID)
print("Calinski-Harabaz index:",ch_score)

# Silhouette score
s_score=metrics.silhouette_score(X,zoneID,metric=metricName)
print("Silhouette score:",s_score)

## Davies-Bouldin index (with sklearn 0.20)
#db_score=metrics.davies_bouldin_score(X,zoneID,metric=metricName)
#print("Davies-Bouldin index:",db_score)


input("\n Press Enter to exit (close down all figures)\n")
