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

def blclassification(path_to_dataset,linkageStategy="average",metricName="cityblock",normStrategy='meanstd',storeImages=False,fmtImages='.png',figureDir=""):
	'''Perform boundary layer classificationwith ascending hierarchical
	clustering on one of the devoted dataset from Passy-2015 campaign.
	This function has been written in order to easily change the parameters
	of hierarchical clustering : the linkage method, the metric used, the normalisation.
	
	[IN]
		- path_to_dataset (str): path where is located the dataset
		- linkageStategy (str): linkage method in hierarchical clustering
				Must be one admissible by scipy.cluster.hierarchy.linkage
		- metricName (str): metric to use between points
				Must be one admissible by scipy.spatial.distance.pdist
		- normStrategy (str): set how the dataset will be normalised
				Must be one among ['meanstd','minmax','none']
		- storeImages (opt, bool): if True, the figures are saved in the figureDir directory. Default is False
		- fmtImages (opt, str): format under which figures are saved when storeImages=True. Default is .png
		- figureDir (opt, str): directory in which figures are saved when storeImages=True. Default is current directory.
	
	[OUT]
		- CH_values (np.array[6]): Calinski-Harabaz scores when the number of clusters ranges from 2 to 7
		- S_values (np.array[6]): Silhouette scores when the number of clusters ranges from 2 to 7
		+ create and save figures (see scipy.hierarchy.dendrogram, utils.cluster2Dview_multi, utils.clusterZTview_multi)'''
	
	# LOADING AND CHECKING DATASET
	#==============================
	
	#print("Loading dataset :",datasetname)
	X_raw,predictors,interpMethod,z_common,t_common,z_max,Dz,Dt=load_dataset(dataDir+datasetname,t0=dt.datetime(2015,2,19))
	#print("Shape X_raw=",np.shape(X_raw))
	#print("Percentage of NaN=",100*np.sum(np.isnan(X_raw))/np.size(X_raw),"%")
	
	##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	#plt.figure()
	#plt.title("Data for classification : temperature and backscatter")
	#if 'Z' in predictors:
	    #plt.scatter(X_raw[:,0],X_raw[:,1],c=X_raw[:,2],cmap='Blues')
	    #plt.colorbar(label="Altitude (m agl)")
	#else:
	    #plt.scatter(X_raw[:,0],X_raw[:,1],s=5)
	#plt.xlabel("Backscatter (dB)")
	#plt.ylabel("Temperature (K)")
	#plt.show(block=False)
	##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	
	X=normalization(X_raw,strategy=normStrategy,return_toolbox=False)
	
	##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	#plt.figure()
	#plt.title("Data after normalisation")
	#if 'Z' in predictors:
	    #plt.scatter(X[:,0],X[:,1],c=X[:,2],cmap='Blues')
	    #plt.colorbar(label="Altitude (m agl)")
	#else:
	    #plt.scatter(X[:,0],X[:,1],s=5)
	#plt.xlabel("Backscatter")
	#plt.ylabel("Temperature")
	#plt.show(block=False)
	##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	
	
	# HIERARCHICAL CLUSTERING
	#=========================
	
	from scipy.cluster import hierarchy as cha
	
	# Get hierarchy
	#---------------
	
	
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
	#DB_values = []
	zoneIDs = []
	
	for target_nb_clusters in K_values:
		zoneID=cha.fcluster(linkageMatrix,t=target_nb_clusters,criterion='maxclust')
		zoneIDs.append(zoneID)
		
		# Quality scores
		#----------------
		
		# Calinski-Harabaz index
		ch_score=metrics.calinski_harabaz_score(X,zoneID)
		CH_values.append(ch_score)
		
		# Silhouette score
		s_score=metrics.silhouette_score(X,zoneID,metric=metricName)
		S_values.append(s_score)
		
		## Davies-Bouldin index (with sklearn 0.20)
		#db_score=metrics.davies_bouldin_score(X,zoneID,metric=metricName)
		#print("Davies-Bouldin index:",db_score)
		
		#clusterZTview(t_common,z_common,zoneID,storeImages=storeImages,fmtImages=fmtImages,figureDir=figureDir)
		
		#cluster2Dview(X_raw[:,0],predictors[0],X_raw[:,1],predictors[1],zoneID,storeImages=storeImages,fmtImages=fmtImages,figureDir=figureDir)
	
	cluster2Dview_multi(X_raw[:,0],predictors[0],X_raw[:,1],predictors[1],zoneIDs,storeImages=storeImages,fmtImages=fmtImages,figureDir=figureDir)
	clusterZTview_multi(t_common,z_common,zoneIDs,storeImages=storeImages,fmtImages=fmtImages,figureDir=figureDir)
	
	return CH_values,S_values




if __name__=='__main__':

	# Font change
	import matplotlib as mpl
	mpl.rc('font',family='Fira Sans')
	mpl.rcParams.update({'font.size': 16})
	mpl.rc('xtick',labelsize=13)
	mpl.rc('ytick',labelsize=13)
	
	dataDir = "datasets/"
	datasetname="DATASET_PASSY2015_BT-T_linear_dz40_dt30_zmax2000.nc"
	
	storeImages=True
	fmtImages='.png'
	figureDir=""
	
	normStrategy='meanstd'
	linkageStategy="average"
	metricName="cityblock"
	
	blclassification(dataDir+datasetname)
	

	input("\n Press Enter to exit (close down all figures)\n")
