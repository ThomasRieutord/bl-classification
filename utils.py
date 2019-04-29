# -*- coding: utf-8 -*-
"""
Intented for Python 3.5

Toolbox package for boundary layer classification.
 1. Conversion datetime/float
 2. Conversion grid/scatter
 3. Loading data
 4. Graphics

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
import datetime as dt
import netCDF4 as nc
import matplotlib.pyplot as plt

#--------------------------------------------------
# 	List of datetime / List of seconds
#--------------------------------------------------

def dtlist2slist(dtlist):
	'''Convert a list of datetime into a list of seconds.
	The first datetime object of the list is taken as origin of time.
	
	[IN]
		- dtlist (list of datetime): ist of datetime objects
		
	[OUT]
		- slist (list of float): list of seconds spent since the origin of time.'''
	
	slist = []
	for t in dtlist:
		slist.append( (t-dtlist[0]).total_seconds() )
	
	return slist


def slist2dtlist(slist,t0):
	'''Convert a list of seconds into a list of datetime The origin of
	time (second zero) must be specified by a datetime object.
	
	[IN]
		- slist (list of float): list of seconds spent since the origin of time.
		- t0 (datetime): origin of time.
		
	[OUT]
		- dtlist (list of datetime): list of datetime objects'''
	
	dtlist = []
	for t in slist:
		dtlist.append( t0 + dt.timedelta(seconds=t))
	
	return dtlist

#--------------------------------------------------
# 	Grid x[nx],y[ny],z[nx,ny] / Scatter X[nx*ny,2],Y[nx*ny]
#--------------------------------------------------

def grid_to_scatter(x,y,z=None):
	'''Convert grid point data into scattered points data.
	
	Grid point data : (x,y,z) with z[i,j]=f(x[i],y[j])
	Scatter point data : (X,Y) with Y[i]=f(X[i,0],X[i,1])
	
	[IN]
		- x (np.array[nx]): coordinates on X-axis
		- y (np.array[ny]): coordinates on Y-axis
		- z (np.array[nx,ny]): data for each point (x,y)
	
	[OUT]
		- X (np.array[nx*ny,2]): coordinate matrix
		- Y (np.array[nx*ny]): data vector'''
	nx=np.size(x)
	ny=np.size(y)
	
	X=np.full((nx*ny,2),np.nan)
	X[:,0]=np.repeat(x,ny)
	X[:,1]=np.tile(y,nx)
	if np.isnan(X).any():
		print(" /!\ Warning: ",np.isnan(X).sum()," NaN in X (grid_to_scatter)")
	
	if z is None:
		result = X
	else:
		if np.size(z)!=nx*ny:
			raise Exception("Problem with inputs dimensions (grid_to_scatter)")
		Y=z.ravel()
		result = X,Y
	
	return result


def scatter_to_grid(X,Y=None):
	'''Convert scattered points data into grid point data.
	
	Grid point data : (x,y,z) with z[i,j]=f(x[i],y[j])
	Scatter point data : (X,Y) with Y[i]=f(X[i,0],X[i,1])
	
	[IN]
		- X (np.array[nx*ny,2]): coordinate matrix
		- Y (np.array[nx*ny]): data vector
	
	[OUT]
		- x (np.array[nx]): coordinates on X-axis
		- y (np.array[ny]): coordinates on Y-axis
		- z (np.array[nx,ny]): data for each point (x,y)'''
	N,d=np.shape(X)
	#print("N=",N)
	if d!=2:
		raise Exception("More than 2 inputs. Not ready so far (scatter_to_grid)")
	
	if np.sum(np.diff(X[:,0])==0)>np.sum(np.diff(X[:,1])==0):
		xcoord=0
	else:
		xcoord=1
	ny = (np.diff(X[:,xcoord])==0).tolist().index(False)+1
	#print("ny=",ny)
	if np.mod(N/ny,1)!=0:
		raise Exception("Number of points doesn't match with dimensions")
	nx = int(N/ny)
	#print("nx=",nx)
	x = X[0:N:ny,xcoord]
	y = X[0:ny,1-xcoord]
	if Y is None:
		result = x,y
	else:
		if np.size(Y)!=N:
			raise Exception("Problem with inputs dimensions (scatter_to_grid)")
		z = np.reshape(Y,(nx,ny))
		result = x,y,z
	
	return result

#--------------------------------------------------
# 	Loading data
#--------------------------------------------------

def load_dataset(datasetname,t0=dt.datetime(2015,2,19)):
	'''Charge un jeu de donnée trié depuis un fichier netCDF (boundary layer classification).
	Convention pour nommer les dataset :
		'DATASET_CAMPAGNE_PREDICTEURS_INTERPOLATION_dz***_dt***_zmax***.nc'
	
	[IN]
		- loadfilename (str): chemin et nom du fichier à charger.
		
	[OUT]
		- profile_values (np.array): valeurs du profil.
		- z_values (np.array): altitudes où les valeurs sont prises.
		- prototypeID (str): identifying key of the prototype.'''
	
	Filename=datasetname.split("/")[-1]
	dataset,campagne,predicteurs,interpMethod,dzx,dtx,zmaxnc=Filename.split("_")
	
	predictors=predicteurs.split('-')
	
	try:
		Dz=int(dzx[2:])
	except ValueError:
		Dz=dzx[2:]
	try:
		Dt=int(dtx[2:])
	except ValueError:
		Dt=dtx[2:]
	zmax=int(zmaxnc.split('.')[0][4:])
	
	dataset = nc.Dataset(datasetname)
	
	z_common = np.array(dataset.variables['altitude'])
	t_common = slist2dtlist(dataset.variables['time'],t0)
	X_raw = np.array(dataset.variables['X_raw'])
	
	return X_raw,predictors,interpMethod.lower(),z_common,t_common,zmax,Dz,Dt


def normalization(X_raw,strategy,return_toolbox=False):
	'''Normalize a raw observation matrix according to the specified strategy.
	If strategy='meanstd', it removes the mean and divide by the standard deviation, columnwise.
	If strategy='minmax', it removes the min and divide by the maximum amplitude, columnwise.
	If strategy='none', it does nothing.
	
	[IN]
		- X_raw (np.array[N_raw,p]): dataset with one obs per line and p variables per obs.
		- strategy (str): type of normalization. Must be one of ['meanstd','minmax','none']
	
	[OUT]
		- X_norm (np.array[N_raw,p]): dataset with one obs per line and p variables per obs.'''
	
	if return_toolbox:
		unnorm_toolbox = {"strategy":strategy}
	
	if strategy=="meanstd":
		X_norm=(X_raw-np.nanmean(X_raw,axis=0))/np.nanstd(X_raw,axis=0)
		if return_toolbox:
			unnorm_toolbox["mean"]=np.nanmean(X_raw,axis=0)
			unnorm_toolbox["std"]=np.nanstd(X_raw,axis=0)
	elif strategy=="minmax":
		X_norm=(X_raw-np.nanmin(X_raw,axis=0))/(np.nanmax(X_raw,axis=0)-np.nanmin(X_raw,axis=0))
		if return_toolbox:
			unnorm_toolbox["min"]=np.nanmin(X_raw,axis=0)
			unnorm_toolbox["max"]=np.nanmax(X_raw,axis=0)
	elif strategy=="none":
		X_norm=X_raw
	else:
		raise Exception("Unknown normalization. Must be in ['meanstd','minmax','none']")
	
	
	if return_toolbox:
		result=X_norm,unnorm_toolbox
	else:
		result=X_norm
	
	return result

#--------------------------------------------------
# 	Graphics
#--------------------------------------------------

def cluster2Dview(variable1,varname1,variable2,varname2,zoneID,clustersIDs=None,storeImages=False,fmtImages=".svg",figureDir=""):
	'''Plots the projection of the clusters onto the space generated by
	two predictors. It can be used to visualize clusters (boundary layer classification).
	
	[IN]
		- variable1 (np.array[N]): first variable (vector of values, regardless with their coordinates)
		- varname1 (str): standard name of first variable
		- variable2 (np.array[N]): second variable (vector of values, regardless with their coordinates)
		- varname2 (str): standard name of second variable
		- zoneID (np.array[N]): cluster labels for each point
		- storeImages (opt, bool): if True, the figures are saved in the figureDir directory. Default is False
		- fmtImages (opt, str): format under which figures are saved when storeImages=True. Default is .svg
		- figureDir (opt, str): directory in which figures are saved when storeImages=True. Default is current directory.
		
	[OUT]
		Nothing.'''
	
	# Database of names and marks for the plots
	clusterMarks={1:'bo',2:'gx',3:'r^',4:'cv',5:'ys',6:'m*',7:'kp',8:'gd',9:'bx',10:'ro',11:'c*',12:'y+',13:'m<',14:'k,',
		'SBL':'bo',
		'FA':'gx',
		'ML':'r^',
		'EZ':'cv',
		'CP':'ys'}
	DicLeg = {'DD':'Wind direction (deg)','FF':'Wind Intensity (m/s)',
	          'U':'Zonal Wind (m/s)','V':'Meridional Wind (m/s)', 'W':'Vertical Wind (m/s)',
	          'T': 'Temperature (K) ',   'THETA':'Potential Temperature (K) ',
	          'BT': 'Aerosol Backscatter (dB) ',   'SNRW':'Vertical SNR (dB) ',
	          'RH' :'Relative Humidity (%)', 'PRES':'pressure (hPa) '}
	if varname1 not in DicLeg.keys():
		DicLeg[varname1]=varname1
	if varname2 not in DicLeg.keys():
		DicLeg[varname2]=varname2
	if np.min(zoneID)!=0:
		print("Cluster labels didn't start from 0")
		zoneID-=np.min(zoneID)
	
	# Number of clusters
	K=int(np.max(zoneID)+1)
	
	if clustersIDs is None:
		clustersIDs=np.arange(K)+1
	
	#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	plt.figure()
	plt.title("Dataset partition | "+str(K)+" clusters")
	plt.plot(variable1,variable2,'k.')
	for k in range(K):
		plt.plot(variable1[np.where(zoneID==k)],variable2[np.where(zoneID==k)],clusterMarks[clustersIDs[k]],linewidth=2,label="Cluster "+str(clustersIDs[k]))
	plt.xlabel(DicLeg[varname1])
	plt.ylabel(DicLeg[varname2])
	plt.show(block=False)
	if storeImages:
		plt.savefig(figureDir+"cluster2Dview_"+varname1+'-'+varname2+"_K"+str(K)+fmtImages)
	#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def draw_clusterlabels(t_values,z_values,zoneID,delete_mask=None,clustersIDs=None,storeImages=False,fmtImages=".svg",figureDir=""):
	'''Plots cluster labels (areas) against time and height (boundary layer classification).
	
	[IN]
		- t_values (np.array[nt]): vector of time
		- z_values (np.array[nalt]): vector of altitude
		- zoneID (np.array[N]): cluster labels of each obs
		- delete_mask (np.array[nt*nalt]): mask at True when observation has been removed by the deletelines function (to avoid NaNs)
		- storeImages (opt, bool): if True, the figures are saved in the figureDir directory. Default is False
		- fmtImages (opt, str): format under which figures are saved when storeImages=True. Default is .svg
		- figureDir (opt, str): directory in which figures are saved when storeImages=True. Default is current directory.
		
	[OUT]
		Nothing.'''
	
	from matplotlib.colors import ListedColormap
	
	clusterMarks={1:'bo',2:'gx',3:'r^',4:'cv',5:'ys',6:'m*',7:'kp',8:'gd',9:'bx',10:'ro',11:'c*',12:'y+',13:'m<',14:'k,',
		'SBL':'bo',
		'FA':'gx',
		'ML':'r^',
		'EZ':'cv',
		'CP':'ys'}
	
	if np.min(zoneID)!=0:
		print("Cluster labels didn't start from 0")
		zoneID-=np.min(zoneID)
	
	# Number of clusters
	K=int(np.max(zoneID)+1)
	
	if clustersIDs is None:
		clustersIDs=np.arange(K)+1
	
	clist = []
	cticks = []
	cticklabels = []
	for k in range(K):
		cticks.append(k+0.5)
		cticklabels.append(clustersIDs[k])
		clist.append(clusterMarks[clustersIDs[k]][0])
	colormap=ListedColormap(clist)
	
	# 1. Complétion labels supprimés
	if delete_mask is not None:
		fullzoneID=np.full(np.size(delete_mask),np.nan)
		fullzoneID[~delete_mask]=zoneID
	else:
		fullzoneID=zoneID
	
	# 2. Conversion datetime -> secondes
	t0=t_values[0]
	st_values = dtlist2slist(t_values)
	
	# 3. Passage z,t au format scatter
	TZ = grid_to_scatter(st_values,z_values)
	
	# 4. Passage labels au format grille
	t_trash,z_trash,labels = scatter_to_grid(TZ,fullzoneID)
	if np.max(np.abs(z_values-z_trash))+np.max(np.abs(st_values-t_trash))>1e-13:
		raise Exception("Error in z,t retrieval : max(|z_values-z_trash|)=",np.max(np.abs(z_values-z_trash)),"max(|t_values-t_trash|)=",np.max(np.abs(st_values-t_trash)))
	
	labels=np.ma.array(labels,mask=np.isnan(labels))
	
	# 5. Graphique
	#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	plt.figure()
	plt.title("HIERARCHICAL clustering | "+str(K)+" clusters")
	plt.pcolormesh(t_values,z_values,labels.T,vmin=0,vmax=K,cmap=colormap)
	cbar=plt.colorbar(label="Cluster label")
	cbar.set_ticks(cticks)
	cbar.set_ticklabels(cticklabels)
	plt.gcf().autofmt_xdate()
	plt.xlabel("Time (UTC)")
	plt.ylabel("Alt (m agl)")
	plt.show(block=False)
	if storeImages:
		plt.savefig(figureDir+"clusterlabelsZT_K"+str(K)+fmtImages)
	#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
