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


def instrumentkind(path):
    '''Give the type of instrument that made the measures in the netcdf file.
    This type of instrument is used as a standard ID in other functions like quickplot.
    
    [IN]
		- path (str): path of the file
		
	[OUT]
		- (str): instrument ID'''
    fichier=path.split('/')[-1]
    csltiymdv = fichier.split('_')
    return csltiymdv[4].lower()


def scandate(path):
    ''' Give date and time when the measurement has been done, from its name.
    It is the date of the first measure. Hours are UTC.
    
    [IN]
		- path (str): path to the file
		
	[OUT]
		- (datetime): date of the file'''
    fichier=path.split('/')[-1]
    campaign,site,lab,techno,instru,yyyy,mmdd,v01 = fichier.split('_')
    return dt.datetime(int(yyyy),int(mmdd[0:2]),int(mmdd[2:4])) 

def scandate_inv(day,campaign='PASSY',site='PASSY',lab='CNRM',techno='CEILOMETER',instru='CT25K',v01='V01.nc'):
	''' Give the name of the file corresponding to the measurement at the given date.
	
	[IN]
		- day (datetime): date of the file
		
	[OUT]
		- filename (str): name of the file'''
	yyyy=day.strftime('%Y')
	mmdd=day.strftime('%m%d')
	return '_'.join([campaign,site,lab,techno,instru,yyyy,mmdd,v01])


def extractOneFile(path,vartoextract='Default',altmax=4000,convert_to_dB=False):
	'''Extract data from a netcdf file at the given path.
	Some variables are extracted by default by it is highly recommended
	to precise them. Input files must match the GMEI standard.
	The name of the files is standardized as such:

	CAMPAIGNYYYY_SITE_LAB_TECHNO_INSTRUMENTID_YYYY_MMDD_VERSION.nc
	
	[IN]
		- path (str): path of the file to be read
		- vartoextract (str, opt): variables to extract (depends on the kind of intrument)
		- altmax (float): highest altitude to use (m agl)
		- option (str, opt) : caractères supplémentaires pour trier
		
	[OUT]
		- Taxis (np.array[nt] of datetimes): time vector (datetime objects)
		- Zaxis (np.array[nalt]): altitude vector (m agl)
		- dataOut (np.array[nt,nalt]): first variable extracted'''
	
	# General infos
	defautvartoextract={'ct25k':'BT','hatpro': 'T'}
	
	instru=instrumentkind(path)
	data= nc.Dataset(path)
	
	# Time axis
	Taxis= []
	t0=scandate(path)
	imin=0
	imax=len(data.variables['time'])
	for i in range(len(data.variables['time'])):
		Taxis.append(t0+dt.timedelta(seconds=int(data.variables['time'][i])))
	
	# Altitude axis
	altmaxdex = np.where(data.variables['Z'][:]<altmax)[0][-1]+1
	Zaxis = data.variables['Z'][:altmaxdex]
	
	# Data extraction
	if vartoextract=='Default':
		vartoextract = defautvartoextract[instru]
	
	if convert_to_dB:
		dataOut=10*np.log10(data.variables[vartoextract][:,:altmaxdex])
	else:
		dataOut=np.array(data.variables[vartoextract][:,:altmaxdex])
	
	return Taxis,Zaxis,dataOut

#--------------------------------------------------
# 	Prepare the data
#--------------------------------------------------

def generategrid(datemin,datemax,altmax,Dz,Dt,altmin=0):
	''' Generate a time-altitude grid at the given resolution.
	
	[IN]
		- datemin (datetime): starting time of the grid
		- datemax (datetime): ending time of the grid
		- altmax (float): maximum altitude of the grid (m agl).
		- Dz (float): spatiale resolution (m).
		- Dt (float or timedelta): time resolution (minutes).
		- altmin (float): maximum altitude of the grid (m agl). Default is 0.
		
	[OUT]
		- z_values (np.array[n_z]): altitude vector of the grid (m agl)
		- t_values (list[n_t] of datetime): time vector of the grid
	'''
	if isinstance(datemax,dt.timedelta):
		datefin=datemin+datemax
	else:
		datefin=datemax
	
	if isinstance(Dt,dt.timedelta):
		td=Dt
	else:
		td=dt.timedelta(minutes=Dt)
	
	n_t = int((datefin-datemin).total_seconds()/td.total_seconds())
	n_z = int(altmax/Dz)
	
	z_values = np.arange(altmin,altmax,Dz)
	
	t_values = []
	for k in range(n_t):
		t_values.append(datemin+k*td)
	
	return z_values,t_values

def estimateongrid(z_target,t_target,z_known,t_known,V_known,method='cubic',decibel=False,crossval=False,nfolds=10):
	'''Interpolate the data on a target grid knowning it on another grid.
	Grids are time-altitude.
	
	Interpolation is necessary for two reasons: getting rid of missing
	values and have the information of all instrument on a common grid.
	
	[IN]
		- z_target (np.array[n1_z]): altitude vector of the target grid (m agl)
		- t_target (list[n1_t] of datetime): time vector of the target grid
		- z_known (np.array[n0_z]): altitude vector of the known grid (m agl)
		- t_known (list[n0_t] of datetime): time vector of the known grid
		- V_known (np.array[n0_t,n0_z]): data values on the known grid
	[IN,OPT]
		- method (str): interpolation method
		- decibel (bool): set to True if the data is in decibel: the data is converted to physical unit before interpolation and converted back to decibel after.
		- crossval (bool): if True, evaluate the uncertainty of the estimation by Kfold cross-validation
		- nfolds (int): if crossval=True, number of folds in Kfold cross-validation
		
	[OUT]
		- V_target (np.array[n1_t,n1_z]): valeurs sur la grilles cible'''
	
	if decibel:
		print("V_known<=0:",np.sum(V_known<=0))
		V_known=np.exp(V_known/10)
		print("Nan of Inf after exp:",np.sum(np.isnan(V_known))+np.sum(np.isinf(V_known)))
	
	# Switch from format "data=f(coordinates)" to format "obs=f(predictors)"
	st_known = dtlist2slist(t_known)
	st_target = dtlist2slist(t_target)
	X_known,Y_known=grid_to_scatter(st_known,z_known,V_known)
	X_target=grid_to_scatter(st_target,z_target)
	
	
	#### ========= Estimation with 4-nearest neighbors
	if method=="nearestneighbors":
		from sklearn.neighbors import KNeighborsRegressor
		KNN=KNeighborsRegressor(n_neighbors=4)
		
		# NaN are removed
		X_known = X_known[~np.isnan(Y_known),:]
		Y_known = Y_known[~np.isnan(Y_known)]
		KNN.fit(X_known,Y_known)
		Y_target = KNN.predict(X_target)
	else:
	#### ========= Estimation with 2D interpolation
		from scipy.interpolate import griddata
	
		# NaN are removed
		X_known = X_known[~np.isnan(Y_known),:]
		Y_known = Y_known[~np.isnan(Y_known)]
		Y_target = griddata(X_known,Y_known,X_target,method=method)
	
	
	if crossval:
		from sklearn.model_selection import KFold
		
		kf=KFold(n_splits=nfolds,shuffle=True,random_state=14)
		kf_error = []
		for train_dex,test_dex in kf.split(X_known):
			Y_pred = griddata(X_known[train_dex],Y_known[train_dex],X_known[test_dex],method=method)
			kf_error.append(np.sqrt(np.mean((Y_pred-Y_known[test_dex])**2)))
		prediction_error = np.nanmean(np.array(kf_error))
		print("prediction_error =",prediction_error)
		
	# Mise en forme sortie
	t1,z1,V_target = scatter_to_grid(X_target,Y_target)
	
	if decibel:
		print("V_target<=0:",np.sum(V_target<=0))
		V_target = 10*np.log10(V_target+np.finfo(float).eps)
		print("Nan or Inf after log:",np.sum(np.isnan(V_target))+np.sum(np.isinf(V_target)))
	
	# Sanity checks
	if np.shape(V_target) != (np.size(st_target),np.size(z_target)):
		raise Exception("Output has not expected shape : shape(st_target)",np.shape(st_target),"shape(z_target)",np.shape(z_target),"shape(V_target)",np.shape(V_target))
	if (np.abs(t1-st_target)>10**(-10)).any():
		raise Exception("Time vector has been altered : max(|t1-t_target|)=",np.max(np.abs(t1-st_target)))
	if (np.abs(z1-z_target)>10**(-10)).any():
		raise Exception("Altitude vector has been altered : max(|z1-z_target|)=",np.max(np.abs(z1-z_target)))
	
	if crossval:
		result = V_target,prediction_error
	else:
		result = V_target
	
	return result


def deletelines(X_raw,nan_max=None,return_mask=False,transpose=False,verbose=False):
	'''Remove lines containing many Not-a-Number values from the dataset.
	
	All lines containing nan_max missing values or more are also removed.
	By default, nan_max=p-1 (lines with only NaN or only but 1 are removed)
	
	[IN]
		- X_raw (np.array[N_raw,p]): original dataset.
		- nan_max (int, opt): maximum number of missing value tolerated. Default is p-1
		- return_mask (bool): if True, returns the mask at True for deleted lines
		- transpose (bool): if True, removes the columns instead of the lines
		- verbose (bool): if True, does more prints
	
	[OUT]
		- X (np.array[N,p]): filtered dataset.
		- delete_mask (np.array[N_raw] of bool): if return_mask=True, mask at True for deleted lines'''
	
	if transpose:
		X_raw=X_raw.T
	
	N_raw,p = np.shape(X_raw)
	if nan_max is None:
		nan_max=p-1
	
	if verbose:
		print("Delete all lines with ",nan_max,"missing values or more.")
	to_delete = []
	numberOfNan=np.zeros(N_raw)
	for i in range(N_raw):
		numberOfNan[i]=np.sum(np.isnan(X_raw[i,:]))
		if numberOfNan[i]>=nan_max:
			if verbose:
				print("Too many NaN for obs ",i,". Removed")
			to_delete.append(i)
	if verbose:
		print("to_delete=",to_delete,". Total:",len(to_delete))
	X=np.delete(X_raw,to_delete,axis=0)
	
	if transpose:
		X=X.T
	
	if return_mask:
		delete_mask=np.full(N_raw,False,dtype=bool)
		delete_mask[to_delete]=True
		result= X,delete_mask
	else:
		result=X
	
	return result


def write_dataset(datasetpath,X_raw,t_common,z_common):
	'''Write the data prepared for the classification in a netcdf file
	with the grid on which it has been estimated.
	Dataset name must of the form:
		'DATASET_CAMPAGNE_PREDICTEURS_INTERPOLATION_dz***_dt***_zmax***.nc'
	
	[IN]
		- datasetpath (str): path and name of the netcdf file to be created.
		- X_raw (np.array[N,p]): data matrix (not normalised)
		- t_common (np.array[Nt] of datetime): time vector of the grid
		- z_common (np.array[Nz] of datetime): altitude vector of the grid
			Must have N=Nz*Nt
	
	[OUT]
		- msg (str): message saying the netcdf file has been successfully written.
	'''
	
	import time
	
	N,p=X_raw.shape
	if N!=len(t_common)*len(z_common):
		raise ValueError("Shapes of X_raw and grid do not match. Dataset NOT CREATED.")
	
	n_invalidValues = np.sum(np.isnan(X_raw))+np.sum(np.isinf(X_raw))
	if n_invalidValues>0:
		raise ValueError(n_invalidValues,"invalid values. Dataset NOT CREATED.")
	
	dataset = nc.Dataset(datasetpath,'w')
	
	# General information
	dataset.description="Dataset cleaned and prepared in order to make unsupervised boundary layer classification. The file is named according to the variables present in the dataset, their vertical and time resolution (all avariable are on the same grid) and the upper limit of the grid."
	dataset.source = 'Meteo-France CNRM/GMEI/LISA'
	dataset.history = 'Created '+time.ctime(time.time())
	dataset.contactperson = 'Thomas Rieutord (thomas.rieutord@meteo.fr)'
	
	
	# In[117]:
	
	# Coordinate declaration
	dataset.createDimension('individuals',N)
	dataset.createDimension('predictors',p)
	dataset.createDimension('time',len(t_common))
	dataset.createDimension('altitude',len(z_common))
	
	
	# Fill in altitude vector
	altitude = dataset.createVariable('altitude',np.float64, ('altitude',))
	altitude[:] = z_common
	altitude.units = "Meter above ground level (m)"
	
	# Fill in time vector
	Time = dataset.createVariable('time',np.float64, ('time',))
	Time[:] = dtlist2slist(t_common)
	Time.units = "Second since midnight (s)"
	
	# Fill in the design matrix
	designMatrix = dataset.createVariable('X_raw',np.float64, ('individuals','predictors'))
	designMatrix[:,:] = X_raw
	designMatrix.units = "Different for each column. Adimensionalisation is necessary before comparing columns."
	
	# Closing the netcdf file
	dataset.close()
	
	return "Dataset sucessfully written in the file "+datasetpath

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
		- fig (matplolib.pyplot figure): 2-dimensional view of the clusters
		In X-axis is the first variable given
		In Y-axis is the second variable given
		Clusters are shown with differents colors and marks.'''
	
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
		zoneID-=np.min(zoneID)
	
	# Number of clusters
	K=int(np.max(zoneID)+1)
	
	if clustersIDs is None:
		clustersIDs=np.arange(K)+1
	
	#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	fig=plt.figure()
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
	
	return fig

def cluster2Dview_multi(variable1,varname1,variable2,varname2,zoneIDs,figTitle="Clusters in feature space",storeImages=False,fileName=None,fmtImages=".svg",figureDir=""):
	'''Plots the projection of the clusters onto the space generated by
	two predictors. It can be used to visualize clusters (boundary layer classification).
	
	[IN]
		- variable1 (np.array[N]): first variable (vector of values, regardless with their coordinates)
		- varname1 (str): standard name of first variable
		- variable2 (np.array[N]): second variable (vector of values, regardless with their coordinates)
		- varname2 (str): standard name of second variable
		- zoneIDs (list of np.array[N]): cluster labels for each point
		- storeImages (opt, bool): if True, the figures are saved in the figureDir directory. Default is False
		- fmtImages (opt, str): format under which figures are saved when storeImages=True. Default is .svg
		- figureDir (opt, str): directory in which figures are saved when storeImages=True. Default is current directory.
		
	[OUT]
		- fig (matplolib.pyplot figure): 2-dimensional view of the clusters
		In X-axis is the first variable given
		In Y-axis is the second variable given
		Clusters are shown with differents colors and marks.'''
	
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
	
	n_kvalues = len(zoneIDs)
	nl=int(np.sqrt(n_kvalues))
	nc=int(np.ceil(n_kvalues/nl))
	
	#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	fig, axes = plt.subplots(nrows=nl, ncols=nc, figsize=(12, 8),sharex=True,sharey=True)
	plt.suptitle(figTitle)
	
	for ink in range(n_kvalues):
		zoneID = zoneIDs[ink]
	
		if np.min(zoneID)!=0:
			zoneID-=np.min(zoneID)
		
		# Number of clusters
		K=int(np.max(zoneID)+1)
		
		clustersIDs=np.arange(K)+1
		
		plt.subplot(nl,nc,ink+1)
		plt.title(str(K)+" clusters")
		plt.plot(variable1,variable2,'k.')
		for k in range(K):
			plt.plot(variable1[np.where(zoneID==k)],variable2[np.where(zoneID==k)],clusterMarks[clustersIDs[k]],linewidth=2,label="Cluster "+str(clustersIDs[k]))
	
		if np.mod(ink,nc)==0:
			plt.ylabel(DicLeg[varname2])
		if ink>=(nl-1)*nc:
			plt.xlabel(DicLeg[varname1])
		
	plt.tight_layout()
	#plt.show(block=False)
	if storeImages:
		if fileName is None:
			fileName="multi_cluster2Dview_"+varname1+'-'+varname2+fmtImages
		plt.savefig(figureDir+fileName)
	#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	
	return fig

def clusterZTview(t_values,z_values,zoneID,delete_mask=None,clustersIDs=None,storeImages=False,fmtImages=".svg",figureDir=""):
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
		- fig (matplolib.pyplot figure): clusters labels on a time-altitude grid
		In X-axis is the time
		In Y-axis is the height (m agl)
		Clusters are shown with differents colors.'''
	
	from matplotlib.colors import ListedColormap
	
	clusterMarks={1:'bo',2:'gx',3:'r^',4:'cv',5:'ys',6:'m*',7:'kp',8:'gd',9:'bx',10:'ro',11:'c*',12:'y+',13:'m<',14:'k,',
		'SBL':'bo',
		'FA':'gx',
		'ML':'r^',
		'EZ':'cv',
		'CP':'ys'}
	
	if np.min(zoneID)!=0:
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
	
	# 1. Deleted labels completion (when missing data)
	if delete_mask is not None:
		fullzoneID=np.full(np.size(delete_mask),np.nan)
		fullzoneID[~delete_mask]=zoneID
	else:
		fullzoneID=zoneID
	
	# 2. Conversion datetime -> seconds
	t0=t_values[0]
	st_values = dtlist2slist(t_values)
	
	# 3. Format from grid(z,t) to scatter
	TZ = grid_to_scatter(st_values,z_values)
	
	# 4. Set labels at grid(z,t) format
	t_trash,z_trash,labels = scatter_to_grid(TZ,fullzoneID)
	if np.max(np.abs(z_values-z_trash))+np.max(np.abs(st_values-t_trash))>1e-13:
		raise Exception("Error in z,t retrieval : max(|z_values-z_trash|)=",np.max(np.abs(z_values-z_trash)),"max(|t_values-t_trash|)=",np.max(np.abs(st_values-t_trash)))
	
	labels=np.ma.array(labels,mask=np.isnan(labels))
	
	# 5. Graphic
	#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	fig=plt.figure()
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
		plt.savefig(figureDir+"clusterZTview_K"+str(K)+fmtImages)
	#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	
	return fig

def clusterZTview_multi(t_values,z_values,zoneIDs,delete_mask=None,figTitle="Clusters in time-altitude grid",storeImages=False,fileName=None,fmtImages=".svg",figureDir=""):
	'''Plots cluster labels (areas) against time and height (boundary layer classification).
	
	[IN]
		- t_values (np.array[nt]): vector of time
		- z_values (np.array[nalt]): vector of altitude
		- zoneIDs (list of np.array[N]): cluster labels of each obs
		- delete_mask (np.array[nt*nalt]): mask at True when observation has been removed by the deletelines function (to avoid NaNs)
		- storeImages (opt, bool): if True, the figures are saved in the figureDir directory. Default is False
		- fmtImages (opt, str): format under which figures are saved when storeImages=True. Default is .svg
		- figureDir (opt, str): directory in which figures are saved when storeImages=True. Default is current directory.
		
	[OUT]
		- fig (matplolib.pyplot figure): clusters labels on a time-altitude grid
		In X-axis is the time
		In Y-axis is the height (m agl)
		Clusters are shown with differents colors.'''
	
	from matplotlib.colors import ListedColormap
	
	clusterMarks={1:'bo',2:'gx',3:'r^',4:'cv',5:'ys',6:'m*',7:'kp',8:'gd',9:'bx',10:'ro',11:'c*',12:'y+',13:'m<',14:'k,',
		'SBL':'bo',
		'FA':'gx',
		'ML':'r^',
		'EZ':'cv',
		'CP':'ys'}
	
	z_values=z_values/1000 		#convert meters to kilometers
	
	# 1. Conversion datetime -> seconds
	t0=t_values[0]
	st_values = dtlist2slist(t_values)
	
	# 2. Format from grid(z,t) to scatter
	TZ = grid_to_scatter(st_values,z_values)
	
	n_kvalues = len(zoneIDs)
	nl=int(np.sqrt(n_kvalues))
	nc=int(np.ceil(n_kvalues/nl))
	
	#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	fig, axes = plt.subplots(nrows=nl, ncols=nc, figsize=(12, 8),sharex=True,sharey=True)
	plt.suptitle(figTitle)
	for ink in range(n_kvalues):
		zoneID = zoneIDs[ink]
			
		if np.min(zoneID)!=0:
			zoneID-=np.min(zoneID)
		
		# Number of clusters
		K=int(np.max(zoneID)+1)
		
		clustersIDs=np.arange(K)+1
		
		clist = []
		cticks = []
		cticklabels = []
		for k in range(K):
			cticks.append(k+0.5)
			cticklabels.append(clustersIDs[k])
			clist.append(clusterMarks[clustersIDs[k]][0])
		colormap=ListedColormap(clist)
		
		# 3. Set labels at grid(z,t) format
		t_trash,z_trash,labels = scatter_to_grid(TZ,zoneID)
		if np.max(np.abs(z_values-z_trash))+np.max(np.abs(st_values-t_trash))>1e-13:
			raise Exception("Error in z,t retrieval : max(|z_values-z_trash|)=",np.max(np.abs(z_values-z_trash)),"max(|t_values-t_trash|)=",np.max(np.abs(st_values-t_trash)))
		
		labels=np.ma.array(labels,mask=np.isnan(labels))
		
		# 4. Graphic
		plt.subplot(nl,nc,ink+1)
		plt.title(str(K)+" clusters")
		im=plt.pcolormesh(t_values,z_values,labels.T,vmin=0,vmax=K,cmap=colormap)
		plt.gcf().autofmt_xdate()
		
		# Colorbar
		cbar=plt.colorbar()
		cbar.set_ticks(cticks)
		cbar.set_ticklabels(cticklabels)
		plt.tight_layout()
		
		if np.mod(ink,nc)==nl:
			cbar.set_label("Cluster labels")
		if np.mod(ink,nc)==0:
			plt.ylabel("Alt (km agl)")
		if ink>=(nl-1)*nc:
			plt.xlabel("Time (UTC)")
	
	
	fig.subplots_adjust(wspace=0,hspace=0.1)
	plt.tight_layout()
	#plt.show(block=False)
	if storeImages:
		if fileName is None:
			fileName="multi_clusterZTview"+fmtImages
		plt.savefig(figureDir+fileName)
	#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
	
	return fig
