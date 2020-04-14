#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MODULE FOR DATASET PREPARATION TO BOUNDARY LAYER CLASSIFICATION
Prepare dataset from ceilometer and radiometer data in order to apply
boundary layer classification. The point of this preparation is to
interpolate the atmospheric variables onto a common grid and save it in
a ready-to-use format for classification algorithms.

Functions are sorted in alphabetic order.

 +-----------------------------------------+
 |  Date of creation: 01 Apr. 2020         |
 +-----------------------------------------+
 |  Meteo-France                           |
 |  CNRM/GMEI/LISA                         |
 +-----------------------------------------+
 
Copyright Meteo-France, 2020, [CeCILL-C](https://cecill.info/licences.en.html) license (open source)

This module is a computer code that is part of the Boundary Layer
Classification program. This program performs atmospheric boundary layer
classification using machine learning algorithms.

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
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import time
import sys

from blcovid import utils
from blcovid import graphics

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


def estimateongrid(z_target,t_target,z_known,t_known,V_known,method='linear'):
    '''Interpolate the data on a target grid knowning it on another grid.
    Grids are time-altitude.
    
    Supported interpolation methods: 'linear','cubic','nearestneighbors'
    
    For nearest neighbors, the number of neighbors must be passed as the
    first character. For example: method='nearestneighbors'
    For more insights about how to choose the good methods (error, computing time...)
    please refer to the notebook `tuto-0to1-prepdataset.ipynb`
    
    [IN]
        - z_target (np.array[n1_z]): altitude vector of the target grid (m agl)
        - t_target (list[n1_t] of datetime): time vector of the target grid
        - z_known (np.array[n0_z]): altitude vector of the known grid (m agl)
        - t_known (list[n0_t] of datetime): time vector of the known grid
        - V_known (np.array[n0_t,n0_z]): data values on the known grid
        - method (str): interpolation method
        
    [OUT]
        - V_target (np.array[n1_t,n1_z]): valeurs sur la grilles cible'''
    
    
    # Switch from format "data=f(coordinates)" to format "obs=f(predictors)"
    st_known = utils.dtlist2slist(t_known)
    st_target = utils.dtlist2slist(t_target)
    X_known,Y_known=utils.grid_to_scatter(st_known,z_known,V_known)
    X_target=utils.grid_to_scatter(st_target,z_target)
    
    # NaN are removed
    X_known = X_known[~np.isnan(Y_known),:]
    Y_known = Y_known[~np.isnan(Y_known)]
    
    #### ========= Estimation with K-nearest neighbors
    if method[1:].lower()=="nearestneighbors":
        from sklearn.neighbors import KNeighborsRegressor
        KNN=KNeighborsRegressor(n_neighbors=int(method[0]))
        
        KNN.fit(X_known,Y_known)
        Y_target = KNN.predict(X_target)
        
    else:
    #### ========= Estimation with 2D interpolation
        from scipy.interpolate import griddata
        Y_target = griddata(X_known,Y_known,X_target,method=method.lower())
        
    # Shape the output
    t1,z1,V_target = utils.scatter_to_grid(X_target,Y_target)
    
    # Sanity checks
    if np.shape(V_target) != (np.size(st_target),np.size(z_target)):
        raise Exception("Output has not expected shape : shape(st_target)",np.shape(st_target),"shape(z_target)",np.shape(z_target),"shape(V_target)",np.shape(V_target))
    if (np.abs(t1-st_target)>10**(-10)).any():
        raise Exception("Time vector has been altered : max(|t1-t_target|)=",np.max(np.abs(t1-st_target)))
    if (np.abs(z1-z_target)>10**(-10)).any():
        raise Exception("Altitude vector has been altered : max(|z1-z_target|)=",np.max(np.abs(z1-z_target)))
    
    return V_target


def estimateInterpolationError(z_target,t_target,z_known,t_known,V_known,n_randoms=10,plot_on=True):
    '''Estimate the error and the computing time for several interpolation
    method.
    
    Errors are estimated by cross-validation. The function repeats the
    interpolation with all methods for severals train/test splits.
    The list of tested methods as well as their parameters must be
    changed inside the function.
    
    Default list: '4NearestNeighbors','8NearestNeighbors','linear','cubic'
    
    [IN]
        - z_target (np.array[n1_z]): altitude vector of the target grid (m agl)
        - t_target (list[n1_t] of datetime): time vector of the target grid
        - z_known (np.array[n0_z]): altitude vector of the known grid (m agl)
        - t_known (list[n0_t] of datetime): time vector of the known grid
        - V_known (np.array[n0_t,n0_z]): data values on the known grid
        - method (str): interpolation method
        
    [OUT]
        - V_target (np.array[n1_t,n1_z]): valeurs sur la grilles cible'''
    
    from sklearn.neighbors import KNeighborsRegressor
    from scipy.interpolate import griddata
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    
    # Switch from format "data=f(coordinates)" to format "obs=f(predictors)"
    st_known = utils.dtlist2slist(t_known)
    st_target = utils.dtlist2slist(t_target)
    X_known,Y_known=utils.grid_to_scatter(st_known,z_known,V_known)
    X_target=utils.grid_to_scatter(st_target,z_target)
    
    # NaN are removed
    X_known = X_known[~np.isnan(Y_known),:]
    Y_known = Y_known[~np.isnan(Y_known)]
    
    regressors = []
    reg_names = []
    
    #### ========= Estimation with 4-nearest neighbors
    KNN4=KNeighborsRegressor(n_neighbors=4)
    regressors.append(KNN4)
    reg_names.append("4NearestNeighbors")
    
    #### ========= Estimation with 8-nearest neighbors
    KNN8=KNeighborsRegressor(n_neighbors=8)
    regressors.append(KNN8)
    reg_names.append("8NearestNeighbors")
    
    chronos = np.zeros((len(regressors)+2,n_randoms))
    accuracies = np.zeros((len(regressors)+2,n_randoms))
    for icl in range(len(regressors)):
        reg = regressors[icl]
        print("Testing ",str(reg).split('(')[0])
        for ird in range(n_randoms):
            X_train, X_test, y_train, y_test = train_test_split(X_known,Y_known,test_size=0.2, random_state=ird)
            t0=time.time()      #::::::
            reg.fit(X_train,y_train)
            accuracies[icl,ird]=reg.score(X_test,y_test)
            t1=time.time()      #::::::
            chronos[icl,ird]=t1-t0
    
    #### ========= Estimation with 2D linear interpolation
    reg_names.append('Linear2DInterp')
    print("Testing Linear2DInterp")
    for ird in range(n_randoms):
        X_train, X_test, y_train, y_test = train_test_split(X_known,Y_known,test_size=0.2, random_state=ird)
        y_pred = griddata(X_train,y_train,X_test,method='linear')
        # Some data can still be missing even after the interpolation
        #   * Radiometer : resolution coarsens with altitude => last gates missing
        #   * Ceilometer : high lowest range => first gates missing
        y_test = y_test[~np.isnan(y_pred)]
        y_pred = y_pred[~np.isnan(y_pred)]
        accuracies[-2,ird]=r2_score(y_test,y_pred)
        t1=time.time()      #::::::
        chronos[-2,ird]=t1-t0
    
    
    #### ========= Estimation with 2D cubic interpolation
    reg_names.append('Cubic2DInterp')
    print("Testing Cubic2DInterp")
    for ird in range(n_randoms):
        X_train, X_test, y_train, y_test = train_test_split(X_known,Y_known,test_size=0.2, random_state=ird)
        y_pred = griddata(X_train,y_train,X_test,method='linear')
        # Some data can still be missing even after the interpolation
        #   * Radiometer : resolution coarsens with altitude => last gates missing
        #   * Ceilometer : high lowest range => first gates missing
        y_test = y_test[~np.isnan(y_pred)]
        y_pred = y_pred[~np.isnan(y_pred)]
        accuracies[-1,ird]=r2_score(y_test,y_pred)
        t1=time.time()      #::::::
        chronos[-1,ird]=t1-t0
    
    if plot_on:
        graphics.estimator_quality(accuracies,chronos,reg_names)
        
    return accuracies,chronos,reg_names


def generategrid(datemin,datemax,altmax,Dz,Dt,altmin=0):
    ''' Generate a time-altitude grid at the given resolution.
    
    [IN]
        - datemin (datetime): starting time of the grid
        - datemax (datetime): ending time of the grid
        - altmax (float): maximum altitude of the grid (m agl).
        - Dz (float): spatial resolution (m).
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
    
    z_values = np.arange(altmin,altmax,Dz)
    
    t_values = []
    for k in range(n_t):
        t_values.append(datemin+k*td)
    
    return z_values,t_values


def prepdataset(CEI_file,MWR_file, outputDir="../working-directories/1-unlabelled-datasets/",
                predictors = ['BT','T'], interpMethod='linear',
                z_max=2000, dt_common=30, dz_common=40,
                verbose=False, saveNetcdf=False,plot_on=False):
    '''Dataset preparation main function.
    
    Create a dataset ready-to-use for classification algorithms from
    original measurement data. This preparation implies an interpolation
    on a target grid.
    
    [IN]
        - CEI_file (str): path to the ceilometer measurements
        - MWR_file (str): path to the micro-wave radiometer measurements
        - outputDir (str): directory where the dataset will be stored
        - predictors (list of str): list of variables to be put in the dataset
        - interpMethod (str): name of the interpolation method used to estimate the values on the target grid
        - z_max (int): top altitude of the target grid
        - dt_common (int): time resolution of the target grid (minutes)
        - dz_common (int): vertical resolution of the target grid (m)
        - verbose (bool): if True, returns extended print outputs
        - saveNetcdf (bool): if False, the prepared dataset is not saved
        - plot_on (bool): if False, all graphics are disabled
    
    [OUT] Create a netCDF file containing the prepared dataset
        - (str): path to the created dataset
    '''
    
    if utils.scandate(CEI_file)!=utils.scandate(MWR_file):
        raise ValueError("Original data file do not have the same date")
    
    # setup toolbar
    toolbar_width = 7
    sys.stdout.write("Prep. data: [%s]" % ("." * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
    
    # EXTRACTION
    #============
    
    # Raw data
    #----------
    t_cei,z_cei,backscatter=utils.extractOrigData(CEI_file,altmax=z_max)
    sys.stdout.write("*")
    sys.stdout.flush()
    t_mwr,z_mwr,temperature=utils.extractOrigData(MWR_file,altmax=z_max)
    sys.stdout.write("*")
    sys.stdout.flush()
    # NB: here the backscatter signal is raw, it is NOT in decibel.
    
    # Dismiss obvious outliers
    #--------------------------
    
    # Negative backscatter are outliers
    with np.errstate(invalid='ignore'):
        backscatter[backscatter<=0]=np.nan
    
    # Convert backscatter to decibel
    backscatter=10*np.log10(backscatter)
    
    # Sanity checks
    #---------------
    if verbose:
        print('\n--- CEILOMETER ---')
        print("Size=",np.size(backscatter))
        print("Percentage of NaN=",100*np.sum(np.isnan(backscatter))/np.size(backscatter),"%")
        with np.errstate(invalid='ignore'):
            print("Percentage of negative values=",100*np.sum(backscatter<0)/np.size(backscatter),"%")
        print("VALUES : min=",np.nanmin(backscatter),"max=",np.nanmax(backscatter),"mean=",np.nanmean(backscatter),"median=",np.nanmedian(backscatter))
        print("GRID : dt=",np.mean(np.diff(t_cei)),"dz=",np.mean(np.diff(z_cei)),"Nt=",len(t_cei),"Nz=",len(z_cei),"data shape=",np.shape(backscatter))
        print('------')
        print('\n--- RADIOMETER ---')
        print("Size=",np.size(temperature))
        print("Percentage of NaN=",100*np.sum(np.isnan(temperature))/np.size(temperature),"%")
        print("VALUES : min=",np.nanmin(temperature),"max=",np.nanmax(temperature),"mean=",np.nanmean(temperature),"median=",np.nanmedian(temperature))
        print("GRID : dt=",np.mean(np.diff(t_mwr)),"dz=",np.mean(np.diff(z_mwr)),"Nt=",len(t_mwr),"Nz=",len(z_mwr),"data shape=",np.shape(temperature))
        print('------')
    
    if plot_on:
        graphics.quicklook(CEI_file,altmax=z_max)
        graphics.quicklook(MWR_file,altmax=z_max)
    
    
    sys.stdout.write("*")
    sys.stdout.flush()
    
    # INTERPOLATION
    #===============
    
    
    # Arrival grid
    #--------------
    day=utils.scandate(CEI_file)
    
    if dz_common=='MWR':
        # Autre alternative : prendre une grille existante.
        z_common,t_common=z_mwr,t_mwr
    else:
        t_min=day
        t_max=day+dt.timedelta(days=1)
        z_min=max(z_cei.min(),z_mwr.min())
        z_common,t_common=generategrid(t_min,t_max,z_max,dz_common,dt_common,z_min)
    
    sys.stdout.write("*")
    sys.stdout.flush()
    
    # Interpolation
    #---------------
    
    if verbose:
        print("\nInvalid values after interpolation:")
    # Radiometer
    if dz_common=='MWR' and np.sum(np.isnan(temperature[:,1:]))==0:
        T_query=temperature
    else:
        T_query=estimateongrid(z_common,t_common,z_mwr,t_mwr,temperature,method=interpMethod)
    sys.stdout.write("*")
    sys.stdout.flush()
    
    if verbose:
        with np.errstate(invalid='ignore'):
            print(" - T_query - #NaN=",np.sum(np.isnan(T_query)),"#Inf=",np.sum(np.isinf(T_query)),"#Neg=",np.sum(T_query<=0))
    
    # Ceilometer
    BT_query=estimateongrid(z_common,t_common,z_cei,t_cei,backscatter,method=interpMethod)
    sys.stdout.write("*")
    sys.stdout.flush()
    
    if verbose:
        with np.errstate(invalid='ignore'):
            print(" - BT_query - #NaN=",np.sum(np.isnan(BT_query)),"#Inf=",np.sum(np.isinf(BT_query)),"#Neg=",np.sum(BT_query<=0))
    
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
    
    # Add altitude in predictors (optional)
    if 'Z' in predictors:
        X_raw = np.array([BT.ravel(),T.ravel(),np.tile(z_common,(T.shape[0],1)).ravel()]).T
    else:
        X_raw = np.array([BT.ravel(),T.ravel()]).T
    
    # Sanity checks
    if verbose:
        print("\nInvalid values after removing bad boundaries:")
        with np.errstate(invalid='ignore'):
            print(" - X_raw - #NaN=",np.sum(np.isnan(X_raw)),"#Inf=",np.sum(np.isinf(X_raw)),"#Neg=",np.sum(X_raw<=0))
        print("shape(X_raw)=",X_raw.shape)
    
    sys.stdout.write("*")
    sys.stdout.flush()
    
    # Write dataset in netcdf file
    #------------------------------
    # The matrix X_raw is stored with the names of its columns and the grid on which is has been estimated.
    # Naming convention: DATASET_yyyy_mmdd.prepkey.nc
    
    prepkey='_'.join(['PASSY2015','-'.join(predictors),interpMethod,'dz'+str(dz_common),'dt'+str(dt_common),'zmax'+str(z_max)])
    yyyy=day.strftime('%Y')
    mmdd=day.strftime('%m%d')
    datasetname='DATASET_'+yyyy+'_'+mmdd+'.'+prepkey+'.nc'
    
    n_invalidValues = np.sum(np.isnan(X_raw))+np.sum(np.isinf(X_raw))
    
    sys.stdout.write("\n")
    sys.stdout.flush()
    
    if saveNetcdf and n_invalidValues==0:
        msg=write_dataset(outputDir+datasetname,X_raw,t_common,z_common)
        print(msg)
    else:
        print("No netcdf file produced!! saveNetcdf=",saveNetcdf,"n_invalidValues=",n_invalidValues)
    
    return outputDir+datasetname


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
            Dimensions must fulfil N=Nz*Nt
    
    [OUT]
        - msg (str): message saying the netcdf file has been successfully written.
    '''
    import netCDF4 as nc
    
    N,p=X_raw.shape
    if N!=len(t_common)*len(z_common):
        raise ValueError("Shapes of X_raw and grid do not match. Dataset NOT CREATED.")
    
    n_invalidValues = np.sum(np.isnan(X_raw))+np.sum(np.isinf(X_raw))
    if n_invalidValues>0:
        raise ValueError(n_invalidValues,"invalid values. Dataset NOT CREATED.")
    
    # print("datasetpath=",datasetpath)
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
    Time[:] = utils.dtlist2slist(t_common)
    Time.units = "Second since midnight (s)"
    
    # Fill in the design matrix
    designMatrix = dataset.createVariable('X_raw',np.float64, ('individuals','predictors'))
    designMatrix[:,:] = X_raw
    designMatrix.units = "Different for each column. Adimensionalisation is necessary before comparing columns."
    
    # Closing the netcdf file
    dataset.close()
    
    return "Dataset sucessfully written in the file "+datasetpath


########################
#      TEST BENCH      #
########################
# Launch with
# >> python prepdataset.py
#
# For interactive mode
# >> python -i prepdataset.py
#
if __name__ == '__main__':
    
    outputDir="../working-directories/1-unlabelled-datasets/"
    saveNetcdf=True
    graphics.figureDir = outputDir
    graphics.storeImages=True
    
    # Test of generategrid
    #------------------------
    print("\n --------------- Test of generategrid")
    dt_common = 30			# Time resolution common grid
    dz_common = 40			# Vertical resolution common grid
    z_max=2000						# Maximum altitude (m agl)
    day=dt.datetime(2015,2,19)
    z_common,t_common=generategrid(day,day+dt.timedelta(days=1),z_max,dz_common,dt_common,altmin=0)
    print("Shape check: t_common",len(t_common),'z_common',z_common.shape,'#points in grid:',len(t_common)*z_common.size)
    
    
    # Test of deletelines
    #------------------------
    print("\n --------------- Test of deletelines")
    
    X=np.arange(32,dtype='float').reshape((8,4))
    X[6,:]=np.nan
    X[1,2]=np.nan
    X[7,0]=np.nan
    print("Input: X=",X)
    X_del=deletelines(X)
    print("Output: X=",X_del)
    
    # Test of estimateongrid
    #------------------------
    print("\n --------------- Test of estimateongrid")
    
    dataDir = "../working-directories/0-original-data/"
    CEI_file = dataDir+"CEILOMETER/PASSY_PASSY_CNRM_CEILOMETER_CT25K_2015_0219_V01.nc"
    MWR_file = dataDir+"MWR/PASSY2015_SALLANCHES_CNRM_MWR_HATPRO_2015_0219_V01.nc"

    t_cei,z_cei,backscatter=utils.extractOrigData(CEI_file,altmax=z_max)
    # NB: here the backscatter signal is raw, it is NOT in decibel.
    
    # Negative backscatter are outliers
    with np.errstate(invalid='ignore'):
        backscatter[backscatter<=0]=np.nan

    t_mwr,z_mwr,temperature=utils.extractOrigData(MWR_file,altmax=z_max)
    
    for interpMethod in ['linear','cubic','4nearestneighbors']:
        print('\n'+interpMethod.upper())
        T_query=estimateongrid(z_common,t_common,z_mwr,t_mwr,temperature,method=interpMethod)
        print(" - T_query - #NaN=",np.sum(np.isnan(T_query)),"#Inf=",np.sum(np.isinf(T_query)),"#Neg=",np.sum(T_query<=0),'shape',T_query.shape)
        BT_query=estimateongrid(z_common,t_common,z_cei,t_cei,backscatter,method=interpMethod)
        print(" - BT_query - #NaN=",np.sum(np.isnan(BT_query)),"#Inf=",np.sum(np.isinf(BT_query)),"#Neg=",np.sum(BT_query<=0),'shape',BT_query.shape)
    
    
    # Test of estimateInterpolationError
    #------------------------
    print("\n --------------- Test of estimateInterpolationError")
    acc,tic,rn=estimateInterpolationError(z_common,t_common,z_mwr,t_mwr,temperature,n_randoms=10)
    
    
    # Test of prepdataset
    #------------------------
    print("\n --------------- Test of prepdataset")
    dataDir = "../working-directories/0-original-data/"
    CEI_file = dataDir+"CEILOMETER/PASSY_PASSY_CNRM_CEILOMETER_CT25K_2015_0219_V01.nc"
    MWR_file = dataDir+"MWR/PASSY2015_SALLANCHES_CNRM_MWR_HATPRO_2015_0219_V01.nc"
    prepkey=prepdataset(CEI_file, MWR_file, outputDir=outputDir,
                        plot_on=True, verbose=False,saveNetcdf=saveNetcdf)
    print("Done: ",prepkey)


