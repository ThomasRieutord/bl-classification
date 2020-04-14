#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Intented for Python 3.5

MODULE FOR PREDICTING BOUNDARY LAYER CLASSIFICATION FOR SEVERAL DAYS

Functions are sorted in complexity order:
    - scandate_inv
    - check_availability
    - unsupervised_path
    - supervised_path
    - loop_unsupervised_path
    - loop_supervised_path

 +-----------------------------------------+
 |  Date of creation: 09 Apr. 2020         |
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

import os
import numpy as np
import datetime as dt


from blcovid import utils
from blcovid import graphics
from blcovid.prepdataset import prepdataset
from blcovid.unsupervised import ublc,ublc_manyclusters
from blcovid.supervisedpred import predict_sblc

def scandate_inv(day,campaign='PASSY',site='PASSY',lab='CNRM',techno='CEILOMETER',instru='CT25K',v01='V01.nc'):
    ''' Give the name of the file corresponding to the measurement at the given date.
    
    [IN]
        - day (datetime): date of the file
        
    [OUT]
        - filename (str): name of the file
    '''
    yyyy=day.strftime('%Y')
    mmdd=day.strftime('%m%d')
    return '_'.join([campaign,site,lab,techno,instru,yyyy,mmdd,v01])


def check_availability(CEI_dir,MWR_dir):
    '''Check the date of all the files in the directory and return the 
    list of days when they are both available.
    
    [IN]
        - CEI_dir (str): path to the directory where ceilometer files are stored.
        - MWR_dir (str): path to the directory where radiometer files are stored.
        
    [OUT]
        - daybothavail (list of datetime): day where both instruments are available
    '''
    import os
    
    ls_mwr=os.listdir(MWR_dir)
    ls_mwr.sort()
    ls_cei=os.listdir(CEI_dir)
    ls_cei.sort()
    
    t_min=min([utils.scandate(ls_cei[0]),utils.scandate(ls_mwr[0])])
    t_max=max([utils.scandate(ls_cei[-1]),utils.scandate(ls_mwr[-1])])
    
    daybothavail=[]
    for d in range(int((t_max-t_min).total_seconds()/(3600*24))):
        day = t_min + dt.timedelta(days=d)
        f_CEI=scandate_inv(day)
        f_MWR=scandate_inv(day,campaign='PASSY2015',site='SALLANCHES',techno='MWR',instru='HATPRO')
        if (f_CEI in ls_cei) and (f_MWR in ls_mwr):
            daybothavail.append(day)
    
    return daybothavail


def unsupervised_path(CEI_file,MWR_file,algo="hierarchical-average.euclidean",
                predictors = ['BT','T'], interpMethod='linear',
                z_max=2000, dt_common=30, dz_common=40, target_nb_clusters=4,
                outputDir='../working-directories/6-output-multidays/',forcePrepdataset=False):
    '''Perform unsupervised classification from original data.
    
    Concatenate the preparation of the dataset (module `prepdataset.py`) and
    the unsupervised classification (module `unsupervised.py`).
    
    [IN]
        - CEI_file (str): path to the ceilometer measurements
        - MWR_file (str): path to the micro-wave radiometer measurements
        - algo (str): identifier of the classification algorithm
                Must be of the form algoFamily-param1.param2.param3...
        - predictors (list of str): list of variables to be put in the dataset
        - interpMethod (str): name of the interpolation method used to estimate the values on the target grid
        - z_max (int): top altitude of the target grid
        - dt_common (int): time resolution of the target grid (minutes)
        - dz_common (int): vertical resolution of the target grid (m)
        - target_nb_clusters (int or str): number of desired clusters. If 'auto' (or else non-integer) several number of clusters are tested
        - outputDir (str): directory where the dataset will be stored
        - forcePrepdataset (bool): if False, the preparation step is skipped when the required file already exists
    
    [OUT] Graphics created in outputDir
    '''
    
    if utils.scandate(CEI_file)!=utils.scandate(MWR_file):
        raise ValueError("Original data file do not have the same date")
    
    day=utils.scandate(CEI_file)
    prepkey = '_'.join(['PASSY2015','-'.join(predictors),interpMethod,'dz'+str(dz_common),'dt'+str(dt_common),'zmax'+str(z_max)])
    datasetpath='../working-directories/1-unlabelled-datasets/DATASET_'+day.strftime('%Y_%m%d')+'.'+prepkey+'.nc'
    
    if os.path.isfile(datasetpath) and not forcePrepdataset:
        print("Prep. data: already existing",datasetpath)
    else:
        datasetpath=prepdataset(CEI_file, MWR_file, saveNetcdf=True, plot_on=True,
                    predictors=predictors, interpMethod=interpMethod,
                    z_max=z_max, dt_common=dt_common, dz_common=dz_common)
    
    if isinstance(target_nb_clusters,int):
        ublc(datasetpath, algo=algo, target_nb_clusters=target_nb_clusters, plot_on=True)
    else:
        ublc_manyclusters(datasetpath, algo=algo, plot_on=True)
    
    print("Unsupervised classification results available in:",outputDir)


def supervised_path(CEI_file,MWR_file,classifierpath=None,algo='LabelSpreading',
                predictors = ['BT','T'], interpMethod='linear',
                z_max=2000, dt_common=30, dz_common=40,
                outputDir='../working-directories/6-output-multidays/',forcePrepdataset=False):
    '''Perform unsupervised classification from original data
    
    Concatenate the preparation of the dataset (module `prepdataset.py`) and
    the supervised classification (module `supervisedpred.py`).
    
    [IN]
        - CEI_file (str): path to the ceilometer measurements
        - MWR_file (str): path to the micro-wave radiometer measurements
        - classifierpath (str): identifier of the classification algorithm. If provided, the argument algo is not used
        - algo (str): algo (str): name of the supervised algorithm to use. Possible choices:
            RandomForestClassifier, KNeighborsClassifier, DecisionTreeClassifier, AdaBoostClassifier, LabelSpreading
            If classifierpath is provided, this argument is not used.
        - predictors (list of str): list of variables to be put in the dataset
        - interpMethod (str): name of the interpolation method used to estimate the values on the target grid
        - z_max (int): top altitude of the target grid
        - dt_common (int): time resolution of the target grid (minutes)
        - dz_common (int): vertical resolution of the target grid (m)
        - outputDir (str): directory where the dataset will be stored
        - forcePrepdataset (bool): if False, the preparation step is skipped when the required file already exists
    
    [OUT] Graphics created in outputDir
    '''
    
    if utils.scandate(CEI_file)!=utils.scandate(MWR_file):
        raise ValueError("Original data file do not have the same date")
    
    prepkey='_'.join(['PASSY2015','-'.join(predictors),interpMethod,'dz'+str(dz_common),'dt'+str(dt_common),'zmax'+str(z_max)])
    
    if classifierpath is None:
        classifierpath="../working-directories/4-pre-trained-classifiers/"+'.'.join([algo,prepkey,'pkl'])
    
    if not os.path.isfile(classifierpath):
        raise ValueError("Classifier does not exist:",classifierpath)
    
    classifiername = classifierpath.split('/')[-1]
    prefx,prepkey2,dotnc = classifiername.split('.')
    if prepkey!=prepkey2:
        raise ValueError("Specified classifier do not fit with the preparation parameters")
    
    day=utils.scandate(CEI_file)
    datasetpath='../working-directories/1-unlabelled-datasets/DATASET_'+day.strftime('%Y_%m%d')+'.'+prepkey+'.nc'
    print("datasetpath=",datasetpath)
    
    if os.path.isfile(datasetpath) and not forcePrepdataset:
        print("Prep. data: already existing",datasetpath)
    else:
        datasetpath=prepdataset(CEI_file, MWR_file, saveNetcdf=True, plot_on=True,
                    predictors=predictors, interpMethod=interpMethod,
                    z_max=z_max, dt_common=dt_common, dz_common=dz_common)
    
    predict_sblc(datasetpath, classifierpath,plot_on=True)
    
    print("Supervised classification results available in:",outputDir)


def loop_unsupervised_path(CEI_dir,MWR_dir,algo="hierarchical-average.euclidean",
                predictors = ['BT','T'], interpMethod='linear',
                z_max=2000, dt_common=30, dz_common=40,
                outputDir='../working-directories/6-output-multidays/',forcePrepdataset=False):
    '''Repeat the function `unsupervised_path` over all the days when both
    instruments are available.
    
    [IN]
        - CEI_dir (str): path to the directory with ceilometer measurements
        - MWR_dir (str): path to the directory with micro-wave radiometer measurements
        All other outputs are the same as for `unsupervised_path` function. Please refer to its docstring.
    
    [OUT] Graphics created in outputDir
    '''
    
    daybothavail = check_availability(CEI_dir,MWR_dir)
    
    print("UNSUPERVISED LOOP")
    crashing=[]
    i=0
    for day in daybothavail:
        i+=1
        print("\n-- Day",day,'(',i,'/',len(daybothavail),')')
        CEI_file = scandate_inv(day)
        MWR_file = scandate_inv(day,campaign='PASSY2015',site='SALLANCHES',techno='MWR',instru='HATPRO')
        
        dailydir=outputDir+day.strftime('%Y%m%d')+"/"
        if not os.path.isdir(dailydir):
            os.mkdir(dailydir)
        try:
            graphics.figureDir=dailydir
            unsupervised_path(CEI_dir+CEI_file, MWR_dir+MWR_file, outputDir=dailydir,
                    algo=algo, predictors=predictors, interpMethod=interpMethod,
                    z_max=z_max, dt_common=dt_common, dz_common=dz_common,
                    forcePrepdataset=forcePrepdataset)
        except FileNotFoundError:
            print('Error in the preparation of data')
            crashing.append(day)
    
    print("End of loop. Crashes:",crashing)


def loop_supervised_path(CEI_dir,MWR_dir,classifierpath=None,algo='LabelSpreading',
                predictors = ['BT','T'], interpMethod='linear',
                z_max=2000, dt_common=30, dz_common=40,
                outputDir='../working-directories/6-output-multidays/',forcePrepdataset=False):
    '''Repeat the function `supervised_path` over all the days when both
    instruments are available.
    
    [IN]
        - CEI_dir (str): path to the directory with ceilometer measurements
        - MWR_dir (str): path to the directory with micro-wave radiometer measurements
        All other outputs are the same as for `supervised_path` function. Please refer to its docstring.
    
    [OUT] Graphics created in outputDir
    '''
    
    daybothavail = check_availability(CEI_dir,MWR_dir)
    
    print("SUPERVISED LOOP")
    crashing=[]
    i=0
    for day in daybothavail:
        i+=1
        print("\n-- Day",day,'(',i,'/',len(daybothavail),')')
        CEI_file = scandate_inv(day)
        MWR_file = scandate_inv(day,campaign='PASSY2015',site='SALLANCHES',techno='MWR',instru='HATPRO')
        
        dailydir=outputDir+day.strftime('%Y%m%d')+"/"
        if not os.path.isdir(dailydir):
            os.mkdir(dailydir)
        
        try:
            graphics.figureDir=dailydir
            supervised_path(CEI_dir+CEI_file, MWR_dir+MWR_file, outputDir=dailydir,
                    algo=algo, classifierpath=classifierpath, predictors=predictors,
                    interpMethod=interpMethod, z_max=z_max, dt_common=dt_common,
                    dz_common=dz_common, forcePrepdataset=forcePrepdataset)
        except FileNotFoundError:
            print('Error in the preparation of data')
            crashing.append(day)
        
    print("End of loop. Crashes:",crashing)


########################
#      TEST BENCH      #
########################
# Launch with
# >> python multidays.py
#
# For interactive mode
# >> python -i multidays.py
#
if __name__ == '__main__':
    
    outputDir = "../working-directories/6-output-multidays/"
    graphics.storeImages=True
    graphics.figureDir=outputDir
    
    CEI_dir = "../working-directories/0-original-data/CEILOMETER/"
    MWR_dir = "../working-directories/0-original-data/MWR/"
    
    # Test of check_availability
    #------------------------
    print("\n --------------- Test of check_availability")
    daybothavail = check_availability(CEI_dir,MWR_dir)
    print(daybothavail,"Total:", len(daybothavail),"days")
    
    
    # Test of unsupervised_path
    #------------------------
    print("\n --------------- Test of unsupervised_path")
    
    CEI_file = CEI_dir+"PASSY_PASSY_CNRM_CEILOMETER_CT25K_2015_0217_V01.nc"
    MWR_file = MWR_dir+"PASSY2015_SALLANCHES_CNRM_MWR_HATPRO_2015_0217_V01.nc"
    
    unsupervised_path(CEI_file,MWR_file,outputDir=outputDir)
    
    # Test of supervised_path
    #------------------------
    print("\n --------------- Test of supervised_path")
    
    CEI_file = CEI_dir+"PASSY_PASSY_CNRM_CEILOMETER_CT25K_2015_0217_V01.nc"
    MWR_file = MWR_dir+"PASSY2015_SALLANCHES_CNRM_MWR_HATPRO_2015_0217_V01.nc"
    
    supervised_path(CEI_file,MWR_file,outputDir=outputDir)
    
    
    # Test of loop_unsupervised_path
    #------------------------
    print("\n --------------- Test of loop_unsupervised_path")
    loop_unsupervised_path(CEI_dir,MWR_dir,outputDir=outputDir)
    
    
    # Test of loop_supervised_path
    #------------------------
    print("\n --------------- Test of loop_supervised_path")
    loop_supervised_path(CEI_dir,MWR_dir,outputDir=outputDir)

