#!/usr/bin/env python
# coding: utf-8
"""
IDENTIFY UNSUPERVISED BOUNDARY LAYER CLASSIFICATION
Notebook to identify boundary layer types corresponding to the labels given by unsupervised classification.
Take in input the dataset generated by `blclassification.py`.

This program requires USER INTERACTION.

 +-----------------------------------------+
 |  Date of creation: 03 Apr. 2020         |
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

graphics.storeImages=False

def isValidPath(rawpath):
    '''Check if the answer given by the user is correct.
    
    [IN]
        - rawpath (str): input provided by the user. It is supposed to be the path to unidentified clusters
    
    [OUT]
        - (bool): True if the path is correct
    '''
    
    result=False
    
    if os.path.isfile(rawpath):
        if rawpath[-3:].lower()=='.nc':
            try:
                utils.load_dataset(rawpath,variables_to_load=['rawlabels'])
                result=True
            except KeyError:
                print("Provided file has no field 'rawlabels'")
        else:
            print("Provided file is to a netCDF file")
    else:
        print("Provided path do not point to a file")
    
    return result

question1='''
+----------------------------------------------------------------------+
Welcome to the BL classification program!

Please type the relative path to the UNIDFLABELS*.nc file you want to process:
  * default (type Enter): 2-unidentified-labels/UNIDFLABELS_2015_0219.PASSY2015_BT-T_linear_dz40_dt30_zmax2000.nc
  * to abort, type "abort"
+----------------------------------------------------------------------+
Enter your path: 
'''
inp=input(question1)


if inp=="abort":
    exit("Aborted.")
elif inp=="":
    rawlabelspath="../working-directories/2-unidentified-labels/UNIDFLABELS_2015_0219.PASSY2015_BT-T_linear_dz40_dt30_zmax2000.nc"
else:
    if isValidPath(inp):
        rawlabelspath=inp
    else:
        raise ValueError("Incorrect typing (see last print). Please try again.")


X_raw,z_common,t_common,rawlabl = utils.load_dataset(rawlabelspath,variables_to_load=['X_raw','altitude','time','rawlabels'])
print("Data loaded correctly! Now preparing the graphics...")

# Vizualisation of clusters
# -------------------------

### Original data

origdataDir='../working-directories/0-original-data/'

rawlabelsname = rawlabelspath.split('/')[-1]
prefx,prepkey,dotnc = rawlabelsname.split('.')
predictors,interp,Dz,Dt,zmax=utils.load_preparation_params(prepkey)

for var in predictors:
    
    if var=='BT':
        origdatafile='CEILOMETER/PASSY_PASSY_CNRM_CEILOMETER_CT25K_'+prefx[-9:]+'_V01.nc'
    elif var=='T':
        origdatafile='MWR/PASSY2015_SALLANCHES_CNRM_MWR_HATPRO_'+prefx[-9:]+'_V01.nc'
    else:
        print("Unknown variable in predictors:",var,"Skipped.")
        continue
    
    try:
        graphics.quicklook(origdataDir+origdatafile,altmax=zmax)
    except FileNotFoundError:
        print("Original data not found at",origdataDir+origdatafile,"Skipped.")
        continue


### Unsupervised clustering results

graphics.cluster2Dview(X_raw[:,0],predictors[0],X_raw[:,1],predictors[1],rawlabl,displayClustersIDs=True)
graphics.clusterZTview(t_common,z_common,rawlabl,displayClustersIDs=True)


# Identify clusters
# ----------------

question2='''
+----------------------------------------------------------------------+
According to the previous graphs, please associate each cluster to a boundary layer type.

    Example of BL identification: 0:"CL",1:"SBL",2:"FA",3:"ML"
    ! Format matters: quotes "or' + no space + use ":" !
    * to abort, just type Enter
+----------------------------------------------------------------------+    
Enter your BL identification:
'''
inp=input(question2)

K=rawlabl.max()+1
if not((inp.count('"')==2*K or inp.count("'")==2*K) and inp.count(":")==K):
    raise ValueError("Invalid BL indentification. You must respect the format shown in example")

EXPERT_IDENTIFICATION=eval('{'+inp+'}')

boundary_layer_identified_types = {"CL":"Cloud",
                                  "SBL":"Stable boundary layer",
                                  "FA":"Free atmosphere",
                                  "ML":"Mixed layer"}

if not set(EXPERT_IDENTIFICATION.values()).issubset(boundary_layer_identified_types.keys()):
    print("boundary_layer_identified_types=",boundary_layer_identified_types)
    raise ValueError("Unknown boundary layer type. If it is a new BL type, please add it to the dict 'boundary_layer_identified_types'")


# Check assignment
# -----------------

graphics.cluster2Dview(X_raw[:,0],predictors[0],X_raw[:,1],predictors[1],rawlabl,
            clustersIDs=EXPERT_IDENTIFICATION,displayClustersIDs=True)
graphics.clusterZTview(t_common,z_common,rawlabl,
            clustersIDs=EXPERT_IDENTIFICATION,displayClustersIDs=True)


# Save the assignation
# -----------------------

outputDir="../working-directories/3-identified-labels/"
idfname="IDFLABELS_"+prefx[-9:]+"."+prepkey+".nc"

utils.add_idlabels_to_netcdf(rawlabelspath,outputDir+idfname,EXPERT_IDENTIFICATION,boundary_layer_identified_types,quiet=False)