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
"""

import os
import numpy as np
import datetime as dt
import pickle


from blusc import utils
from blusc import graphics
from blusc.prepdataset import prepdataset
from blusc.unsupervised import ublc, ublc_manyclusters
from blusc.supervisedpred import predict_sblc, sblc_evaluation


def scandate_inv(
    day,
    campaign="PASSY",
    site="PASSY",
    lab="CNRM",
    techno="CEILOMETER",
    instru="CT25K",
    v01="V01.nc",
):
    """ Give the name of the file corresponding to the measurement at the given
    date (specific to the convention used in Passy-2015 data).
    
    
    Parameters
    ----------
    day : datetime.datetime
        Date of the file
        
        
    Returns
    -------
    filename: str
        Name of the file
    """
    yyyy = day.strftime("%Y")
    mmdd = day.strftime("%m%d")
    return "_".join([campaign, site, lab, techno, instru, yyyy, mmdd, v01])


def check_availability(CEI_dir, MWR_dir):
    """Check the date of all the files in the directory and return the 
    list of days when they are both available.
    
    
    Parameters
    ----------
    CEI_dir: str
        Path to the directory where ceilometer files are stored.
    
    MWR_dir: str
        Path to the directory where radiometer files are stored.
        
    Returns
    -------
    daybothavail: list, dtype=`datetime.datetime`
        Days where both instruments are available
    """
    import os

    ls_mwr = os.listdir(MWR_dir)
    ls_mwr.sort()
    ls_cei = os.listdir(CEI_dir)
    ls_cei.sort()

    t_min = min([utils.scandate(ls_cei[0]), utils.scandate(ls_mwr[0])])
    t_max = max([utils.scandate(ls_cei[-1]), utils.scandate(ls_mwr[-1])])

    daybothavail = []
    for d in range(int((t_max - t_min).total_seconds() / (3600 * 24))):
        day = t_min + dt.timedelta(days=d)
        f_CEI = scandate_inv(day)
        f_MWR = scandate_inv(
            day, campaign="PASSY2015", site="SALLANCHES", techno="MWR", instru="HATPRO"
        )
        if (f_CEI in ls_cei) and (f_MWR in ls_mwr):
            daybothavail.append(day)

    return daybothavail


def unsupervised_path(
    CEI_file,
    MWR_file,
    algo="hierarchical-average.euclidean",
    predictors=["BT", "T"],
    interpMethod="linear",
    z_max=2000,
    dt_common=30,
    dz_common=40,
    target_nb_clusters=4,
    outputDir="../working-directories/6-output-multidays/",
    forcePrepdataset=False,
):
    """Perform unsupervised classification from original data.
    
    Concatenate the preparation of the dataset (module `prepdataset.py`) and
    the unsupervised classification (module `unsupervised.py`).
    
    
    Parameters
    ----------
    CEI_file: str
        Path to the ceilometer measurements
    
    MWR_file: str
        Path to the micro-wave radiometer measurements
    
    algo: str
        Identifier of the classification algorithm
        Must be of the form algoFamily-param1.param2.param3...
    
    predictors: list, dtype=str
        Atmospheric variables to be put in the dataset
    
    interpMethod: str
        Name of the interpolation method used to estimate the values on the target grid
    
    z_max: int
        Top altitude of the target grid (meter above ground level)
    
    dt_common: int
        Time resolution of the target grid (minutes)
    
    dz_common: int
        Vertical resolution of the target grid (m)
    
    target_nb_clusters: {int, str}, default=4
        Number of desired clusters.
        If 'auto' (or any non-integer), several number of clusters are tested
    
    outputDir: str
        Directory where the dataset will be stored
    
    forcePrepdataset: bool, default=False
        If False, the preparation step is skipped when the required file
        already exists
    
    
    Returns
    -------
    Graphics created in `outputDir`
    
    
    Examples
    --------
    >>> from blusc.multidays import unsupervised_path
    >>> CEI_dir = "../working-directories/0-original-data/CEILOMETER/"
    >>> MWR_dir = "../working-directories/0-original-data/MWR/"
    >>> CEI_file = CEI_dir+"PASSY_PASSY_CNRM_CEILOMETER_CT25K_2015_0219_V01.nc"
    >>> MWR_file = MWR_dir+"PASSY2015_SALLANCHES_CNRM_MWR_HATPRO_2015_0219_V01.nc"
    >>> unsupervised_path(CEI_file, MWR_file, algo="hierarchical-average.euclidean", target_nb_clusters=4)
    Prep. data: [*******]
    Unsupervised classification results available in: ../working-directories/6-output-multidays/
    """

    if utils.scandate(CEI_file) != utils.scandate(MWR_file):
        raise ValueError("Original data file do not have the same date")

    day = utils.scandate(CEI_file)
    prepkey = "_".join(
        [
            "PASSY2015",
            "-".join(predictors),
            interpMethod,
            "dz" + str(dz_common),
            "dt" + str(dt_common),
            "zmax" + str(z_max),
        ]
    )
    datasetpath = (
        "../working-directories/1-unlabelled-datasets/DATASET_"
        + day.strftime("%Y_%m%d")
        + "."
        + prepkey
        + ".nc"
    )

    if os.path.isfile(datasetpath) and not forcePrepdataset:
        print("Prep. data: already existing", datasetpath)
    else:
        datasetpath = prepdataset(
            CEI_file,
            MWR_file,
            saveNetcdf=True,
            plot_on=True,
            predictors=predictors,
            interpMethod=interpMethod,
            z_max=z_max,
            dt_common=dt_common,
            dz_common=dz_common,
        )

    if isinstance(target_nb_clusters, int):
        ublc(
            datasetpath, algo=algo, target_nb_clusters=target_nb_clusters, plot_on=True
        )
    else:
        ublc_manyclusters(datasetpath, algo=algo, plot_on=True)

    print("Unsupervised classification results available in:", outputDir)


def supervised_path(
    CEI_file,
    MWR_file,
    classifierpath=None,
    algo="LabelSpreading",
    predictors=["BT", "T"],
    interpMethod="linear",
    z_max=2000,
    dt_common=30,
    dz_common=40,
    outputDir="../working-directories/6-output-multidays/",
    forcePrepdataset=False,
    plot_on=True,
):
    """Perform unsupervised classification from original data
    
    Concatenate the preparation of the dataset (module `prepdataset.py`) and
    the supervised classification (module `supervisedpred.py`).
    
    
    Parameters
    ----------
    CEI_file: str
        Path to the ceilometer measurements
    
    MWR_file: str
        Path to the micro-wave radiometer measurements
    
    classifierpath: str, optional
        Path where is located the classifier (.pkl). If provided, the argument
        `algo` is not used
    
    algo: str, optional
        Name of the supervised algorithm to use. Possible choices:
        {RandomForestClassifier, KNeighborsClassifier, DecisionTreeClassifier,
        AdaBoostClassifier, LabelSpreading}
        If `classifierpath` is provided, the algorithm is deducted from it
    
    predictors: list, dtype=str
        Atmospheric variables to be put in the dataset
    
    interpMethod: str
        Name of the interpolation method used to estimate the values on the target grid
    
    z_max: int
        Top altitude of the target grid (meter above ground level)
    
    dt_common: int
        Time resolution of the target grid (minutes)
    
    dz_common: int
        Vertical resolution of the target grid (m)
    
    target_nb_clusters: {int, str}, default=4
        Number of desired clusters.
        If 'auto' (or any non-integer), several number of clusters are tested
    
    outputDir: str
        Directory where the dataset will be stored
    
    forcePrepdataset: bool, default=False
        If False, the preparation step is skipped when the required file
        already exists
    
    plot_on: bool, default=False
        If False, all graphics are disabled
    
    
    Returns
    -------
    Graphics created in `outputDir`
    
    
    Examples
    --------
    >>> from blusc.multidays import supervised_path
    >>> CEI_dir = "../working-directories/0-original-data/CEILOMETER/"
    >>> MWR_dir = "../working-directories/0-original-data/MWR/"
    >>> CEI_file = CEI_dir+"PASSY_PASSY_CNRM_CEILOMETER_CT25K_2015_0219_V01.nc"
    >>> MWR_file = MWR_dir+"PASSY2015_SALLANCHES_CNRM_MWR_HATPRO_2015_0219_V01.nc"
    >>> supervised_path(CEI_file, MWR_file, algo="LabelSpreading")
    Prep. data: [*******]
    Supervised classification results available in: ../working-directories/6-output-multidays/
    """

    if utils.scandate(CEI_file) != utils.scandate(MWR_file):
        raise ValueError("Original data file do not have the same date")

    prepkey = "_".join(
        [
            "PASSY2015",
            "-".join(predictors),
            interpMethod,
            "dz" + str(dz_common),
            "dt" + str(dt_common),
            "zmax" + str(z_max),
        ]
    )

    if classifierpath is None:
        classifierpath = "../working-directories/4-pre-trained-classifiers/" + ".".join(
            [algo, prepkey, "pkl"]
        )

    if not os.path.isfile(classifierpath):
        raise ValueError("Classifier does not exist:", classifierpath)

    classifiername = classifierpath.split("/")[-1]
    prefx, prepkey2, dotnc = classifiername.split(".")
    if prepkey != prepkey2:
        raise ValueError(
            "Specified classifier do not fit with the preparation parameters"
        )

    day = utils.scandate(CEI_file)
    datasetpath = (
        "../working-directories/1-unlabelled-datasets/DATASET_"
        + day.strftime("%Y_%m%d")
        + "."
        + prepkey
        + ".nc"
    )
    print("datasetpath=", datasetpath)

    if os.path.isfile(datasetpath) and not forcePrepdataset:
        print("Prep. data: already existing", datasetpath)
    else:
        datasetpath = prepdataset(
            CEI_file,
            MWR_file,
            saveNetcdf=True,
            plot_on=True,
            predictors=predictors,
            interpMethod=interpMethod,
            z_max=z_max,
            dt_common=dt_common,
            dz_common=dz_common,
        )

    md2traincc, avgprb, labelid = sblc_evaluation(datasetpath, classifierpath, plot_on=plot_on)

    print("Supervised classification results available in:", outputDir)
    
    return  md2traincc, avgprb, labelid


def plotloop_unsupervised_path(
    CEI_dir,
    MWR_dir,
    algo="hierarchical-average.euclidean",
    predictors=["BT", "T"],
    interpMethod="linear",
    z_max=2000,
    dt_common=30,
    dz_common=40,
    outputDir="../working-directories/6-output-multidays/",
    forcePrepdataset=False,
):
    """Repeat the function `unsupervised_path` over all the days when both
    instruments are available in order to produce graphics showing the results
    of the unsupervised classification for each day
    
    
    Parameters
    ----------
    CEI_file: str
        Path to the ceilometer measurements
    
    MWR_file: str
        Path to the micro-wave radiometer measurements
    
    All other outputs are the same as for `unsupervised_path` function. Please refer to its docstring.
    
    
    Returns
    -------
    Graphics created in `outputDir`
    
    
    Examples
    --------
    >>> from blusc.multidays import loop_unsupervised_path
    >>> CEI_dir = "../working-directories/0-original-data/CEILOMETER/"
    >>> MWR_dir = "../working-directories/0-original-data/MWR/"
    >>> plotloop_unsupervised_path(CEI_dir, MWR_dir)
    -- Day 2015-02-10 00:00:00 ( 1 / 9 )
    Prep. data: [*******]
    Unsupervised classification results available in: ../working-directories/6-output-multidays/20150210/
    ...
    End of loop. Crashes: []
    """

    daybothavail = check_availability(CEI_dir, MWR_dir)

    print("UNSUPERVISED LOOP")
    crashing = []
    i = 0
    for day in daybothavail:
        i += 1
        print("\n-- Day", day, "(", i, "/", len(daybothavail), ")")
        CEI_file = scandate_inv(day)
        MWR_file = scandate_inv(
            day, campaign="PASSY2015", site="SALLANCHES", techno="MWR", instru="HATPRO"
        )

        dailydir = outputDir + day.strftime("%Y%m%d") + "/"
        if not os.path.isdir(dailydir):
            os.mkdir(dailydir)
        try:
            graphics.figureDir = dailydir
            unsupervised_path(
                CEI_dir + CEI_file,
                MWR_dir + MWR_file,
                outputDir=dailydir,
                algo=algo,
                predictors=predictors,
                interpMethod=interpMethod,
                z_max=z_max,
                dt_common=dt_common,
                dz_common=dz_common,
                forcePrepdataset=forcePrepdataset,
            )
        except FileNotFoundError:
            print("Error in the preparation of data")
            crashing.append(day)

    print("End of loop. Crashes:", crashing)


def plotloop_supervised_path(
    CEI_dir,
    MWR_dir,
    classifierpath=None,
    algo="LabelSpreading",
    predictors=["BT", "T"],
    interpMethod="linear",
    z_max=2000,
    dt_common=30,
    dz_common=40,
    outputDir="../working-directories/6-output-multidays/",
    forcePrepdataset=False,
):
    """Repeat the function `supervised_path` over all the days when both
    instruments are available in order to produce graphics showing the results
    of the supervised classification for each day.
    
    
    Parameters
    ----------
    CEI_file: str
        Path to the ceilometer measurements
    
    MWR_file: str
        Path to the micro-wave radiometer measurements
    
    All other outputs are the same as for `supervised_path` function. Please refer to its docstring.
    
    
    Returns
    -------
    Graphics created in `outputDir`
    
    
    Examples
    --------
    >>> from blusc.multidays import plotloop_supervised_path
    >>> CEI_dir = "../working-directories/0-original-data/CEILOMETER/"
    >>> MWR_dir = "../working-directories/0-original-data/MWR/"
    >>> plotloop_supervised_path(CEI_dir, MWR_dir)
    -- Day 2015-02-10 00:00:00 ( 1 / 9 )
    Prep. data: [*******]
    Supervised classification results available in: ../working-directories/6-output-multidays/20150210/
    ...
    End of loop. Crashes: []
    """

    daybothavail = check_availability(CEI_dir, MWR_dir)

    print("SUPERVISED LOOP")
    crashing = []
    i = 0
    for day in daybothavail:
        i += 1
        print("\n-- Day", day, "(", i, "/", len(daybothavail), ")")
        CEI_file = scandate_inv(day)
        MWR_file = scandate_inv(
            day, campaign="PASSY2015", site="SALLANCHES", techno="MWR", instru="HATPRO"
        )

        dailydir = outputDir + day.strftime("%Y%m%d") + "/"
        if not os.path.isdir(dailydir):
            os.mkdir(dailydir)

        try:
            graphics.figureDir = dailydir
            supervised_path(
                CEI_dir + CEI_file,
                MWR_dir + MWR_file,
                outputDir=dailydir,
                algo=algo,
                classifierpath=classifierpath,
                predictors=predictors,
                interpMethod=interpMethod,
                z_max=z_max,
                dt_common=dt_common,
                dz_common=dz_common,
                forcePrepdataset=forcePrepdataset,
                plot_on=True,
            )
        except FileNotFoundError:
            print("Error in the preparation of data")
            crashing.append(day)

    print("End of loop. Crashes:", crashing)


def evalloop_supervised_path(
    CEI_dir,
    MWR_dir,
    classifierpath=None,
    algo="LabelSpreading",
    predictors=["BT", "T"],
    interpMethod="linear",
    z_max=2000,
    dt_common=30,
    dz_common=40,
    outputDir="../working-directories/6-output-multidays/",
    forcePrepdataset=False,
    plot_on=True,
):
    """Repeat the function `supervised_path` over all the days when both
    instruments are available in order to produce graphics showing the results
    of the supervised classification for each day.
    
    
    Parameters
    ----------
    CEI_file: str
        Path to the ceilometer measurements
    
    MWR_file: str
        Path to the micro-wave radiometer measurements
    
    All other outputs are the same as for `supervised_path` function. Please refer to its docstring.
    
    
    Returns
    -------
    Graphics created in `outputDir`
    
    
    Examples
    --------
    >>> from blusc.multidays import evalloop_supervised_path
    >>> CEI_dir = "../working-directories/0-original-data/CEILOMETER/"
    >>> MWR_dir = "../working-directories/0-original-data/MWR/"
    >>> evalloop_supervised_path(CEI_dir, MWR_dir)
    -- Day 2015-02-10 00:00:00 ( 1 / 9 )
    Prep. data: [*******]
    Supervised classification results available in: ../working-directories/6-output-multidays/20150210/
    ...
    End of loop. Crashes: []
    """
    
    # Get number of classes and label identification
    #------------------
    prepkey = "_".join(
        [
            "PASSY2015",
            "-".join(predictors),
            interpMethod,
            "dz" + str(dz_common),
            "dt" + str(dt_common),
            "zmax" + str(z_max),
        ]
    )

    if classifierpath is None:
        classifierpath = "../working-directories/4-pre-trained-classifiers/" + ".".join(
            [algo, prepkey, "pkl"]
        )

    if not os.path.isfile(classifierpath):
        raise ValueError("Classifier does not exist:", classifierpath)
    
    fc = open(classifierpath, "rb")
    clf = pickle.load(fc)
    ref_labelid = eval(clf.label_identification_)
    nb_classes = clf.classes_.size
    
    # Get days with both instruments
    #------------------
    daybothavail = check_availability(CEI_dir, MWR_dir)

    # Start the loop
    #------------------
    print("SUPERVISED LOOP")
    
    crashing = []
    avgprb_perday = np.zeros((len(daybothavail),nb_classes))
    md2traincc_perday = np.zeros((len(daybothavail),nb_classes))
    i = 0
    for day in daybothavail:
        i += 1
        print("\n-- Day", day, "(", i, "/", len(daybothavail), ")")
        
        # ### Retrieve paths for the current day
        CEI_file = scandate_inv(day)
        MWR_file = scandate_inv(
            day, campaign="PASSY2015", site="SALLANCHES", techno="MWR", instru="HATPRO"
        )

        dailydir = outputDir + day.strftime("%Y%m%d") + "/"
        if not os.path.isdir(dailydir):
            os.mkdir(dailydir)
        
        # print(
                # CEI_dir + CEI_file,
                # MWR_dir + MWR_file,
                # "outputDir=",dailydir,
                # "algo=",algo,
                # "classifierpath=",classifierpath,
                # "predictors=",predictors,
                # "interpMethod=",interpMethod,
                # "z_max=",z_max,
                # "dt_common=",dt_common,
                # "dz_common=",dz_common,
                # "forcePrepdataset=",forcePrepdataset,
                # "plot_on=",False,
            # )
        
        # ### Perform classification
        try:
            md2traincc, avgprb, labelid = supervised_path(
                CEI_dir + CEI_file,
                MWR_dir + MWR_file,
                outputDir=dailydir,
                algo=algo,
                classifierpath=classifierpath,
                predictors=predictors,
                interpMethod=interpMethod,
                z_max=z_max,
                dt_common=dt_common,
                dz_common=dz_common,
                forcePrepdataset=forcePrepdataset,
                plot_on=False,
            )
        except FileNotFoundError:
            print("Error in the preparation of data")
            crashing.append(day)
            md2traincc = np.full(nb_classes,np.nan)
            avgprb = np.full(nb_classes,np.nan)
            labelid = ref_labelid
        
        # ### Check and save the results
        if labelid!=ref_labelid:
            raise Exception("Unexpected change in label IDs")
        
        avgprb_perday[i-1,:] = avgprb
        md2traincc_perday[i-1,:] = md2traincc
    
    # Display the results
    #--------------------
    if plot_on:
        graphics.agreement_with_training(daybothavail, md2traincc_perday, ref_labelid)
        
    print("End of loop. Crashes:", crashing)
    
    return daybothavail, md2traincc_perday, avgprb_perday, ref_labelid

########################
#      TEST BENCH      #
########################
# Launch with
# >> python multidays.py
#
# For interactive mode
# >> python -i multidays.py
#
if __name__ == "__main__":

    outputDir = "../working-directories/6-output-multidays/"
    graphics.storeImages = True
    graphics.figureDir = outputDir

    CEI_dir = "../working-directories/0-original-data/CEILOMETER/"
    MWR_dir = "../working-directories/0-original-data/MWR/"

    # Test of check_availability
    # ------------------------
    print("\n --------------- Test of check_availability")
    daybothavail = check_availability(CEI_dir, MWR_dir)
    print(daybothavail, "Total:", len(daybothavail), "days")

    # Test of unsupervised_path
    # ------------------------
    print("\n --------------- Test of unsupervised_path")

    CEI_file = CEI_dir + "PASSY_PASSY_CNRM_CEILOMETER_CT25K_2015_0217_V01.nc"
    MWR_file = MWR_dir + "PASSY2015_SALLANCHES_CNRM_MWR_HATPRO_2015_0217_V01.nc"

    unsupervised_path(CEI_file, MWR_file, outputDir=outputDir)

    # Test of supervised_path
    # ------------------------
    print("\n --------------- Test of supervised_path")

    CEI_file = CEI_dir + "PASSY_PASSY_CNRM_CEILOMETER_CT25K_2015_0217_V01.nc"
    MWR_file = MWR_dir + "PASSY2015_SALLANCHES_CNRM_MWR_HATPRO_2015_0217_V01.nc"

    supervised_path(CEI_file, MWR_file, outputDir=outputDir)

    # Test of plotloop_unsupervised_path
    # ------------------------
    print("\n --------------- Test of plotloop_unsupervised_path")
    plotloop_unsupervised_path(CEI_dir, MWR_dir, outputDir=outputDir)

    # Test of plotloop_supervised_path
    # ------------------------
    print("\n --------------- Test of plotloop_supervised_path")
    plotloop_supervised_path(CEI_dir, MWR_dir, outputDir=outputDir)
