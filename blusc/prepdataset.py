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
"""

import numpy as np
import datetime as dt
import time
import sys

from blusc import utils
from blusc import graphics


def deletelines(X_raw, nan_max=None, return_mask=False, transpose=False, verbose=False):
    """Remove lines containing many Not-a-Number values from the dataset.
    
    All lines containing nan_max missing values or more are also removed.
    By default, nan_max=p-1 (lines with only NaN or only but 1 are removed)
    
    
    Parameters
    ----------
    X_raw: ndarray of shape (N_raw,p): 
        Original dataset.
    
    nan_max: int, default=p-1
        Maximum number of missing value tolerated
    
    return_mask: bool, default=False
        If True, returns the mask at True for deleted lines
    
    transpose: bool, defalut=False
        If True, removes the columns instead of the lines
    
    verbose: bool, default=False
        If True, does more prints
    
    
    Returns
    -------
    X: ndarray of shape (N,p)
        Filtered dataset (N<=N_raw)
    
    delete_mask: ndarray of shape (N_raw,), dtype=bool
        Mask at True for deleted lines. Only returned if return_mask=True
    """

    if transpose:
        X_raw = X_raw.T

    N_raw, p = np.shape(X_raw)
    if nan_max is None:
        nan_max = p - 1

    if verbose:
        print("Delete all lines with ", nan_max, "missing values or more.")
    to_delete = []
    numberOfNan = np.zeros(N_raw)
    for i in range(N_raw):
        numberOfNan[i] = np.sum(np.isnan(X_raw[i, :]))
        if numberOfNan[i] >= nan_max:
            if verbose:
                print("Too many NaN for obs ", i, ". Removed")
            to_delete.append(i)
    if verbose:
        print("to_delete=", to_delete, ". Total:", len(to_delete))
    X = np.delete(X_raw, to_delete, axis=0)

    if transpose:
        X = X.T

    if return_mask:
        delete_mask = np.full(N_raw, False, dtype=bool)
        delete_mask[to_delete] = True
        result = X, delete_mask
    else:
        result = X

    return result


def estimateongrid(z_target, t_target, z_known, t_known, V_known, method="linear"):
    """Interpolate the data on a target grid knowning it on another grid.
    Grids are time-altitude.
    
    Supported interpolation methods: 'linear','cubic','nearestneighbors'
    
    For nearest neighbors, the number of neighbors must be passed as the
    first character. For example: method='4nearestneighbors'
    For more insights about how to choose the good methods (error, computing time...)
    please refer to the notebook `tuto-0to1-prepdataset.ipynb`
    
    
    Parameters
    ----------
    z_target: ndarray of shape (n1_z,)
        Altitude vector of the target grid (m agl)
    
    t_target: array-like of shape (n1_t,) with dtype=datetime
        Time vector of the target grid
    
    z_known: ndarray of shape (n0_z,)
        Altitude vector of the known grid (m agl)
    
    t_known: array-like of shape (n0_t,) with dtype=datetime
        Time vector of the known grid
    
    V_known: ndarray of shape (n0_t,n0_z)
        Data values on the known grid
    
    method: {'linear','cubic','nearestneighbors'}, default='linear'
        Interpolation method.
        
        
    Returns
    -------
    V_target: ndarray of shape (n1_t,n1_z)
        Values on the target grid
    """

    # Switch from format "data=f(coordinates)" to format "obs=f(predictors)"
    st_known = utils.dtlist2slist(t_known)
    st_target = utils.dtlist2slist(t_target)
    X_known, Y_known = utils.grid_to_scatter(st_known, z_known, V_known)
    X_target = utils.grid_to_scatter(st_target, z_target)

    # NaN are removed
    X_known = X_known[~np.isnan(Y_known), :]
    Y_known = Y_known[~np.isnan(Y_known)]

    #### ========= Estimation with K-nearest neighbors
    if method[1:].lower() == "nearestneighbors":
        from sklearn.neighbors import KNeighborsRegressor

        KNN = KNeighborsRegressor(n_neighbors=int(method[0]))

        KNN.fit(X_known, Y_known)
        Y_target = KNN.predict(X_target)

    else:
        #### ========= Estimation with 2D interpolation
        from scipy.interpolate import griddata

        Y_target = griddata(X_known, Y_known, X_target, method=method.lower())

    # Shape the output
    t1, z1, V_target = utils.scatter_to_grid(X_target, Y_target)

    # Sanity checks
    if np.shape(V_target) != (np.size(st_target), np.size(z_target)):
        raise Exception(
            "Output has not expected shape : shape(st_target)",
            np.shape(st_target),
            "shape(z_target)",
            np.shape(z_target),
            "shape(V_target)",
            np.shape(V_target),
        )
    if (np.abs(t1 - st_target) > 10 ** (-10)).any():
        raise Exception(
            "Time vector has been altered : max(|t1-t_target|)=",
            np.max(np.abs(t1 - st_target)),
        )
    if (np.abs(z1 - z_target) > 10 ** (-10)).any():
        raise Exception(
            "Altitude vector has been altered : max(|z1-z_target|)=",
            np.max(np.abs(z1 - z_target)),
        )

    return V_target


def estimateInterpolationError(
    z_target, t_target, z_known, t_known, V_known, n_randoms=10, plot_on=True
):
    """Estimate the error and the computing time for several interpolation
    method.
    
    Errors are estimated by cross-validation. The function repeats the
    interpolation with all methods for severals train/test splits.
    The list of tested methods as well as their parameters must be
    changed inside the function.
    
    Default list: '4NearestNeighbors','8NearestNeighbors','linear','cubic'
    
    
    Parameters
    ----------
    z_target: ndarray of shape (n1_z,)
        Altitude vector of the target grid (m agl)
    
    t_target: array-like of shape (n1_t,) with dtype=datetime
        Time vector of the target grid
    
    z_known: ndarray of shape (n0_z,)
        Altitude vector of the known grid (m agl)
    
    t_known: array-like of shape (n0_t,) with dtype=datetime
        Time vector of the known grid
    
    V_known: ndarray of shape (n0_t,n0_z)
        Data values on the known grid
    
    n_randoms: int, default=10
        Number of repeated random split between training and testing sets
        
    plot_on: bool, default=True
        If True, the graphics showing computing time versus accuracy is drawn
        
        
    Returns
    -------
    accuracies: ndarray of shape (n_randoms,n_regressors)
        R2 score of each regressor (one per line) for each random split (one per
        column).
        
    chronos: ndarray of shape (n_randoms,n_regressors)
        Computing time of each regressor (one per line) for each random split
        (one per column).
    
    reg_names: list of shape (n_regressors,)
        Names of regressions methods performed
    """

    from sklearn.neighbors import KNeighborsRegressor
    from scipy.interpolate import griddata
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    # Switch from format "data=f(coordinates)" to format "obs=f(predictors)"
    st_known = utils.dtlist2slist(t_known)
    st_target = utils.dtlist2slist(t_target)
    X_known, Y_known = utils.grid_to_scatter(st_known, z_known, V_known)
    X_target = utils.grid_to_scatter(st_target, z_target)

    # NaN are removed
    X_known = X_known[~np.isnan(Y_known), :]
    Y_known = Y_known[~np.isnan(Y_known)]

    regressors = []
    reg_names = []

    #### ========= Estimation with 4-nearest neighbors
    KNN4 = KNeighborsRegressor(n_neighbors=4)
    regressors.append(KNN4)
    reg_names.append("4NearestNeighbors")

    #### ========= Estimation with 8-nearest neighbors
    KNN8 = KNeighborsRegressor(n_neighbors=8)
    regressors.append(KNN8)
    reg_names.append("8NearestNeighbors")

    chronos = np.zeros((len(regressors) + 2, n_randoms))
    accuracies = np.zeros((len(regressors) + 2, n_randoms))
    for icl in range(len(regressors)):
        reg = regressors[icl]
        print("Testing ", str(reg).split("(")[0])
        for ird in range(n_randoms):
            X_train, X_test, y_train, y_test = train_test_split(
                X_known, Y_known, test_size=0.2, random_state=ird
            )
            t0 = time.time()  #::::::
            reg.fit(X_train, y_train)
            accuracies[icl, ird] = reg.score(X_test, y_test)
            t1 = time.time()  #::::::
            chronos[icl, ird] = t1 - t0

    #### ========= Estimation with 2D linear interpolation
    reg_names.append("Linear2DInterp")
    print("Testing Linear2DInterp")
    for ird in range(n_randoms):
        X_train, X_test, y_train, y_test = train_test_split(
            X_known, Y_known, test_size=0.2, random_state=ird
        )
        y_pred = griddata(X_train, y_train, X_test, method="linear")
        # Some data can still be missing even after the interpolation
        #   * Radiometer : resolution coarsens with altitude => last gates missing
        #   * Ceilometer : high lowest range => first gates missing
        y_test = y_test[~np.isnan(y_pred)]
        y_pred = y_pred[~np.isnan(y_pred)]
        accuracies[-2, ird] = r2_score(y_test, y_pred)
        t1 = time.time()  #::::::
        chronos[-2, ird] = t1 - t0

    #### ========= Estimation with 2D cubic interpolation
    reg_names.append("Cubic2DInterp")
    print("Testing Cubic2DInterp")
    for ird in range(n_randoms):
        X_train, X_test, y_train, y_test = train_test_split(
            X_known, Y_known, test_size=0.2, random_state=ird
        )
        y_pred = griddata(X_train, y_train, X_test, method="linear")
        # Some data can still be missing even after the interpolation
        #   * Radiometer : resolution coarsens with altitude => last gates missing
        #   * Ceilometer : high lowest range => first gates missing
        y_test = y_test[~np.isnan(y_pred)]
        y_pred = y_pred[~np.isnan(y_pred)]
        accuracies[-1, ird] = r2_score(y_test, y_pred)
        t1 = time.time()  #::::::
        chronos[-1, ird] = t1 - t0

    if plot_on:
        graphics.estimator_quality(accuracies, chronos, reg_names)

    return accuracies, chronos, reg_names


def generategrid(datemin, datemax, altmax, Dz, Dt, altmin=0):
    """ Generate a time-altitude grid at the given resolution.
    
    
    Parameters
    ----------
    datemin: datetime.datetime
        Starting time of the grid
    
    datemax: datetime.datetime
        Ending time of the grid
    
    altmax: {int,float}
        Maximum altitude of the grid (m agl).
    
    Dz: {int,float}
        Spatial resolution (m).
    
    Dt: {float, datetime.timedelta
        Time resolution (minutes).
    
    altmin: {int,float}, default=0
        Minimum altitude of the grid (m agl)
        
        
    Returns
    -------
    z_values: ndarray of shape (n_z,)
        Altitude vector of the grid (m agl)
    
    t_values: list of length n_t with dtype=datetime.datetime
        Time vector of the grid
    """
    if isinstance(datemax, dt.timedelta):
        datefin = datemin + datemax
    else:
        datefin = datemax

    if isinstance(Dt, dt.timedelta):
        td = Dt
    else:
        td = dt.timedelta(minutes=Dt)

    n_t = int((datefin - datemin).total_seconds() / td.total_seconds())

    z_values = np.arange(altmin, altmax, Dz)

    t_values = []
    for k in range(n_t):
        t_values.append(datemin + k * td)

    return z_values, t_values


def prepdataset(
    CEI_file,
    MWR_file,
    outputDir="../working-directories/1-unlabelled-datasets/",
    predictors=["BT", "T"],
    interpMethod="linear",
    z_max=2000,
    dt_common=30,
    dz_common=40,
    verbose=False,
    saveNetcdf=False,
    plot_on=False,
):
    """Dataset preparation main function.
    
    Create a dataset ready-to-use for classification algorithms from
    original measurement data. This preparation implies an interpolation
    on a target grid.
    
    
    Parameters
    ----------
    CEI_file: str
        Path to the ceilometer measurements
    
    MWR_file: str
        Path to the micro-wave radiometer measurements
    
    outputDir: str
        Directory where the dataset will be stored
    
    predictors: list with dtype=str
        Atmospheric variables to be put in the dataset
        
    interpMethod: str
        Name of the interpolation method used to estimate the values on the target grid
    
    z_max: int
        Top altitude of the target grid (meter above ground level)
    
    dt_common: int
        Time resolution of the target grid (minutes)
    
    dz_common: int
        Vertical resolution of the target grid (m)
    
    verbose: bool, default=False
        If True, returns extended print outputs
    
    saveNetcdf: bool
        If False, the prepared dataset is not saved
    
    plot_on: bool
        If False, all graphics are disabled
    
    
    Returns
    -------
    Create a netCDF file containing the prepared dataset
    
    datasetpath: str
        Path to the created dataset
    
    
    Examples
    --------
    >>> from blusc.prepdataset import prepdataset
    >>> dataDir = "../working-directories/0-original-data/"
    >>> CEI_file = dataDir + "CEILOMETER/PASSY_PASSY_CNRM_CEILOMETER_CT25K_2015_0219_V01.nc"
    >>> MWR_file = dataDir + "MWR/PASSY2015_SALLANCHES_CNRM_MWR_HATPRO_2015_0219_V01.nc"
    >>> datasetpath = prepdataset(CEI_file, MWR_file, interpMethod="linear")
    Prep. data: [*******]
    No netcdf file produced!! saveNetcdf= False n_invalidValues= 0
    >>> datasetpath
    '../working-directories/1-unlabelled-datasets/DATASET_2015_0219.PASSY2015_BT-T_linear_dz40_dt30_zmax2000.nc'
    """

    if utils.scandate(CEI_file) != utils.scandate(MWR_file):
        raise ValueError("Original data file do not have the same date")

    # setup toolbar
    if verbose:
        print("Entering function ",__name__,"with 9 checkpoints")
    else:
        toolbar_width = 7
        sys.stdout.write("Prep. data: [%s]" % ("." * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['

    # EXTRACTION
    # ============

    # Raw data
    # ----------
    t_cei, z_cei, backscatter = utils.extractOrigData(CEI_file, altmax=z_max)
    if verbose:
        print("\n[1/9] Extraction of backscatter: DONE")
        print("backscatter.shape=",backscatter.shape)
    else:
        sys.stdout.write("*")
        sys.stdout.flush()
    
    t_mwr, z_mwr, temperature = utils.extractOrigData(MWR_file, altmax=z_max)
    if verbose:
        print("\n[2/9] Extraction of temperature: DONE")
        print("temperature.shape=",temperature.shape)
    else:
        sys.stdout.write("*")
        sys.stdout.flush()
    # NB: here the backscatter signal is raw, it is NOT in decibel.

    # Dismiss obvious outliers
    # --------------------------

    # Negative backscatter are outliers
    with np.errstate(invalid="ignore"):
        backscatter[backscatter <= 0] = np.nan

    # Convert backscatter to decibel
    backscatter = 10 * np.log10(backscatter)

    # Sanity checks
    # ---------------
    if verbose:
        print("\n[3/9] Sanity checks")
        print("--- CEILOMETER ---")
        print("Size=", np.size(backscatter))
        print(
            "Percentage of NaN=",
            100 * np.sum(np.isnan(backscatter)) / np.size(backscatter),
            "%",
        )
        with np.errstate(invalid="ignore"):
            print(
                "Percentage of negative values=",
                100 * np.sum(backscatter < 0) / np.size(backscatter),
                "%",
            )
        print(
            "VALUES : min=",
            np.nanmin(backscatter),
            "max=",
            np.nanmax(backscatter),
            "mean=",
            np.nanmean(backscatter),
            "median=",
            np.nanmedian(backscatter),
        )
        print(
            "GRID : dt=",
            np.mean(np.diff(t_cei)),
            "dz=",
            np.mean(np.diff(z_cei)),
            "Nt=",
            len(t_cei),
            "Nz=",
            len(z_cei),
            "data shape=",
            np.shape(backscatter),
        )
        print("------")
        print("--- RADIOMETER ---")
        print("Size=", np.size(temperature))
        print(
            "Percentage of NaN=",
            100 * np.sum(np.isnan(temperature)) / np.size(temperature),
            "%",
        )
        print(
            "VALUES : min=",
            np.nanmin(temperature),
            "max=",
            np.nanmax(temperature),
            "mean=",
            np.nanmean(temperature),
            "median=",
            np.nanmedian(temperature),
        )
        print(
            "GRID : dt=",
            np.mean(np.diff(t_mwr)),
            "dz=",
            np.mean(np.diff(z_mwr)),
            "Nt=",
            len(t_mwr),
            "Nz=",
            len(z_mwr),
            "data shape=",
            np.shape(temperature),
        )
        print("------")
    else:
        sys.stdout.write("*")
        sys.stdout.flush()

    if plot_on:
        graphics.quicklook(CEI_file, altmax=z_max)
        graphics.quicklook(MWR_file, altmax=z_max)


    # INTERPOLATION
    # ===============

    # Arrival grid
    # --------------
    day = utils.scandate(CEI_file)

    if dz_common == "MWR" or dt_common == "MWR":
        # Autre alternative : prendre une grille existante.
        z_common, t_common = z_mwr, t_mwr
    else:
        t_min = day
        t_max = day + dt.timedelta(days=1)
        z_min = max(z_cei.min(), z_mwr.min())
        z_common, t_common = generategrid(
            t_min, t_max, z_max, dz_common, dt_common, z_min
        )
    
    if verbose:
        print("\n[4/9] Grid generation: DONE")
        print("z_common.shape=",z_common.shape, "len(t_common)=",len(t_common))
    sys.stdout.write("*")
    sys.stdout.flush()

    # Interpolation
    # ---------------

    if verbose:
        print("\nInvalid values after interpolation:")
    # Radiometer
    if dz_common == "MWR" and np.sum(np.isnan(temperature[:, 1:])) == 0:
        T_query = temperature
    else:
        T_query = estimateongrid(
            z_common, t_common, z_mwr, t_mwr, temperature, method=interpMethod
        )
    if verbose:
        print("\n[5/9] Interpolation of temperature: DONE")
        print("T_query.shape=",T_query.shape)
        with np.errstate(invalid="ignore"):
            print(
                "#NaN=",
                np.sum(np.isnan(T_query)),
                "#Inf=",
                np.sum(np.isinf(T_query)),
                "#Neg=",
                np.sum(T_query <= 0),
            )
    else:
        sys.stdout.write("*")
        sys.stdout.flush()

    # Ceilometer
    BT_query = estimateongrid(
        z_common, t_common, z_cei, t_cei, backscatter, method=interpMethod
    )
    if verbose:
        print("\n[6/9] Interpolation of temperature: DONE")
        print("BT_query.shape=",BT_query.shape)
        with np.errstate(invalid="ignore"):
            print(
                "#NaN=",
                np.sum(np.isnan(BT_query)),
                "#Inf=",
                np.sum(np.isinf(BT_query)),
                "#Neg=",
                np.sum(BT_query <= 0),
            )
    else:
        sys.stdout.write("*")
        sys.stdout.flush()

    # Shape the data
    # ---------------
    # Some data can still be missing even after the interpolation
    #   * Radiometer : resolution coarsens with altitude => last gates missing
    #   * Ceilometer : high lowest range => first gates missing

    trash, mask_T = deletelines(
        T_query, nan_max=T_query.shape[0] - 1, return_mask=True, transpose=True
    )

    trash, mask_BT = deletelines(
        BT_query, nan_max=BT_query.shape[0] - 1, return_mask=True, transpose=True
    )

    z_common = z_common[~np.logical_or(mask_T, mask_BT)]
    T = T_query[:, ~np.logical_or(mask_T, mask_BT)]
    BT = BT_query[:, ~np.logical_or(mask_T, mask_BT)]

    # Concatenate predictors
    # ----------------------
    if verbose:
        print("\n[7/9] Concatenate predictors ",predictors)
    preX = []
    if "BT" in predictors:
        preX.append(BT.ravel())
        if verbose:
            print("Add BT")
    if "T" in predictors:
        preX.append(T.ravel())
        if verbose:
            print("Add T")
    if "Z" in predictors:
        preX.append(np.tile(z_common, (T.shape[0], 1)).ravel())
        if verbose:
            print("Add Z")
    
    X_raw = np.array(preX).T
    
    # # Add altitude in predictors (optional)
    # if "Z" in predictors:
        # X_raw = np.array(
            # [BT.ravel(), T.ravel(), np.tile(z_common, (T.shape[0], 1)).ravel()]
        # ).T
    # else:
        # X_raw = np.array([BT.ravel(), T.ravel()]).T

    # Sanity checks
    if verbose:
        print("\n[8/9] Final check")
        print("shape(X_raw)=", X_raw.shape)
        with np.errstate(invalid="ignore"):
            print(
                "#NaN=",
                np.sum(np.isnan(X_raw)),
                "#Inf=",
                np.sum(np.isinf(X_raw)),
                "#Neg=",
                np.sum(X_raw <= 0),
            )
    else:
        sys.stdout.write("*")
        sys.stdout.flush()
    
    # Write dataset in netcdf file
    # ------------------------------
    # The matrix X_raw is stored with the names of its columns and the grid on which is has been estimated.
    # Naming convention: DATASET_yyyy_mmdd.prepkey.nc

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
    yyyy = day.strftime("%Y")
    mmdd = day.strftime("%m%d")
    datasetname = "DATASET_" + yyyy + "_" + mmdd + "." + prepkey + ".nc"

    n_invalidValues = np.sum(np.isnan(X_raw)) + np.sum(np.isinf(X_raw))

    if verbose:
        print("\n[9/9] Exiting",__name__,"with")
        print("prepkey =",prepkey)
        print("n_invalidValues =",n_invalidValues)
    else:
        sys.stdout.write("\n")
        sys.stdout.flush()

    if saveNetcdf and n_invalidValues == 0:
        msg = write_dataset(outputDir + datasetname, X_raw, t_common, z_common)
        print(msg)
    else:
        print(
            "No netcdf file produced!! saveNetcdf=",
            saveNetcdf,
            "n_invalidValues=",
            n_invalidValues,
        )

    return outputDir + datasetname


def write_dataset(datasetpath, X_raw, t_common, z_common):
    """Write the data prepared for the classification in a netcdf file
    with the grid on which it has been estimated.
    Dataset name must of the form:
        'DATASET_CAMPAGNE_PREDICTEURS_INTERPOLATION_dz***_dt***_zmax***.nc'
    
    
    Parameters
    ----------
    datasetpath: str
        Path and name of the netcdf file to be created.
    
    X_raw: ndarray of shape (N,p)
        Data matrix (not normalised)
    
    t_common: array-like of shape (Nt,) with dtype=datetime.datetime
        Time vector of the grid
    
    z_common array-like of shape (Nz,):
        Altitude vector of the grid
    
    
    Returns
    -------
    msg: str
        Message saying the netcdf file has been successfully written
    """
    import netCDF4 as nc

    N, p = X_raw.shape
    if N != len(t_common) * len(z_common):
        raise ValueError("Shapes of X_raw and grid do not match. Dataset NOT CREATED.")

    n_invalidValues = np.sum(np.isnan(X_raw)) + np.sum(np.isinf(X_raw))
    if n_invalidValues > 0:
        raise ValueError(n_invalidValues, "invalid values. Dataset NOT CREATED.")

    # print("datasetpath=",datasetpath)
    dataset = nc.Dataset(datasetpath, "w")

    # General information
    dataset.description = "Dataset cleaned and prepared in order to make unsupervised boundary layer classification. The file is named according to the variables present in the dataset, their vertical and time resolution (all avariable are on the same grid) and the upper limit of the grid."
    dataset.source = "Meteo-France CNRM/GMEI/LISA"
    dataset.history = "Created " + time.ctime(time.time())
    dataset.contactperson = "Thomas Rieutord (thomas.rieutord@meteo.fr)"

    # In[117]:

    # Coordinate declaration
    dataset.createDimension("individuals", N)
    dataset.createDimension("predictors", p)
    dataset.createDimension("time", len(t_common))
    dataset.createDimension("altitude", len(z_common))

    # Fill in altitude vector
    altitude = dataset.createVariable("altitude", np.float64, ("altitude",))
    altitude[:] = z_common
    altitude.units = "Meter above ground level (m)"

    # Fill in time vector
    Time = dataset.createVariable("time", np.float64, ("time",))
    Time[:] = utils.dtlist2slist(t_common)
    Time.units = "Second since midnight (s)"

    # Fill in the design matrix
    designMatrix = dataset.createVariable(
        "X_raw", np.float64, ("individuals", "predictors")
    )
    designMatrix[:, :] = X_raw
    designMatrix.units = "Different for each column. Adimensionalisation is necessary before comparing columns."

    # Closing the netcdf file
    dataset.close()

    return "Dataset sucessfully written in the file " + datasetpath


########################
#      TEST BENCH      #
########################
# Launch with
# >> python prepdataset.py
#
# For interactive mode
# >> python -i prepdataset.py
#
if __name__ == "__main__":

    outputDir = "../working-directories/1-unlabelled-datasets/"
    saveNetcdf = True
    graphics.figureDir = outputDir
    graphics.storeImages = True

    # Test of generategrid
    # ------------------------
    print("\n --------------- Test of generategrid")
    dt_common = 30  # Time resolution common grid
    dz_common = 40  # Vertical resolution common grid
    z_max = 2000  # Maximum altitude (m agl)
    day = dt.datetime(2015, 2, 19)
    z_common, t_common = generategrid(
        day, day + dt.timedelta(days=1), z_max, dz_common, dt_common, altmin=0
    )
    print(
        "Shape check: t_common",
        len(t_common),
        "z_common",
        z_common.shape,
        "#points in grid:",
        len(t_common) * z_common.size,
    )

    # Test of deletelines
    # ------------------------
    print("\n --------------- Test of deletelines")

    X = np.arange(32, dtype="float").reshape((8, 4))
    X[6, :] = np.nan
    X[1, 2] = np.nan
    X[7, 0] = np.nan
    print("Input: X=", X)
    X_del = deletelines(X)
    print("Output: X=", X_del)

    # # Test of estimateongrid
    # #------------------------
    # print("\n --------------- Test of estimateongrid")

    # dataDir = "../working-directories/0-original-data/"
    # CEI_file = dataDir+"CEILOMETER/PASSY_PASSY_CNRM_CEILOMETER_CT25K_2015_0219_V01.nc"
    # MWR_file = dataDir+"MWR/PASSY2015_SALLANCHES_CNRM_MWR_HATPRO_2015_0219_V01.nc"

    # t_cei,z_cei,backscatter=utils.extractOrigData(CEI_file,altmax=z_max)
    # # NB: here the backscatter signal is raw, it is NOT in decibel.

    # # Negative backscatter are outliers
    # with np.errstate(invalid='ignore'):
    # backscatter[backscatter<=0]=np.nan

    # t_mwr,z_mwr,temperature=utils.extractOrigData(MWR_file,altmax=z_max)

    # for interpMethod in ['linear','cubic','4nearestneighbors']:
    # print('\n'+interpMethod.upper())
    # T_query=estimateongrid(z_common,t_common,z_mwr,t_mwr,temperature,method=interpMethod)
    # print(" - T_query - #NaN=",np.sum(np.isnan(T_query)),"#Inf=",np.sum(np.isinf(T_query)),"#Neg=",np.sum(T_query<=0),'shape',T_query.shape)
    # BT_query=estimateongrid(z_common,t_common,z_cei,t_cei,backscatter,method=interpMethod)
    # print(" - BT_query - #NaN=",np.sum(np.isnan(BT_query)),"#Inf=",np.sum(np.isinf(BT_query)),"#Neg=",np.sum(BT_query<=0),'shape',BT_query.shape)

    # # Test of estimateInterpolationError
    # #------------------------
    # print("\n --------------- Test of estimateInterpolationError")
    # acc,tic,rn=estimateInterpolationError(z_common,t_common,z_mwr,t_mwr,temperature,n_randoms=10)

    # Test of prepdataset
    # ------------------------
    print("\n --------------- Test of prepdataset")
    dataDir = "../working-directories/0-original-data/"
    CEI_file = dataDir + "CEILOMETER/PASSY_PASSY_CNRM_CEILOMETER_CT25K_2015_0219_V01.nc"
    MWR_file = dataDir + "MWR/PASSY2015_SALLANCHES_CNRM_MWR_HATPRO_2015_0219_V01.nc"
    prepkey = prepdataset(
        CEI_file,
        MWR_file,
        outputDir=outputDir,
        plot_on=True,
        verbose=False,
        saveNetcdf=saveNetcdf,
    )
    print("Done: ", prepkey)
