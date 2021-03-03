#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MODULE OF BASICS FUNCTIONS FOR BOUNDARY CLASSIFICATION.
Contains all functions that are common to more than one module (excepted
graphics that are in the `graphics.py` module)

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
import netCDF4 as nc
import matplotlib.pyplot as plt


def add_idlabels_to_netcdf(
    inputFile, outputFile, label_identification, label_long_names, quiet=False
):
    """Copy the netCDF file containing dataset and labels without
    identification and add their identification in two new fields.
    
    
    Parameters
    ----------
    inputFile: str
        Path to the input file (netCDF with dataset + labels WITHOUT identification)
    
    outputFile : str
        Path to the output file (to be written: netCDF with dataset + labels WITH identification)
    
    label_identification : dict
        Dictionary associating cluster labels to boundary layer types
        Example: {0:"CL",1:"SBL",2:"FA",3:"ML"}
    
    label_long_names : dict
        Dictionary associating short acronyms to their extended names
        Example: {"CL":"Cloud","SBL":"Stable boundary layer","FA":"Free atmosphere","ML":"Mixed layer"}
    
    quiet : bool, default=True
        If True, all prints are disabled.
    
    
    Returns
    -------
    Generate a netCDF file at the specified location
    
    
    Notes
    -----
    Called by blusc.blindentification
    """

    create_file_from_source(inputFile, outputFile)
    ncf = nc.Dataset(outputFile, "r+")

    ncf.label_identification = str(label_identification)
    ncf.label_long_names = str(label_long_names)

    ncf.close()
    if not quiet:
        print(
            "Identification of labels have been added in the netcdf file ", outputFile
        )


def add_rawlabels_to_netcdf(inputFile, outputFile, labels, quiet=False):
    """Copy the netCDF file containing dataset and add the labels
    resulting from unsupervised classification in a new variable.
    
    
    Parameters
    ----------
    inputFile : str
        Path to the input file (netCDF with dataset WITHOUT labels)
    
    outputFile : str
        Path to the output file (to be written: netCDF with dataset WITH labels)
    
    labels : ndarray of shape (N,) with dtype=int
        Label of each individual in the dataset. Numbers values have no meaning,
        they only serve to separate the groups.
    
    quiet : bool, default=False
        If True, all prints are disabled.
    
    
    Returns
    -------
    Generate a netCDF file at the specified location
    
    
    Notes
    -----
    Called by blusc.unsupervised.ublc
    """

    create_file_from_source(inputFile, outputFile)
    ncf = nc.Dataset(outputFile, "r+")

    N, p = ncf.variables["X_raw"].shape
    if labels.shape != (N,):
        raise ValueError(
            "Inconsistent shape of labels:", labels.shape, ". Raw labels NOT WRITTEN."
        )

    n_invalidValues = np.sum(np.isnan(labels)) + np.sum(np.isinf(labels))
    if n_invalidValues > 0:
        raise ValueError(
            n_invalidValues, "invalid values in labels. Raw labels NOT WRITTEN."
        )

    LABL = ncf.createVariable("rawlabels", np.int32, ("individuals",))
    LABL[:] = labels[:]
    LABL.units = "Raw (unidentified) labels from unsupervised classification"
    ncf.close()
    if not quiet:
        print("Raw labels have been added in the netcdf file ", outputFile)


def create_file_from_source(src_file, trg_file):
    """Copy a netCDF file into another, using only netCDF4 package.
    It is used to create an output file with the same information as the
    input file but without spoiling it.
    
    
    Parameters
    ----------
    src_file : str
        Path to the input file
    
    trg_file : str
        Path to the output file
    
    
    Returns
    -------
    Generate a netCDF file at the specified location
    
    
    Notes
    -----
    From Stack Overflow (2019): https://stackoverflow.com/questions/13936563/copy-netcdf-file-using-python
    """

    src = nc.Dataset(src_file)
    trg = nc.Dataset(trg_file, mode="w")

    # Create the dimensions of the file
    for name, dim in src.dimensions.items():
        trg.createDimension(name, len(dim) if not dim.isunlimited() else None)

    # Copy the global attributes
    trg.setncatts({a: src.getncattr(a) for a in src.ncattrs()})

    # Create the variables in the file
    for name, var in src.variables.items():
        trg.createVariable(name, var.dtype, var.dimensions)

        # Copy the variable attributes
        trg.variables[name].setncatts({a: var.getncattr(a) for a in var.ncattrs()})

        # Copy the variables values (as 'f4' eventually)
        trg.variables[name][:] = src.variables[name][:]

    # Save the file
    trg.close()


def dtlist2slist(dtlist):
    """Convert a list of datetime into a list of seconds.
    The first datetime object of the list is taken as origin of time.
    
    
    Parameters
    ----------
    dtlist : list of datetime
        List of datetime objects
    
    
    Returns
    -------
    slist : list of float
        List of seconds spent since the time in the first datetime object.
    
    
    Notes
    -----
    Inverse function -> slist2dtlist
    
    TODO: use timestamps (2020/04/13)
    """

    slist = []
    for t in dtlist:
        slist.append((t - dtlist[0]).total_seconds())

    return slist


def extractOrigData(
    originaldatapath: str,
    vartoextract: str="Default",
    altmax: int=4000,
    zoffset: float=0,
    convert_to_dB: bool=False,
):
    """Extract data from a netcdf file at the given path.
    
    Some variables are extracted by default but it is highly recommended
    to precise them. 
    Expected format for the name is:
        CAMPAIGNYYYY_SITE_LAB_TECHNO_INSTRUMENTID_YYYY_MMDD_VERSION.nc
    
    
    Parameters
    ----------
    originaldatapath : str
        Path of the file to be read
    
    vartoextract : str, optional
        Variable to extract (depends on the kind of intrument).
        Default extracts backscatter from ceilometer and absolute temperature
        from microwave radiometer
    
    altmax : int, default=4000
        Highest altitude to use (m agl)
    
    zoffset : float, default=0.0
        Correction of the altitude axis (m agl)
    
    convert_to_dB: bool, default=False
        If True, convert the variable in decibels (apply 10*log)
    
    
    Returns
    -------
    Taxis: ndarray of shape (nt,)
        Time vector (datetime objects)
    
    Zaxis: ndarray of shape (nalt,)
        Altitude vector (m agl)
    
    dataOut: ndarray of shape (nt,nalt)
        Time-altitude matrix with the values of the variable extracted
    """

    # General infos
    defautvartoextract = {"ct25k": "BT", "hatpro": "T"}

    data = nc.Dataset(originaldatapath)
    instru = instrumentkind(originaldatapath)

    # Time axis
    # 2020/04/08: cannot use slist2dtlist because netCDF variables are masked array
    # + some file have bugged time series (PASSY_PASSY_CNRM_CEILOMETER_CT25K_2015_0211_V01.nc)
    Taxis = []
    t0 = scandate(originaldatapath)
    tprev = 0
    t_dismissed = []
    for i in range(len(data.variables["time"])):
        t = int(data.variables["time"][i])
        if t >= tprev:
            Taxis.append(t0 + dt.timedelta(seconds=t))
            tprev = t
        else:
            t_dismissed.append(i)

    # Altitude axis
    altmaxdex = np.where(zoffset + data.variables["Z"][:] < altmax)[0][-1] + 1
    Zaxis = zoffset + np.array(data.variables["Z"][:altmaxdex])

    # Data extraction
    if vartoextract == "Default":
        vartoextract = defautvartoextract[instru]

    if convert_to_dB:
        dataOut = 10 * np.log10(data.variables[vartoextract][:, :altmaxdex])
    else:
        dataOut = np.array(data.variables[vartoextract][:, :altmaxdex])

    # First level of radiometer is bad
    if instru == "hatpro":
        dataOut = dataOut[:, 1:]
        Zaxis = Zaxis[1:]

    # Fix bugged time series (2020/04/08)
    if len(t_dismissed) > 0:
        tmask = np.ones(dataOut.shape[0], dtype=bool)
        tmask[t_dismissed] = False
        dataOut = dataOut[tmask, :]

    return Taxis, Zaxis, dataOut


def grid_to_scatter(x, y, z=None):
    """Convert grid point data into scattered points data.
    
    Grid point data : (x,y,z) with z[i,j]=f(x[i],y[j])
    Scatter point data : (X,Y) with Y[i]=f(X[i,0],X[i,1])
    
    
    Parameters
    ----------
    x : ndarray of shape (nx,) 
        Coordinates on X-axis
        
    y : ndarray of shape (ny,)
        Coordinates on Y-axis
    
    z : ndarray of shape (nx,ny), default=None
        Data for each point (x,y)
    
    
    Returns
    -------
    X : ndarray of shape (nx*ny,2)
        Matrix of coordinates
        
    Y : ndarray of shape (nx*ny,)
        Data vector
    
    
    Notes
    -----
    Inverse function -> scatter_to_grid
    """
    nx = np.size(x)
    ny = np.size(y)

    X = np.full((nx * ny, 2), np.nan)
    X[:, 0] = np.repeat(x, ny)
    X[:, 1] = np.tile(y, nx)
    if np.isnan(X).any():
        print(" ! Warning: ", np.isnan(X).sum(), " NaN in X (grid_to_scatter)")

    if z is None:
        result = X
    else:
        if np.size(z) != nx * ny:
            raise Exception("Problem with inputs dimensions (grid_to_scatter)")
        Y = z.ravel()
        result = X, Y

    return result


def instrumentkind(path):
    """Give the type of instrument that made the measures in the netcdf file.
    This type of instrument is used as a standard ID in other functions like graphics.quicklook.
    
   
    Parameters
    ----------
    path: str
        Path of the file
    
    
    Returns
    -------
    :str
        Instrument ID"""
    fichier = path.split("/")[-1]
    csltiymdv = fichier.split("_")
    return csltiymdv[4].lower()


def load_dataset(datasetname, variables_to_load=["X_raw"], fields_to_load=[]):
    """Load a dataset from a netCDF file (boundary layer classification).
    Naming convention for datasets:
        'DATASETTYPE_yyyy_mmdd.prepkey.nc'
    
    
    Parameters
    ----------
    datasetname: str
        Path to the file to load
    
    variables_to_load: list of str
        List of array-like variables to return
    
    fields_to_load: list of str
        List of parameters to return which are stored in netCDF fields
        
    
    Returns
    -------
    :list
        All requested data in variables_to_load and fields_to_load
    """

    result = []

    dataset = nc.Dataset(datasetname)

    for var in variables_to_load:
        if var == "time":
            Filename = datasetname.split("/")[-1]
            dsdate, prepkey, dotnc = Filename.split(".")
            dstype, yyyy, mmdd = dsdate.split("_")
            t0 = dt.datetime(int(yyyy), int(mmdd[0:2]), int(mmdd[2:4]))

            t_common = np.array(slist2dtlist(np.array(dataset.variables["time"]), t0))
            result.append(t_common)
        else:
            result.append(np.array(dataset.variables[var]))

    for fld in fields_to_load:
        result.append(eval("dataset." + fld))

    if len(result) == 1:
        result = result[0]

    return result


def load_preparation_params(datasetpath):
    """Load the parameters of the preparation of a dataset (boundary layer 
    classification).
    
    Naming convention for datasets:
        'DATASETTYPE_yyyy_mmdd.prepkey.nc'
    With prepkey convention:
        prepkey=CAMPAGNE_PREDICTEURS_INTERPOLATION_dz***_dt***_zmax***
    
    
    Parameters
    ----------
    datasetpath: str
        Path to the file to load
        
    
    Returns
    -------
    predictors: list
        List of atmosheric variables in the dataset
    
    interpMethod: str
        Name of the interpolation method used to estimate the values on the target grid
    
    Dz : int
        Vertical resolution of the target grid (m)
    
    Dt: int
        Time resolution of the target grid (minutes)
    
    zmax: int
        Top altitude of the target grid
    """

    # Input can be a path (*/DATASETTYPE_yyyy_mmdd.prepkey.nc) or a prepkey
    if datasetpath.count(".") > 1:
        datasetname = datasetpath.split("/")[-1]
        dataset_date, prepkey, dotnc = datasetname.split(".")
    else:
        prepkey = datasetpath

    campagne, predicteurs, interpMethod, dzx, dtx, zmaxs = prepkey.split("_")

    predictors = predicteurs.split("-")

    try:
        Dz = int(dzx[2:])
    except ValueError:
        Dz = dzx[2:]

    try:
        Dt = int(dtx[2:])
    except ValueError:
        Dt = dtx[2:]

    zmax = int(zmaxs[4:])

    return predictors, interpMethod.lower(), Dz, Dt, zmax


def scandate(path):
    """ Give the date when the measurement has been done, from its name.
    
    Expected format for the name is:
        CAMPAIGN_SITE_LAB_TECHNO_INSTRUMENTID_YYYY_MMDD_VERSION.nc
    
    
    Parameters
    ----------
    path: str
        Path to the file
        
    
    Returns
    -------
    :datetime
        Date of the file (UTC)
    """
    fichier = path.split("/")[-1]
    campaign, site, lab, techno, instru, yyyy, mmdd, v01 = fichier.split("_")
    return dt.datetime(int(yyyy), int(mmdd[0:2]), int(mmdd[2:4]))


def scatter_to_grid(X, Y=None):
    """Convert scattered points data into grid point data.
    
    Grid point data : (x,y,z) with z[i,j]=f(x[i],y[j])
    Scatter point data : (X,Y) with Y[i]=f(X[i,0],X[i,1])
    
    
    Parameters
    ----------
    X : ndarray of shape (nx*ny,2)
        Matrix of coordinates
        
    Y : ndarray of shape (nx*ny,), default=None
        Data vector
    
    
    Returns
    -------
    x : ndarray of shape (nx,) 
        Coordinates on X-axis
        
    y : ndarray of shape (ny,)
        Coordinates on Y-axis
    
    z : ndarray of shape (nx,ny)
        Data for each point (x,y). Only returned when `Y` is provided
    
    
    Notes
    -----
    Inverse function -> grid_to_scatter
    """

    N, d = np.shape(X)

    if d != 2:
        raise ValueError("More than 2 columns. Not ready so far (scatter_to_grid)")

    if np.sum(np.diff(X[:, 0]) == 0) > np.sum(np.diff(X[:, 1]) == 0):
        xcoord = 0
    else:
        xcoord = 1

    ny = (np.diff(X[:, xcoord]) == 0).tolist().index(False) + 1

    if np.mod(N / ny, 1) != 0:
        raise ValueError("Number of points does not match with dimensions")

    nx = int(N / ny)

    x = X[0:N:ny, xcoord]
    y = X[0:ny, 1 - xcoord]
    if Y is None:
        result = x, y
    else:
        if np.size(Y) != N:
            raise Exception("Inconsistent inputs dimensions (scatter_to_grid)")
        z = np.reshape(Y, (nx, ny))
        result = x, y, z

    return result


def slist2dtlist(slist, t0):
    """Convert a list of seconds into a list of datetime The origin of
    time (second zero) must be specified by a datetime object.
    
    
    Parameters
    ----------
    slist: array-like, dtype=float
        List of seconds spent since the origin of time.
    
    t0: datetime
        Origin of time.
    
    
    Returns
    -------
    dtlist: list, dtype=datetime
        List of datetime objects
    
    
    Notes
    -----
    Inverse function -> dtlist2slist
    
    TODO: use timestamps (2020/04/13)
    """

    dtlist = []
    for t in slist:
        dtlist.append(t0 + dt.timedelta(seconds=t))

    return dtlist


########################
#      TEST BENCH      #
########################
# Launch with
# >> python utils.py
#
# For interactive mode
# >> python -i utils.py
#
if __name__ == "__main__":

    # ARRAY MANIPULATIONS
    # =====================

    # Test of dtlist2slist
    # ------------------------
    print("\n --------------- Test of dtlist2slist")
    t0 = dt.datetime(2020, 4, 1, 0, 0)
    dtlist = [t0, dt.datetime(2020, 4, 1, 0, 0, 4), dt.datetime(2020, 4, 1, 18, 54)]
    slist = dtlist2slist(dtlist)
    print("Return", slist, "(should be [0.0, 4.0, 68040.0])")

    # Test of slist2dtlist
    # ------------------------
    print("\n --------------- Test of slist2dtlist")
    dtlist2 = slist2dtlist(slist, t0)
    print("Is reversible?", dtlist2 == dtlist)

    # Test of grid_to_scatter
    # ------------------------
    print("\n --------------- Test of grid_to_scatter")
    x = np.linspace(12.1, 23.4, 32)
    y = np.linspace(0.6, 1.4, 22)
    z = np.arange(len(x) * len(y)).reshape((len(x), len(y)))
    print("x=", x, "Shape:", x.shape)
    print("y=", y, "Shape:", y.shape)
    print("z Shape:", z.shape)
    X, Y = grid_to_scatter(x, y, z)
    print("X Shape:", X.shape, "(should be (", len(x) * len(y), ",2))")
    print("Y Shape:", Y.shape, "(should be (", len(x) * len(y), ",))")

    # Test of grid_to_scatter
    # ------------------------
    print("\n --------------- Test of grid_to_scatter")
    x1, y1, z1 = scatter_to_grid(X, Y)
    print("Is reversible?", (x == x1).all(), (y == y1).all(), (z == z1).all())

    # LOAD DATA
    # ===========
    dataDir = "../working-directories/0-original-data/"
    CEI_file = dataDir + "CEILOMETER/PASSY_PASSY_CNRM_CEILOMETER_CT25K_2015_0219_V01.nc"
    MWR_file = dataDir + "MWR/PASSY2015_SALLANCHES_CNRM_MWR_HATPRO_2015_0219_V01.nc"

    # Test of instrumentkind
    # ------------------------
    print("\n --------------- Test of instrumentkind")
    print(instrumentkind(CEI_file), "(should be ct25k)")
    print(instrumentkind(MWR_file), "(should be hatpro)")

    # Test of scandate
    # ------------------------
    print("\n --------------- Test of scandate")
    print(scandate(CEI_file), "(should be 2015 Feb. 19)")

    # Test of extractOrigData
    # ------------------------
    print("\n --------------- Test of extractOrigData")
    t, z, V = extractOrigData(MWR_file)
    print("Type check: t", type(t), "z", type(z), "V", type(V))
    print("Shape check: t", len(t), "z", z.shape, "V", V.shape)

    # Test of load_dataset
    # ------------------------
    print("\n --------------- Test of load_dataset")
    testfile = "../working-directories/1-unlabelled-datasets/DATASET_2015_0219.PASSY2015_BT-T_linear_dz40_dt30_zmax2000.nc"
    X_raw = load_dataset(testfile)
    print("DEFAULT Shapes: X_raw", X_raw.shape)
    X_raw, t_val, z_val, desc = load_dataset(
        testfile,
        variables_to_load=["X_raw", "time", "altitude"],
        fields_to_load=["description"],
    )
    print("+T+Z Shapes: X_raw", X_raw.shape, "t_val", t_val.shape, "z_val", z_val.shape)
    print("Description of netCDF:", desc)

    # Test of load_preparation_params
    # ------------------------
    print("\n --------------- Test of load_preparation_params")
    testfile = "../working-directories/1-unlabelled-datasets/DATASET_2015_0219.PASSY2015_BT-T_linear_dz40_dt30_zmax2000.nc"
    print("Returns:", load_preparation_params(testfile))
