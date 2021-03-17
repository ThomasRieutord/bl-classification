#!/usr/bin/env python
# coding: utf-8
"""
MODULE FOR PERFORM UNSUPERVISED BOUNDARY LAYER CLASSIFICATION
Perform unsupervised boundary layer classification on one of the dataset of Passy-2015 2nd IOP.
Take in input the dataset generated by `prepdataset.py`.

Functions are sorted in complexity order:
    - ublc
    - ublc_manyclusters
    - ublc_manyalgo

 +-----------------------------------------+
 |  Date of creation: 02 Apr. 2020         |
 +-----------------------------------------+
 |  Meteo-France                           |
 |  CNRM/GMEI/LISA                         |
 +-----------------------------------------+
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from scipy.cluster import hierarchy as hc
from sklearn.cluster import KMeans

from blusc import utils
from blusc import graphics


def ublc(
    datasetpath,
    algo="hierarchical-average.euclidean",
    target_nb_clusters=4,
    outputDir="../working-directories/2-unidentified-labels/",
    saveNetcdf=False,
    plot_on=False,
):
    """Unsupervised Boundary Layer Classification.
    
    Perform boundary layer classification with ascending hierarchical
    clustering on one of the devoted dataset.
    
    
    Parameters
    ----------
    datasetpath: str
        Path where is located the dataset
    
    algo: str, default="hierarchical-average.euclidean"
        Identifier of the classification algorithm
        Must be of the form algoFamily-param1.param2.param3...
    
    target_nb_clusters: int, default=4
        Number of desired clusters
    
    outputDir: str
        Directory where will be stored the outputs
    
    saveNetcdf: bool, default=False
        If False, the labelled dataset is not saved
    
    plot_on: bool, default=False
        If False, all graphics are disabled
    
    
    Returns
    -------
    zoneID: ndarray
        Cluster labels
    
    If saveNetcdf=True, saves labelled dataset in `outputDir`
    
    If plot_on=True, displays figures of the classification
    
    
    Example
    -------
    >>> from blusc.unsupervised import ublc
    >>> dataDir = "../working-directories/1-unlabelled-datasets/"
    >>> datasetname = "DATASET_2015_0219.PASSY2015_BT-T_linear_dz40_dt30_zmax2000.nc"
    >>> labels = ublc(dataDir + datasetname,algo="hierarchical-average.cityblock")
    >>> labels
    array([1, 1, 1, ..., 3, 3, 3], dtype=int32)
    >>> labels.shape
    (2112,)
    """

    # LOADING AND CHECKING DATASET
    # ==============================

    # Loading
    # --------
    X_raw, t_common, z_common = utils.load_dataset(
        datasetpath, variables_to_load=["X_raw", "time", "altitude"]
    )
    
    if plot_on:
        graphics.quicklook_dataset(datasetpath)

    # Normalization
    # -------------
    scaler = StandardScaler()
    scaler.fit(X_raw)
    X = scaler.transform(X_raw)

    # PERFORM CLASSIFICATION
    # ========================

    algoFamily, algoParams = algo.split("-")

    if algoFamily == "hierarchical":
        # Hierarchical clustering
        # ---------------

        linkageStategy, metricName = algoParams.split(".")

        linkageMatrix = hc.linkage(X, method=linkageStategy, metric=metricName)

        if plot_on:
            graphics.plot_dendrogram(linkageMatrix, algoParams)

        zoneID = hc.fcluster(linkageMatrix, t=target_nb_clusters, criterion="maxclust")

    elif algoFamily == "kmeans":
        # Kmeans clustering
        # ---------------

        km = KMeans(n_clusters=target_nb_clusters, init=algoParams)
        km.fit(X)

        zoneID = km.labels_
    else:
        raise ValueError("Wrong algo argument:", algo)

    if np.min(zoneID) != 0:
        zoneID -= np.min(zoneID)

    # Vizualisation of clusters
    # ===========================

    datasetname = datasetpath.split("/")[-1]
    prefx, prepkey, dotnc = datasetname.split(".")

    if plot_on:
        campg, pred, intp, dz, dt, zm = prepkey.split("_")
        predictors = pred.split("-")
        graphics.clusterZTview(
            t_common, z_common, zoneID, fileName="UBLC_timeAlti_" + prefx[-9:]
        )
        # graphics.cluster2Dview(
            # X_raw[:, 0],
            # predictors[0],
            # X_raw[:, 1],
            # predictors[1],
            # zoneID,
            # fileName="UBLC_featureSpace_" + prefx[-9:],
        # )

    # Write labels in netCDF
    # ----------------------
    if saveNetcdf:
        unidfname = "UNIDFLABELS_" + prefx[-9:] + "." + prepkey + ".nc"
        utils.add_rawlabels_to_netcdf(
            datasetpath, outputDir + unidfname, zoneID, quiet=False
        )

    return zoneID


def ublc_manyclusters(datasetpath, algo="hierarchical-average.euclidean", plot_on=True):
    """Repeat classification for several number of clusters. The output
    is thus the classification scores, not the labels.
    
    Parameters
    ----------
    datasetpath: str
        Path where is located the dataset
    
    algo: str, default="hierarchical-average.euclidean"
        Identifier of the classification algorithm
        Must be of the form algoFamily-param1.param2.param3...
    
    plot_on: bool, default=True
        If False, all graphics are disabled
    
    
    Returns
    -------
    CH_values: list of length 6, dtype=float
        Calinski-Harabaz scores when the number of clusters ranges from 2 to 7
    
    S_values: list of length 6, dtype=float
        Silhouette scores when the number of clusters ranges from 2 to 7
    
    DB_values: list of length 6, dtype=float
        Davies-Bouldin scores when the number of clusters ranges from 2 to 7
    
    
    Example
    -------
    >>> from blusc.unsupervised import ublc_manyclusters
    >>> dataDir = "../working-directories/1-unlabelled-datasets/"
    >>> datasetname = "DATASET_2015_0219.PASSY2015_BT-T_linear_dz40_dt30_zmax2000.nc"
    >>> ch,s,db = ublc_manyclusters(dataDir + datasetname, algo="hierarchical-average.cityblock")
    >>> ch
    [714.5367454983027, 1588.0731546364725, 1119.6997509463438, 1021.069973037269, 1009.4771697162553, 913.2436698606014]
    >>> len(db)
    6
    """

    # LOADING AND CHECKING DATASET
    # ==============================

    # Loading
    # --------
    X_raw, t_common, z_common = utils.load_dataset(
        datasetpath, variables_to_load=["X_raw", "time", "altitude"]
    )

    # Normalization
    # -------------
    scaler = StandardScaler()
    scaler.fit(X_raw)
    X = scaler.transform(X_raw)

    # PERFORM CLASSIFICATION
    # ========================

    algoFamily, algoParams = algo.split("-")

    K_values = np.arange(2, 8)
    CH_values = []
    S_values = []
    DB_values = []
    zoneIDs = []

    if algoFamily == "hierarchical":
        # Hierarchical clustering
        # ---------------

        linkageStategy, metricName = algoParams.split(".")

        linkageMatrix = hc.linkage(X, method=linkageStategy, metric=metricName)

        if plot_on:
            graphics.plot_dendrogram(linkageMatrix, algoParams)

        for target_nb_clusters in K_values:
            zoneID = hc.fcluster(
                linkageMatrix, t=target_nb_clusters, criterion="maxclust"
            )
            if np.min(zoneID) != 0:
                zoneID -= np.min(zoneID)
            zoneIDs.append(zoneID)

            # Quality scores
            # ----------------

            # Calinski-Harabaz index
            ch_score = metrics.calinski_harabasz_score(X, zoneID)
            CH_values.append(ch_score)

            # Silhouette score
            s_score = metrics.silhouette_score(X, zoneID, metric=metricName)
            S_values.append(s_score)

            # Davies-Bouldin index
            db_score = metrics.davies_bouldin_score(X, zoneID)
            DB_values.append(db_score)

    elif algoFamily == "kmeans":
        # Kmeans clustering
        # ---------------

        for target_nb_clusters in K_values:
            km = KMeans(n_clusters=target_nb_clusters, init=algoParams)
            km.fit(X)
            zoneID = km.labels_
            if np.min(zoneID) != 0:
                zoneID -= np.min(zoneID)
            zoneIDs.append(zoneID)

            # Quality scores
            # ----------------

            # Calinski-Harabaz index
            ch_score = metrics.calinski_harabasz_score(X, zoneID)
            CH_values.append(ch_score)

            # Silhouette score
            s_score = metrics.silhouette_score(X, zoneID)
            S_values.append(s_score)

            # Davies-Bouldin index
            db_score = metrics.davies_bouldin_score(X, zoneID)
            DB_values.append(db_score)

    else:
        raise ValueError("Wrong algo argument:", algo)

    # Vizualisation of clusters
    # ===========================

    datasetname = datasetpath.split("/")[-1]
    prefx, prepkey, dotnc = datasetname.split(".")

    if plot_on:
        campg, pred, intp, dz, dt, zm = prepkey.split("_")
        predictors = pred.split("-")
        graphics.clusterZTview_manyclusters(
            t_common, z_common, zoneIDs, fileName="ZTgrid_" + algo
        )
        graphics.cluster2Dview_manyclusters(
            X_raw[:, 0],
            predictors[0],
            X_raw[:, 1],
            predictors[1],
            zoneIDs,
            fileName="featSpace_" + algo,
        )
        graphics.scores_manyclusters(
            K_values,
            [CH_values, S_values, DB_values],
            ["Calinski-Harabasz", "Silhouette", "Davies-Bouldin"],
        )

    return CH_values, S_values, DB_values


def ublc_manyalgo(datasetpath, algo_list=None, plot_on=False):
    """Repeat classification for several algorithms and several number
    of clusters. The output is thus the classification scores, not the labels.
    
    
    Parameters
    ----------
    datasetpath: str
        Path where is located the dataset
    
    algo_list: list, dtype=str
        List of all algorithm identifiers to be tested
        Default is ["hierarchical-average.cityblock", "hierarchical-average.euclidean",
        "hierarchical-complete.cityblock", "hierarchical-complete.euclidean",
        "hierarchical-ward.euclidean", "kmeans-random"]
    
    plot_on: bool, default=False
        If False, all graphics are disabled
    
    
    Returns
    -------
    CH_values: ndarray of shape (len(algo_list),6), dtype=float
        Calinski-Harabaz scores when the number of clusters ranges from 2 to 7
    
    S_values: ndarray of shape (len(algo_list),6), dtype=float
        Silhouette scores when the number of clusters ranges from 2 to 7
    
    DB_values: ndarray of shape (len(algo_list),6), dtype=float
        Davies-Bouldin scores when the number of clusters ranges from 2 to 7
    
    algo_list: list, dtype=str
        List of all algorithm identifiers that have been tested
    
    
    Example
    -------
    >>> from blusc.unsupervised import ublc_manyalgo
    >>> dataDir = "../working-directories/1-unlabelled-datasets/"
    >>> datasetname = "DATASET_2015_0219.PASSY2015_BT-T_linear_dz40_dt30_zmax2000.nc"
    >>> ch,s,db,default_algo_list = ublc_manyalgo(dataDir + datasetname)
    Algo= hierarchical-average.cityblock ( 1 / 6 )
    Algo= hierarchical-average.euclidean ( 2 / 6 )
    Algo= hierarchical-complete.cityblock ( 3 / 6 )
    Algo= hierarchical-complete.euclidean ( 4 / 6 )
    Algo= hierarchical-ward.euclidean ( 5 / 6 )
    Algo= kmeans-random ( 6 / 6 )
    Best in average for CH: kmeans-random
    Best in average for S: hierarchical-average.euclidean
    Best in average for DB: hierarchical-average.euclidean
    >>> ch.shape
    (6, 6)
    >>> default_algo_list
    ['hierarchical-average.cityblock', 'hierarchical-average.euclidean', 'hierarchical-complete.cityblock', 'hierarchical-complete.euclidean', 'hierarchical-ward.euclidean', 'kmeans-random']
        """

    if algo_list is None:
        algo_list = [
            "hierarchical-average.cityblock",
            "hierarchical-average.euclidean",
            "hierarchical-complete.cityblock",
            "hierarchical-complete.euclidean",
            "hierarchical-ward.euclidean",
            "kmeans-random",
        ]

    CH = np.zeros((len(algo_list), 6))
    S = np.zeros((len(algo_list), 6))
    DB = np.zeros((len(algo_list), 6))
    for i in range(len(algo_list)):
        print("Algo=", algo_list[i], "(", i + 1, "/", len(algo_list), ")")
        CH_values, S_values, DB_values = ublc_manyclusters(
            datasetpath, algo=algo_list[i], plot_on=plot_on
        )
        CH[i, :] = CH_values
        S[i, :] = S_values
        DB[i, :] = DB_values

    ibest = np.argmax(np.mean(CH, axis=1))
    print("Best in average for CH:", algo_list[ibest])
    ibest = np.argmax(np.mean(S, axis=1))
    print("Best in average for S:", algo_list[ibest])
    ibest = np.argmin(np.mean(DB, axis=1))
    print("Best in average for DB:", algo_list[ibest])

    return CH, S, DB, algo_list


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

    linkageStategy = "average"
    metricName = "cityblock"
    dataDir = "../working-directories/1-unlabelled-datasets/"
    datasetname = "DATASET_2015_0219.PASSY2015_BT-T_linear_dz40_dt30_zmax2000.nc"
    outputDir = "../working-directories/2-unidentified-labels/"

    graphics.figureDir = outputDir
    graphics.storeImages = False

    # Test of ublc
    # ------------------------
    print("\n --------------- Test of ublc")
    ublc(
        dataDir + datasetname,
        algo="hierarchical-average.cityblock",
        outputDir=outputDir,
        plot_on=True,
        saveNetcdf=True,
    )

    # Test of ublc_manyclusters
    # ------------------------
    print("\n --------------- Test of ublc_manyclusters")
    CH_values, S_values, DB_values = ublc_manyclusters(
        dataDir + datasetname, algo="hierarchical-complete.cityblock", plot_on=True
    )

    # Test of ublc_manyalgo
    # ------------------------
    print("\n --------------- Test of ublc_manyalgo")
    CH_values, S_values, DB_values, algo_list = ublc_manyalgo(
        dataDir + datasetname, plot_on=False
    )
    graphics.internal_quality_map(
        S_values, DB_values, ["Silhouette", "Davies-Bouldin"], algo_list
    )