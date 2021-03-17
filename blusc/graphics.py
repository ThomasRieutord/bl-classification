#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MODULE OF GRAPHICAL TOOLS FOR BOUNDARY CLASSIFICATION.
Contains all the functions depending on a graphical library. It allows 
the use of centralized variables for graphics (save or display? format?)

Variables are first, just after imports.
Functions are sorted in alphabetic order.

 +-----------------------------------------+
 |  Date of creation: 01 Apr. 2020         |
 +-----------------------------------------+
 |  Meteo-France                           |
 |  CNRM/GMEI/LISA                         |
 +-----------------------------------------+

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
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from blusc import utils


fmtImages = ".png"
# Images will be saved under this format (suffix of plt.savefig)

figureDir = ""
# Images will be saved in this directory (prefix of plt.savefig)

storeImages = False
# If True, figures are saved in files but not shown
# If False, figures are not saved in files but always shown

# Database of atmospheric variables long names
DicLeg = {
    "DD": "Wind direction (deg)",
    "FF": "Wind Intensity (m/s)",
    "U": "Zonal Wind (m/s)",
    "V": "Meridional Wind (m/s)",
    "W": "Vertical Wind (m/s)",
    "T": "Temperature (K) ",
    "THETA": "Potential Temperature (K) ",
    "BT": "Aerosol Backscatter (dB) ",
    "SNRW": "Vertical SNR (dB) ",
    "RH": "Relative Humidity (%)",
    "PRES": "pressure (hPa) ",
}

# Database of unique and reproductible clusters colors
cm = {
    0: "o",
    1: "x",
    2: "^",
    3: "v",
    4: "s",
    5: "p",
    6: "*",
    7: "d",
    8: "x",
    9: "o",
    10: "*",
    11: "+",
    12: "<",
    13: ",",
}
clusterMarks = {k:'C'+str(k)+cm[k] for k in range(len(cm))}


def agreement_with_training(
    dayslist,
    md2traincc_perday,
    labelid,
    training_day=dt.datetime(2015,2,19),
    titl=None
):
    """Display the evolution of average distance to cluster center (centroids)
    in each class. It is interpreted as the level of agreement with the class 
    as described in the training set.
    
    
    Parameters
    ----------
    dayslist: array-like of datetime.datetime
        Days on which supervised classification was applied in prediction
    
    md2traincc_perday: array-like
        Score quantifying the agreement with the training set. Currently
        it is the average distance to centroids
    
    labelid: dict
        Connection between cluster numbers and cluster names
    
    titl: str, optional
        Customised title for the figure
    
    
    Returns
    -------
    Create the graphics
    """
    
    if titl is None:
        titl = "Agreement with training along days"
    
    plt.figure()
    plt.title(titl)
    
    plt.plot([training_day]*2, [0.2, 2.0], 'k--')
    
    for k in range(md2traincc_perday.shape[1]):
        plt.plot(
            dayslist,
            md2traincc_perday[:,k],
            clusterMarks[k][:-1]+'+-',
            alpha=0.7,
            label=labelid[k],
            linewidth=2
        )
    
    plt.xlabel("Days")
    plt.ylabel("Average distance to training set cluster centers")
    plt.grid()
    plt.gcf().autofmt_xdate()
    plt.legend(loc="best")
    if storeImages:
        fileName = "agreement_with_training"
        plt.savefig(figureDir + fileName + fmtImages)
        plt.close()
        print("Figure saved:", figureDir + fileName + fmtImages)
    else:
        plt.show(block=False)
    

def cluster2Dview(
    variable1,
    varname1,
    variable2,
    varname2,
    zoneID,
    fileName=None,
    clustersIDs=None,
    displayClustersIDs=False,
    titl=None,
):
    """Plots the projection of the clusters onto the space generated by
    two predictors. It can be used to visualize clusters (boundary layer classification).
    
    
    Parameters
    ----------
    variable1: array-like of shape (N,)
        First variable (vector of values, regardless with their coordinates)
    
    varname1: str
        Standard name of first variable
    
    variable2: array-like of shape (N,)
        Second variable (vector of values, regardless with their coordinates)
    
    varname2: str
        Standard name of second variable
    
    zoneID: array-like of shape (N,)
        Cluster labels for each point
    
    fileName: str, optional
        Customised file name for saving the figure
    
    clustersIDs: dict, optional
        Connection between cluster numbers and boundary layer types
        Example: {0:"CL",1:"SBL",2:"FA",3:"ML"}. Default is {0:0,1:1,...}.
        
    displayClustersIDs: bool, default=False
        If True, displays the clusterIDs over the graph, at the center of the cluster.
    
    titl: str, optional
        Customised title for the figure
    
    
    Returns
    -------
    2-dimensional view of the clusters
        In X-axis is the first variable given
        In Y-axis is the second variable given
        Clusters are shown with differents colors and marks.
    """

    if varname1 not in DicLeg.keys():
        DicLeg[varname1] = varname1
    if varname2 not in DicLeg.keys():
        DicLeg[varname2] = varname2

    # Number of clusters
    K = np.max(zoneID) + 1

    if clustersIDs is None:
        clustersIDs = np.arange(K)
    else:
        for it in clustersIDs.items():
            key, val = it
            clusterMarks[val] = clusterMarks[key]

    if titl is None:
        titl = "Cluster in feature space | " + str(K) + " clusters"

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    fig = plt.figure()
    plt.title(titl)
    plt.plot(variable1, variable2, "k.")
    for k in np.unique(zoneID):
        idxk = np.where(zoneID == k)[0]
        plt.plot(
            variable1[idxk],
            variable2[idxk],
            clusterMarks[clustersIDs[k]],
            linewidth=2,
            label="Cluster " + str(clustersIDs[k]),
        )
        if displayClustersIDs:
            x0text = np.mean(variable1[idxk])
            x1text = np.mean(variable2[idxk])
            plt.text(x0text, x1text, clustersIDs[k], fontweight="bold", fontsize=18)
    plt.xlabel(DicLeg[varname1])
    plt.ylabel(DicLeg[varname2])
    if storeImages:
        if fileName is None:
            fileName = "cluster2Dview_" + varname1 + "-" + varname2 + "_K" + str(K)
        plt.savefig(figureDir + fileName + fmtImages)
        plt.close()
        print("Figure saved:", figureDir + fileName + fmtImages)
    else:
        plt.show(block=False)
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def cluster2Dview_manyclusters(
    variable1, varname1, variable2, varname2, zoneIDs, titl=None, fileName=None
):
    """Plots the projection of the clusters onto the space generated by
    two predictors. It can be used to visualize clusters (boundary layer classification).
    Repeat it for 6 different numbers of clusters.
    
    
    Parameters
    ----------
    variable1: array-like of shape (N,)
        First variable (vector of values, regardless with their coordinates)
    
    varname1: str
        Standard name of first variable
    
    variable2: array-like of shape (N,)
        Second variable (vector of values, regardless with their coordinates)
    
    varname2: str
        Standard name of second variable
    
    zoneIDs: list of array-like of shape (N,)
        Cluster labels for each point and for each number of clusters
    
    titl: str
        Customised title for the figure
    
    fileName: str
        Customised file name for saving the figure
        
    
    Returns
    -------
    3x2 tile of 2-dimensional view of the clusters
        In X-axis is the first variable given
        In Y-axis is the second variable given
        Clusters are shown with differents colors and marks."""

    if varname1 not in DicLeg.keys():
        DicLeg[varname1] = varname1
    if varname2 not in DicLeg.keys():
        DicLeg[varname2] = varname2

    if titl is None:
        titl = "Clusters in feature space"

    n_kvalues = len(zoneIDs)
    nl = int(np.sqrt(n_kvalues))
    nc = int(np.ceil(n_kvalues / nl))

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    fig, axes = plt.subplots(
        nrows=nl, ncols=nc, figsize=(12, 8), sharex=True, sharey=True
    )
    plt.suptitle(titl)

    for ink in range(n_kvalues):
        zoneID = zoneIDs[ink]

        K = np.max(zoneID) + 1
        clustersIDs = np.arange(K)

        plt.subplot(nl, nc, ink + 1)
        plt.plot(variable1, variable2, "k.")
        for k in np.unique(zoneID):
            plt.plot(
                variable1[np.where(zoneID == k)],
                variable2[np.where(zoneID == k)],
                clusterMarks[clustersIDs[k]],
                linewidth=2,
                label="Cluster " + str(clustersIDs[k]),
            )

        if np.mod(ink, nc) == 0:
            plt.ylabel(DicLeg[varname2])
        if ink >= (nl - 1) * nc:
            plt.xlabel(DicLeg[varname1])

    plt.tight_layout()
    if storeImages:
        if fileName is None:
            fileName = "multi_cluster2Dview_" + varname1 + "-" + varname2
        plt.savefig(figureDir + fileName + fmtImages)
        plt.close()
        print("Figure saved:", figureDir + fileName + fmtImages)
    else:
        plt.show(block=False)
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def clusterZTview(
    t_values,
    z_values,
    zoneID,
    delete_mask=None,
    fileName=None,
    clustersIDs=None,
    displayClustersIDs=False,
    titl=None,
):
    """Plots cluster labels in the same time and altitude grid where
    measurements have been done (boundary layer classification).
    
    
    Parameters
    ----------
    t_values: array-like of shape (nt,)
        Vector of time within the day
    
    z_values: array-like of shape (nalt,)
        Vector of altitude
    
    zoneID: array-like of shape (N,)
        Cluster labels of each point
    
    delete_mask: array-like of shape (nt*nalt,)
        Mask at True when observation has been removed by the
        `utils.deletelines` function (to avoid NaNs)
    
    fileName: str, optional
        Customised file name for saving the figure
    
    clustersIDs:  dict, optional
        Connection between cluster numbers and boundary layer types
        Example: {0:"CL",1:"SBL",2:"FA",3:"ML"}. Default is {0:0,1:1,...}.
    
    displayClustersIDs: bool
        If True, displays the clusterIDs over the graph, at the center
        of the cluster.
    
    titl: str, optional
        Customised title for the figure
        
    
    Returns
    -------
    Clusters labels on a time-altitude grid
        In X-axis is the time
        In Y-axis is the height (m agl)
        Clusters are shown with differents colors.
    """

    if clustersIDs is None:
        K = np.max(zoneID) + 1
        clustersIDs = np.arange(K)
    else:
        K = len(clustersIDs.items())
        for it in clustersIDs.items():
            key, val = it
            clusterMarks[val] = clusterMarks[key]

    if titl is None:
        titl = "Cluster in time-altitude grid | " + str(K) + " clusters"

    clist = []
    cticks = []
    cticklabels = []
    for k in range(K):
        cticks.append(k + 0.5)
        cticklabels.append(clustersIDs[k])
        clist.append(clusterMarks[clustersIDs[k]][:-1])
    colormap = ListedColormap(clist)

    # 1. Deleted labels completion (when missing data)
    if delete_mask is not None:
        fullzoneID = np.full(np.size(delete_mask), np.nan)
        fullzoneID[~delete_mask] = zoneID
    else:
        fullzoneID = zoneID

    # 2. Conversion datetime -> seconds
    t0 = t_values[0]
    st_values = utils.dtlist2slist(t_values)

    # 3. Format from grid(z,t) to scatter
    TZ = utils.grid_to_scatter(st_values, z_values)

    # 4. Set labels at grid(z,t) format
    t_trash, z_trash, labels = utils.scatter_to_grid(TZ, fullzoneID)
    if np.max(np.abs(z_values - z_trash)) + np.max(np.abs(st_values - t_trash)) > 1e-13:
        raise Exception(
            "Error in z,t retrieval : max(|z_values-z_trash|)=",
            np.max(np.abs(z_values - z_trash)),
            "max(|t_values-t_trash|)=",
            np.max(np.abs(st_values - t_trash)),
        )

    labels = np.ma.array(labels, mask=np.isnan(labels))

    # 5. Graphic
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    fig = plt.figure()
    # plt.title(titl)
    plt.pcolormesh(t_values, z_values, labels.T, vmin=0, vmax=K, cmap=colormap, shading = "auto")
    if displayClustersIDs:
        for k in np.unique(zoneID):
            idxk = np.where(zoneID == k)[0]
            x0text = t0 + dt.timedelta(seconds=np.mean(TZ[idxk, 0], axis=0))
            x1text = np.mean(TZ[idxk, 1], axis=0)
            plt.text(x0text, x1text, clustersIDs[k], fontweight="bold", fontsize=18)
    cbar = plt.colorbar(label="Cluster label")
    cbar.set_ticks(cticks)
    cbar.set_ticklabels(cticklabels)
    plt.gcf().autofmt_xdate()
    plt.xlabel("Time (UTC)")
    plt.ylabel("Alt (m agl)")
    if storeImages:
        if fileName is None:
            fileName = "clusterZTview_K" + str(K)
        plt.savefig(figureDir + fileName + fmtImages)
        plt.close()
        print("Figure saved:", figureDir + fileName + fmtImages)
    else:
        plt.show(block=False)
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def clusterZTview_manyclusters(
    t_values, z_values, zoneIDs, delete_mask=None, titl=None, fileName=None
):
    """Plots cluster labels in the same time and altitude grid where
    measurements have been done (boundary layer classification).
    Repeat it of 6 differents number of clusters.
    
    
    Parameters
    ----------
    t_values: array-like of shape (nt,)
        Vector of time within the day
    
    z_values: array-like of shape (nalt,)
        Vector of altitude
    
    zoneIDs: list of array-like of shape (N,)
        Cluster labels for each point and for each number of clusters
    
    delete_mask: array-like of shape (nt*nalt,)
        Mask at True when observation has been removed by the
        `utils.deletelines` function (to avoid NaNs)
    
    titl: str, optional
        Customised title for the figure
    
    fileName: str, optional
        Customised file name for saving the figure
    
    
    Returns
    -------
    3x2 tile of clusters labels on a time-altitude grid
        In X-axis is the time
        In Y-axis is the height (m agl)
        Clusters are shown with differents colors.
    """

    if titl is None:
        titl = ""
    
    count2letter = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']

    z_values = z_values / 1000  # convert meters to kilometers

    # 1. Conversion datetime -> seconds
    t0 = t_values[0]
    st_values = utils.dtlist2slist(t_values)

    # 2. Format from grid(z,t) to scatter
    TZ = utils.grid_to_scatter(st_values, z_values)

    n_kvalues = len(zoneIDs)
    nl = int(np.sqrt(n_kvalues))
    nc = int(np.ceil(n_kvalues / nl))

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    fig, axes = plt.subplots(
        nrows=nl, ncols=nc, figsize=(12, 8), sharex=True, sharey=True
    )
    plt.suptitle(titl)
    for ink in range(n_kvalues):
        zoneID = zoneIDs[ink]

        K = np.max(zoneID) + 1
        clustersIDs = np.arange(K)

        clist = []
        cticks = []
        cticklabels = []
        for k in np.unique(zoneID):
            cticks.append(k + 0.5)
            cticklabels.append(clustersIDs[k])
            clist.append(clusterMarks[clustersIDs[k]][:-1])
        colormap = ListedColormap(clist)

        # 3. Set labels at grid(z,t) format
        t_trash, z_trash, labels = utils.scatter_to_grid(TZ, zoneID)
        if (
            np.max(np.abs(z_values - z_trash)) + np.max(np.abs(st_values - t_trash))
            > 1e-13
        ):
            raise Exception(
                "Error in z,t retrieval : max(|z_values-z_trash|)=",
                np.max(np.abs(z_values - z_trash)),
                "max(|t_values-t_trash|)=",
                np.max(np.abs(st_values - t_trash)),
            )

        labels = np.ma.array(labels, mask=np.isnan(labels))

        # 4. Graphic
        plt.subplot(nl, nc, ink + 1)
        im = plt.pcolormesh(t_values, z_values, labels.T, vmin=0, vmax=K, cmap=colormap)
        plt.text(t_values[-7], z_values[-4], count2letter[ink], fontweight='bold', fontsize=16)
        plt.gcf().autofmt_xdate()

        # Colorbar
        cbar = plt.colorbar()
        cbar.set_ticks(cticks)
        cbar.set_ticklabels(cticklabels)

        if np.mod(ink, nc) == nl:
            cbar.set_label("Cluster labels")
        if np.mod(ink, nc) == 0:
            plt.ylabel("Alt (km agl)")
        if ink >= (nl - 1) * nc:
            plt.xlabel("Time (UTC)")

    fig.subplots_adjust(wspace=0, hspace=0.1)
    plt.tight_layout()
    if storeImages:
        if fileName is None:
            fileName = "multi_clusterZTview"
        plt.savefig(figureDir + fileName + fmtImages)
        plt.close()
        print("Figure saved:", figureDir + fileName + fmtImages)
    else:
        plt.show(block=False)


def cobwebplot_categorical(
    X,
    Y,
    posLowest_negHighest=-1,
    n_threads=0.05,
    variablesNames=None,
    inputs_namesNvalues=None,
    titl=None,
):
    '''Draw a cobweb plot from the data. Only for categorical inputs.
    
    Cobweb plots are useful to highlight the influence of several inputs 
    onto an output. For few high/low outputs, the input values yielding
    to each output are linked by a thread. Repeated passage of the threads
    underline the "typical" values of some inputs parameters leading to
    such output. One the other hand, when the threads spread randomly along
    all possible value for an input, it shows that this input is not
    influential.
    
    
    Parameters
    ----------
    X: ndarray of shape (N,p)
        Matrix of inputs parameters (or input parameters indices)
    
    Y: ndarray of shape (N,)
        Vector of outputs
    
    posLowest_negHighest: {-1,1}
        Min or max output ? 1 gives the n_threads lowest. -1 gives the n_threads highest.
    
    n_threads: {int, float}, default=0.05
        Number of threads to display. If it is a float below 1, it is considered
        as a quantile. For example, default value is n_threads=0.05 and it
        returns the 5% more extreme threads.
    
    variablesNames: list of str
        Names of all inputs + name of output (last one)
    
    inputs_namesNvalues: dict
        Names and values of all inputs organized as {input_name:input_values}
    
    titl: str, optional
        Customised title for the figure
    
    
    Returns
    -------
    Display the cobweb plot
        It has p+1 vertical bars, regularly spaced.
        In the X-axis are the p inputs parameters, one for each vertical bar. The last vertical bar is for the output
        In the Y-axis are the values of each inputs, normalised to range within the same bounds as the output.
    
    
    Notes
    -----
    From the work of ThomasRieutord (General Public Licence 3.0)
    URL: https://github.com/ThomasRieutord/cobwebplot-sensitivity
    
    
    Examples
    --------
    >>> import pandas as pd
    >>> from blusc.graphics import cobwebplot_categorical
    >>> filepath = "../local/AS-DATA_categorical_inputs.txt"
    >>> catg=pd.read_csv(filepath,sep=" ")
    >>> variablesNames = list(catg.columns)
    >>> X = catg.iloc[:,:-1].values
    >>> Y = catg.iloc[:,-1].values
    >>> cobwebplot_categorical(X, Y, variablesNames=variablesNames)
    '''
    
    N,p = X.shape
    
    if n_threads>1:
        n_threads=int(n_threads)
    else:
        n_threads=int(N*n_threads)
    
    if variablesNames is None:
        variablesNames= ['X'+str(j+1) for j in range(p)]
        variablesNames.append('Y')
    
    if inputs_namesNvalues is None:
        inputs_namesNvalues={
            variablesNames[j]:np.unique(X[:,j]) for j in range(p)
        }
    
    lowest_highest={1:'lowest',-1:'highest'}
    
    if titl is None:
        titl="Cobweb plot for "+" ".join(
            [
                str(n_threads),
                lowest_highest[posLowest_negHighest],
                variablesNames[-1]
            ]
        )
    
    # Positions of vertical bars (regularly spaced)
    xPos = 2*np.arange(p)
    
    # Positions of categorical variables on the Y-axis
    ymin=min(Y)
    ymax=max(Y)
    yPosText = {
        key:np.linspace(ymin,ymax,len(inputs_namesNvalues[key])+2)[1:-1]
        for key in inputs_namesNvalues.keys()
    }
    
    yPos=np.zeros((N,p))
    for j in range(p):
        yPos[:,j]=yPosText[variablesNames[j]][X[:,j]]
    
    # Sorting the output value (if posLowest_negHighest=1 : ascending, if posLowest_negHighest=-1 decreasing)
    ordrered = np.argsort(posLowest_negHighest*Y)
    
    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    fig=plt.figure(figsize=(10,6))
    
    # p input parameters
    plt.subplot2grid((1,p+1),(0,0),colspan=p)
    
    plt.title(titl)
    
    # Red (horizontalish) threads
    for i in range(n_threads):
        plt.plot(xPos,yPos[ordrered[i],:],'r-',alpha=0.1,linewidth=2)
    
    # Black vertical bars
    for j in range(p):
        plt.plot([xPos[j],xPos[j]],[ymin,ymax],'k-')
        for k in range(len(inputs_namesNvalues[variablesNames[j]])):
            plt.text(
                xPos[j],
                yPosText[variablesNames[j]][k],
                str(inputs_namesNvalues[variablesNames[j]][k])
            )
    plt.xticks(xPos,variablesNames,fontsize=20,rotation=90)
    ax=plt.gca()
    ax.yaxis.set_ticklabels([])
    
    # One output score
    plt.subplot2grid((1,p+1),(0,p))
    plt.title("Output")
    plt.plot([0,0],[min(Y),max(Y)],'k-',linewidth=2)
    plt.plot(np.zeros(n_threads),Y[ordrered[0:n_threads]],'ro')
    plt.xticks([0],[variablesNames[-1]],fontsize=20,rotation=90)
    ax=plt.gca()
    ax.yaxis.set_ticks_position('right')
    
    
    if storeImages:
        fileName = "_".join(
            [
                "cobwebplot",
                lowest_highest[posLowest_negHighest],
                variablesNames[-1]
            ]
        )
        plt.savefig(figureDir + fileName + fmtImages)
        plt.close()
        print("Figure saved:", figureDir + fileName + fmtImages)
    else:
        plt.show(block=False)
    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
    return fig


def comparisonSupervisedAlgo(X_raw, classifiers, resolution=50):
    """Compare several supervised algorithms by display the border of
    the class attribution behind the actual data points.
    
    
    Parameters
    ----------
    X_raw: ndarray of shape (N,p)
        Matrix of data points (one column for each predictor, only the
        two first are used)
    
    classifiers: list of `sklearn` object with `predict` method
        Trained classifiers to test
        
    resolution: int, default=50
        Number of points in each coordinates for the evaluation grid.
        The higher, the more precise is the border.
        
    
    Returns
    -------
    Tile of plot similar to cluster2Dview with the classification
        border in color shades in background.
    """

    print("Prepare comparison graphics...")
    classifiers_keys = [str(clf).split("(")[0] for clf in classifiers]

    BTminbound = X_raw[:, 0].min() - 1
    BTmaxbound = X_raw[:, 0].max() + 1
    Tminbound = X_raw[:, 1].min() - 2
    Tmaxbound = X_raw[:, 1].max() + 2

    T_values = np.linspace(Tminbound, Tmaxbound, resolution)
    BT_values = np.linspace(BTminbound, BTmaxbound, resolution)

    X_pred = utils.grid_to_scatter(BT_values, T_values)

    fig, axs = plt.subplots(1, len(classifiers), figsize=(18, 6))
    plt.tight_layout()
    for icl in range(len(classifiers)):
        clf = classifiers[icl]
        print("Classifier", icl, "/", len(classifiers), classifiers_keys[icl])
        y_pred = clf.predict(X_pred)
        b, t, y = utils.scatter_to_grid(X_pred, y_pred)

        axs[icl].set_title(classifiers_keys[icl])
        axs[icl].pcolormesh(BT_values, T_values, y.T, vmin=-0.5, cmap="nipy_spectral")

        axs[icl].plot(X_raw[:, 0], X_raw[:, 1], "k.")
        axs[icl].set_xlabel(DicLeg["BT"])
        axs[icl].set_ylabel(DicLeg["T"])

    if storeImages:
        fileName = "compSupervisedAlgo"
        plt.savefig(figureDir + fileName + fmtImages)
        plt.close()
        print("Figure saved:", figureDir + fileName + fmtImages)
    else:
        plt.show(block=False)


def estimator_quality(accuracies, chronos, estimator_names, titl=None):
    """Display score versus computation time for a series of estimators.
    Best estimators are in the bottom-right corner.
    
    Abcissa is the R2-score (1-mse/variance: the higher, the better) for
    regressor (typically in prepdataset.py); the accuracy score
    (TP/N: the higher the better) for classifiers (typically in supevised*.py)
    Ordinate is the computing time spent to fit the estimator.
    Both are recorded for Ne estimators and Nr random split of testing
    and training sets.
    
    
    Parameters
    ----------
    accuracies: ndarray of shape (Ne, Nr)
        R2-score (regressors) or accuracy score (classifiers) for all
        estimators and random split
    
    chronos: ndarray of shape (Ne,Nr)
        Computing time for all estimators and random split
    
    estimator_names: array-like of length Ne with dtype=str
        Names of the estimators
    
    titl: str, optional
        Customised title for the figure
    
    
    Returns
    -------
    Create the graphics
    """

    if titl is None:
        titl = "Performance/speed comparison of estimators"

    plt.figure(figsize=(8, 8))
    plt.title(titl)
    for icl in range(len(estimator_names)):
        xtext = np.mean(accuracies[icl, :])
        ytext = np.mean(chronos[icl, :])
        plt.scatter(
            accuracies[icl, :], chronos[icl, :], alpha=0.8, label=estimator_names[icl]
        )
        plt.text(xtext, ytext, estimator_names[icl], fontweight="bold")
    plt.grid()
    if "Regressor" in estimator_names[0]:
        plt.xlabel("R2-score")
    else:
        plt.xlabel("Accuracy score")
    plt.ylabel("Computing time (fit)")
    plt.legend(loc="best")
    if storeImages:
        fileName = "estimator_quality"
        plt.savefig(figureDir + fileName + fmtImages)
        plt.close()
        print("Figure saved:", figureDir + fileName + fmtImages)
    else:
        plt.show(block=False)


def internal_quality_map(
    xscore, yscore, score_names, estimator_names, k_names=None, titl=None
):
    """Display internal quality score (quality score not relying on
    external reference) for several algorithms and number of clusters.
    Best estimators are in the bottom-right corner.
    
    
    Parameters
    ----------
    xscore: ndarray of shape (Ne,Nk)
        Internal score on the x-axis
    
    yscore: ndarray of shape (Ne,Nk)
        Internal score on the y-axis
    
    score_names: array-like of length Ne with dtype=str
        Names of the scores
    
    estimator_names: array-like of length Ne with dtype=str
        Names of the estimators
    
    titl: str, optional
        Customised title for the figure
    
    
    Returns
    -------
    Create the graphics
    """

    if xscore.shape != yscore.shape:
        raise ValueError("Internal scores must have the same shape")

    if titl is None:
        titl = "Internal scores comparison of estimators"

    if k_names is None:
        k_names = np.arange(xscore.shape[1]) + 2

    plt.figure(figsize=(8, 8))
    plt.title(titl)
    for icl in range(len(estimator_names)):
        plt.plot(
            xscore[icl, :], yscore[icl, :], "-o", alpha=0.8, label=estimator_names[icl]
        )
        for ik in range(len(k_names)):
            plt.text(xscore[icl, ik], yscore[icl, ik], k_names[ik])

    plt.grid()
    plt.xlabel(score_names[0])
    plt.ylabel(score_names[1])
    plt.legend(loc="best")
    if storeImages:
        fileName = "internal_quality_map"
        plt.savefig(figureDir + fileName + fmtImages)
        plt.close()
        print("Figure saved:", figureDir + fileName + fmtImages)
    else:
        plt.show(block=False)


def plot_dendrogram(linkageMatrix, algoParams):
    """Plot the dendrogram (hierarchical clustering)
    
    
    Parameters
    ----------
    linkageMatrix: ndarray
        Output of the function `scipy.cluster.hierarchy.linkage`
    
    algoParams: str
        Clustering paramaters (metric and linkage)
    
    
    Returns
    -------
    Dendrogram: merging tree of groups in hierarchical classification.
        Each horizontal lines correspond to the merging of two subgroups.
        X-axis: individuals (or very small groups)
        Y-axis: dissimilarity between groups that are merged
    """

    from scipy.cluster import hierarchy as hc

    plt.figure()
    plt.title(u"Dendrogram | " + algoParams.upper())
    hc.dendrogram(
        linkageMatrix,
        p=10,
        truncate_mode="level",
        distance_sort="ascending",
        no_labels=True,
    )
    plt.ylabel("Cophenetic distance")
    plt.xlabel("Observations")
    if storeImages:
        plt.savefig(figureDir + "dendrogram_" + algoParams + fmtImages)
        plt.close()
    else:
        plt.show(block=False)


def quicklook(originaldatapath, altmax=4000):
    """Quick look at the original data.
    
    
    Parameters
    ----------
    originaldatapath: str
        Path to the data file. Must be using the CF convention for
        naming the file and the variables insides.
        Example: "PASSY2015_SALLANCHES_CNRM_MWR_HATPRO_2015_0210_V01.nc"
    
    altmax: {int, float}
        Top altitude of the graph (meter above ground level)
    
    
    Returns
    -------
    Display the default variable of the given file against time
    and altitude.
        In X-axis is the time
        In Y-axis is the height (m agl)
        Variable values are in shades of colors.
    
    
    Examples
    --------
    >>> from blusc.graphics import quicklook
    >>> CEI_file = "../working-directories/0-original-data/CEILOMETER/PASSY_PASSY_CNRM_CEILOMETER_CT25K_2015_0219_V01.nc"
    >>> quicklook(CEI_file)
    """
    t, z, V = utils.extractOrigData(originaldatapath, altmax=altmax)
    instru = utils.instrumentkind(originaldatapath)

    if instru == "hatpro":
        titl = (
            "Original temperature | "
            + instru.upper()
            + " | "
            + t[0].strftime("%Y/%m/%d")
        )
        valmin = 260
        valmax = 285
        colormp = "jet"
        clabl = "Temperature (K)"
        fileName = "_".join(
            ["orig", "T", instru.upper(), t[0].strftime("%Y%m%d")]
        )
        # fileName = "T_orig_" + t[0].strftime("%Y%m%d")
    else:
        titl = (
            "Original backscatter | "
            + instru.upper()
            + " | "
            + t[0].strftime("%Y/%m/%d")
        )
        valmin = -11
        valmax = 20
        colormp = "plasma"
        clabl = "Aerosol Backscatter (dB)"
        fileName = "_".join(
            ["orig", "BT", instru.upper(), t[0].strftime("%Y%m%d")]
        )
        # fileName = "BT_orig_" + t[0].strftime("%Y%m%d")
        # Negative backscatter are outliers
        with np.errstate(invalid="ignore"):
            V[V <= 0] = np.nan
            V = 10 * np.log10(V)

    plt.figure()
    plt.title(titl)
    plt.pcolormesh(t, z, V.T, vmin=valmin, vmax=valmax, cmap=colormp,shading="auto")
    plt.colorbar(label=clabl)
    plt.gcf().autofmt_xdate()
    plt.xlabel("Time (UTC)")
    plt.ylabel("Alt (m agl)")
    if storeImages:
        plt.savefig(figureDir + fileName + fmtImages)
        plt.close()
    else:
        plt.show(block=False)

def quicklook_dataset(datasetpath, altmax=4000):
    """Quick look at the original data.
    
    
    Parameters
    ----------
    datasetpath: str
        Path to the data file. Must follow the convention adopted in
        the BLUSC program.
        Example: "DATASET_2015_0210.PASSY2015_BT-T_linear_dz40_dt30_zmax2000.nc"
    
    altmax {int, float}
        Top altitude of the graph (meter above ground level)
    
    
    Returns
    -------
    Display the default variable of the given file against time
    and altitude.
        In X-axis is the time
        In Y-axis is the height (m agl)
        Variable values are in shades of colors.
    """
    X_raw, t, z = utils.load_dataset(
        datasetpath, variables_to_load=["X_raw", "time", "altitude"]
    )
    TZ = utils.grid_to_scatter(utils.dtlist2slist(t), z)
    
    for p in range(X_raw.shape[1]):
        
        t1,z1,V = utils.scatter_to_grid(TZ,X_raw[:,p])
        
        plt.figure()
        # plt.title("Variable "+str(p)+" of dataset")
        plt.pcolormesh(t, z, V.T, shading="auto")
        plt.colorbar()
        plt.gcf().autofmt_xdate()
        plt.xlabel("Time (UTC)")
        plt.ylabel("Alt (m agl)")
        if storeImages:
            fileName = "QL_Xraw"+str(p)
            plt.savefig(figureDir + fileName + fmtImages)
            plt.close()
        else:
            plt.show(block=False)


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def scores_manyclusters(K_values, score_list, score_names):
    """Plot the classification scores against the number of clusters.
    
    
    Parameters
    ----------
    K_values: ndarray of shape (Nk,)
        Number of clusters
    
    score_list: list array-like of shape (Nk,)
        Scores arrays for the different number of cluster values and for
        each score (usually three available)
    
    score_names: list of str
        Scores names
    
    
    Returns
    -------
    Vertically-stacked figures sharing X-axis which the number of clusters.
    Each subfigure as one the score as Y-axis.
    Optimal values of the scores are highlighted by yellow squares.
    """

    fig, axs = plt.subplots(len(score_list), sharex=True, gridspec_kw={"hspace": 0})
    fig.suptitle("Optimal n_clusters acc.t. several scores")
    for s in range(len(score_list)):
        axs[s].plot(K_values, score_list[s])
        axs[s].set_ylabel(score_names[s])
        axs[s].set_xlabel("Number of clusters")
        if score_names[s] == "Davies-Bouldin":
            kbest = np.argmin(score_list[s])
        else:
            kbest = np.argmax(score_list[s])
        axs[s].plot(K_values[kbest], score_list[s][kbest], "ys")
        axs[s].xaxis.grid()

    # Hide x labels and tick labels for all but bottom plot.
    # Source (2020/04/02): https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplots_demo.html
    for ax in axs:
        ax.label_outer()

    if storeImages:
        fileName = "classifScores"
        plt.savefig(figureDir + fileName + fmtImages)
        plt.close()
        print("Figure saved:", figureDir + fileName + fmtImages)
    else:
        plt.show(block=False)


########################
#      TEST BENCH      #
########################
# Launch with
# >> python graphics.py
#
# For interactive mode
# >> python -i graphics.py
#
if __name__ == "__main__":

    dataDir = "0-original-data/training-data/"
    CEI_file = "PASSY_PASSY_CNRM_CEILOMETER_CT25K_2015_0219_V01.nc"
    MWR_file = "PASSY2015_SALLANCHES_CNRM_MWR_HATPRO_2015_0219_V01.nc"

    # Test of quicklook
    # ------------------------
    print("\n --------------- Test of quicklook")
    quicklook(dataDir + CEI_file)
    quicklook(dataDir + MWR_file)

    input("\n Press Enter to exit (close down all figures)\n")