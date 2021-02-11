"""
@author: mkuchroo
"""

from scipy.spatial.distance import pdist, cdist
from scipy.spatial.distance import squareform
from scipy.spatial import distance
from scipy import sparse
from joblib import Parallel, delayed
import scipy
from scipy import spatial
import warnings
import graphtools as gt
import scprep
import phate
import pandas as pd
import numpy as np
import vne
from sklearn.utils.extmath import randomized_svd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
import time
import math
import graphtools
from sklearn.decomposition import PCA

def alternating_diffusion_view_powering(datas, n_jobs_multiplier = 1, random_state=42):
    threads = len(datas)
    Gs = Parallel(n_jobs=threads)(delayed(gt.Graph)(datas[i],n_landmark=None, n_pca = 100,
                                                    n_jobs = 1,
                                                   verbose=False, random_state=random_state)
                                  for i in range(len(datas)))
    diff_ops = Parallel(n_jobs=threads)(delayed(scprep.utils.toarray)(Gs[i].diff_op)
                                        for i in range(len(datas)))

    ts = Parallel(n_jobs=threads)(delayed(compute_optimal_t)(diff_ops[i],100)for i in range(len(datas)))
    ts_actual = np.array(ts).copy()
    ts = np.array(ts/ts[np.argmax(ts)])
    ts = np.array(ts / np.min(ts))
    for iteration in range(len(ts)):
        ts[iteration] = math.ceil(ts[iteration])
    ts = ts.astype(int)
    print(ts_actual)
    print(ts)
    diff_ops_t = Parallel(n_jobs=threads)(delayed(np.linalg.matrix_power)(diff_ops[i],
                                                                         ts[i])for i in range(len(datas)))
    i_diff_op = diff_ops_t[0]
    for i in range(1, len(datas)):
        i_diff_op = i_diff_op@diff_ops_t[i]

    return i_diff_op, ts, ts_actual

def localPCA(data, landmarks=None):
    if landmarks == None:
        landmarks = int(data.shape[0]/20)
    G = graphtools.Graph(data, n_landmark=landmarks)
    clusts,_ = np.unique(G.clusters, return_counts=True)
    data_lp = np.zeros(data.shape)
    for c in clusts:
        pca = PCA()
        data_pca = pca.fit_transform(data[G.clusters==c,:])

        save = []
        for i in range(len(pca.explained_variance_)-1):
            save.append(pca.explained_variance_ratio_[i]-pca.explained_variance_ratio_[i+1])
        #rank = np.sum(np.array(save)>.05)
        if len(save)>3:
            rank = vne.find_knee_point(save)
        else:
            rank = np.sum(np.array(save)>.05)
        pca = PCA(n_components=rank)
        data_pca = pca.fit_transform(data[G.clusters==c,:])

        data_lp[G.clusters==c,:] = pca.inverse_transform(data_pca)

    return data_lp


def integrated_diffusion(datas, neighborhoods = None, n_jobs = 1, random_state=42):
    local_datas = []
    for i in range(len(datas)):
        local_datas.append(localPCA(datas[i], landmarks=neighborhoods))
    diff_op, _, _ = alternating_diffusion_view_powering(local_datas, n_jobs_multiplier = n_jobs, random_state=random_state)
    return diff_op

def compute_optimal_t(diff_op, t_max):
    t = np.arange(t_max)
    h = compute_von_neumann_entropy(diff_op, t_max=t_max)
    return find_knee_point(y=h, x=t)

import numpy as np
from scipy.linalg import svd

# Von Neumann Entropy


def compute_von_neumann_entropy(data, t_max=100):
    """
    Determines the Von Neumann entropy of data
    at varying matrix powers. The user should select a value of t
    around the "knee" of the entropy curve.
    Parameters
    ----------
    t_max : int, default: 100
        Maximum value of t to test
    Returns
    -------
    entropy : array, shape=[t_max]
        The entropy of the diffusion affinities for each value of t
    Examples
    --------
    >>> import numpy as np
    >>> import phate
    >>> X = np.eye(10)
    >>> X[0,0] = 5
    >>> X[3,2] = 4
    >>> h = phate.vne.compute_von_neumann_entropy(X)
    >>> phate.vne.find_knee_point(h)
    23
    """
    _, eigenvalues, _ = svd(data)
    entropy = []
    eigenvalues_t = np.copy(eigenvalues)
    for _ in range(t_max):
        prob = eigenvalues_t / np.sum(eigenvalues_t)
        prob = prob + np.finfo(float).eps
        entropy.append(-np.sum(prob * np.log(prob)))
        eigenvalues_t = eigenvalues_t * eigenvalues
    entropy = np.array(entropy)

    return np.array(entropy)


def find_knee_point(y, x=None):
    """
    Returns the x-location of a (single) knee of curve y=f(x)
    Parameters
    ----------
    y : array, shape=[n]
        data for which to find the knee point
    x : array, optional, shape=[n], default=np.arange(len(y))
        indices of the data points of y,
        if these are not in order and evenly spaced
    Returns
    -------
    knee_point : int
    The index (or x value) of the knee point on y
    Examples
    --------
    >>> import numpy as np
    >>> import phate
    >>> x = np.arange(20)
    >>> y = np.exp(-x/10)
    >>> phate.vne.find_knee_point(y,x)
    8
    """
    try:
        y.shape
    except AttributeError:
        y = np.array(y)

    if len(y) < 3:
        raise ValueError("Cannot find knee point on vector of length 3")
    elif len(y.shape) > 1:
        raise ValueError("y must be 1-dimensional")

    if x is None:
        x = np.arange(len(y))
    else:
        try:
            x.shape
        except AttributeError:
            x = np.array(x)
        if not x.shape == y.shape:
            raise ValueError("x and y must be the same shape")
        else:
            # ensure x is sorted float
            idx = np.argsort(x)
            x = x[idx]
            y = y[idx]

    n = np.arange(2, len(y) + 1).astype(np.float32)
    # figure out the m and b (in the y=mx+b sense) for the "left-of-knee"
    sigma_xy = np.cumsum(x * y)[1:]
    sigma_x = np.cumsum(x)[1:]
    sigma_y = np.cumsum(y)[1:]
    sigma_xx = np.cumsum(x * x)[1:]
    det = n * sigma_xx - sigma_x * sigma_x
    mfwd = (n * sigma_xy - sigma_x * sigma_y) / det
    bfwd = -(sigma_x * sigma_xy - sigma_xx * sigma_y) / det

    # figure out the m and b (in the y=mx+b sense) for the "right-of-knee"
    sigma_xy = np.cumsum(x[::-1] * y[::-1])[1:]
    sigma_x = np.cumsum(x[::-1])[1:]
    sigma_y = np.cumsum(y[::-1])[1:]
    sigma_xx = np.cumsum(x[::-1] * x[::-1])[1:]
    det = n * sigma_xx - sigma_x * sigma_x
    mbck = ((n * sigma_xy - sigma_x * sigma_y) / det)[::-1]
    bbck = (-(sigma_x * sigma_xy - sigma_xx * sigma_y) / det)[::-1]

    # figure out the sum of per-point errors for left- and right- of-knee fits
    error_curve = np.full_like(y, np.float("nan"))
    for breakpt in np.arange(1, len(y) - 1):
        delsfwd = (mfwd[breakpt - 1] * x[: breakpt + 1] + bfwd[breakpt - 1]) - y[
            : breakpt + 1
        ]
        delsbck = (mbck[breakpt - 1] * x[breakpt:] + bbck[breakpt - 1]) - y[breakpt:]

        error_curve[breakpt] = np.sum(np.abs(delsfwd)) + np.sum(np.abs(delsbck))

    # find location of the min of the error curve
    loc = np.argmin(error_curve[1:-1]) + 1
    knee_point = x[loc]
    return knee_point
