# -*- coding: utf-8 -*-

# Author: Qi Wang, Ievgen Redko, Sylvain Takerkart 


import numpy as np
from scipy.spatial.distance import cdist


def cost_matrix(x_size, y_size):
    """Compute cost matrix which contains pairwise distances between locations of pixels"""
    nx, ny = x_size, y_size
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, y_size / x_size, ny)
    xv, yv = np.meshgrid(y, x)
    coors = np.vstack((xv.flatten(), yv.flatten())).T
    coor = np.empty(coors.shape)
    coor[:, 0] = coors[:, 1]
    coor[:, 1] = coors[:, 0]
    C = cdist(coor, coor, metric='sqeuclidean')
    return C


def kbcm_dual_optima(g, hs, C, reg, numItermax=1000, stopThr=1e-6, log=False):
    """Compute dual optimas as proposed in the algorithm 3 of [4]
    This function is adapted from POT Python Optimal Transport library.

    Parameters
    -----------
    g: np.ndarray (d,) 
        previous barycenter of hs   
    hs: np.ndarray (d,N)
        N measures which are non-negative with mass smaller than or equal to 1
    C: np.ndarray(d,d)
        Cost matrix containing pairwise euclidean  distances
    reg: float
        Regularization term for entropic regularization
    numItermax: int
        Max number of iterations
    stopThr: float
        Stop threshold 
    log: bool
        Record log if True

    Returns
    -----------
    alphas : np.ndarray (d, N)
        Dual optimas 

    References
    -----------   
    [4] M. Cuturi, and A. Doucet, ``Fast computation of Wasserstein barycenters,''
     In International Conference on Machine Learning pp. 685--693, January 2014.
    """

    g = np.asarray(g, dtype=np.float64).reshape(-1, 1)
    hs = np.asarray(hs, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)

    d = hs.shape[0]
    N = hs.shape[1]
    u = np.ones(hs.shape) / d

    if log:
        log = {'err': []}

    K = np.exp(-C / reg)
    K[K < 1e-300] = 1e-300
    Kp = (1 / g).reshape(-1, 1) * K
    cpt = 0
    err = 1
    while err > stopThr and cpt < numItermax:
        KtransposeU = np.dot(K.T, u)
        v = np.divide(hs, KtransposeU)
        u = 1. / np.dot(Kp, v)
        if cpt % 10 == 0:
            difs = []
            for i in range(N):
                transp = u[:, i].reshape(-1, 1) * (K * v.T[i])
                dif = np.linalg.norm((np.sum(transp, axis=0) - hs[:, i])) ** 2
                difs.append(dif)
            err = np.max(difs)
            if log:
                log['err'].append(err)
        cpt = cpt + 1
    ones = np.ones(hs.shape)
    alphas = -reg * np.log(u) + np.sum(np.log(u), axis=0) * reg / len(g) * ones
    if log:
        return alphas, log
    else:
        return alphas


def kbcm(hs, x_size, y_size, reg, c, q=95, numItermax=500, stopThr=1e-8, log=False):
    """
    KBCM, the algorithm of p-Kantorovich Barycenter with Constrained Mass in [6]

    Parameters
    -----------
    hs : np.ndarray (d, N)
        N measures of size d, x_size times y_size equals d, each measure is non-negative
        with total mass smaller than or equal to 1
    x_size : int
        The longth of each image
    y_size : int
        The width of each image   
    reg : float
        Entropic regularization term 
    c : float
        Step size for gradient update
    q : int
        Quantile
    numItermax: int
        Max number of iterations
    stopThr : float
        Stop threshold 
    log : bool
        Record log if True

    Returns
    -----------
    g : np.ndarray(d, )
        Barycenter of hs

    References
    -----------
    [6] A.Gramfort,  G. Peyr{\'e}, and M. Cuturi,  ``Fast optimal transport
    averaging of neuroimaging data,'' In International Conference on Information
    Processing in Medical Imaging, pp. 261--272, June 2015.
    """
    hs = np.asarray(hs)
    d = hs.shape[0]

    masses_bs = np.sum(hs, axis=0)
    bs_new = np.vstack((hs, 1 - masses_bs))
    mean_mass = np.mean(masses_bs)
    C = cost_matrix(x_size, y_size)
    virtual = np.ones(d) * np.percentile(C, q)
    C_hat = np.zeros((d + 1, d + 1))
    C_hat[:d, :d] = C
    C_hat[d, :d] = virtual
    C_hat[:d, d] = virtual
    g = np.ones(d + 1) / (d + 1)

    cpt = 0
    err = 1
    if log:
        log = {'err': [], 'iter': []}

    while err > stopThr and cpt < numItermax:
        gprev = g
        alphas = kbcm_dual_optima(g, bs_new, C_hat, reg)
        alpha = np.sum(alphas, axis=1)
        gradient = np.exp(-1 * c * alpha)
        g = g * gradient
        a_sum = np.sum(g[:d])
        g = g * mean_mass / a_sum
        g[d] = 1 - mean_mass

        if cpt % 10 == 1:
            err = np.linalg.norm(g - gprev)
            print('{}-th iteration, err: {} '.format(cpt, err))
        if log:
            log['err'].append(err)
            log['iter'].append(cpt)
        cpt = cpt + 1

    if log:
        return g[:d], log
    else:
        return g[:d]


def cost_TLp(h, x_size, g, y_size, eta):
    """Compute cost matrix which is the combination of Euclidean distances
    between the locations and intensities of h and g"""
    h = np.asarray(h)
    g = np.asarray(g)
    if len(h.shape) < 2 or len(g.shape) < 2:
        h = h.reshape((-1, 1))
        g = g.reshape((-1, 1))
    C_hat = cost_matrix(x_size, y_size) + eta * cdist(h, g, 'sqeuclidean')
    return C_hat


def update_Ks(hs, g, reg, x_size, y_size, eta):
    """Update cost matrices of TLp-BI using the obtained barycenter. """
    Cs = np.zeros((hs.shape[0], hs.shape[0], hs.shape[1]))
    for i in range(hs.shape[1]):
        Cs[:, :, i] = cost_TLp(hs[:, i], x_size, g, y_size, eta)
    Ks = np.exp(-Cs / reg)
    Ks[Ks < 1e-300] = 1e-300
    return Ks


def tlp_bi(hs, hs_hat, x_size, y_size, reg, eta, weights=None, outItermax=10,
           inItermax=100, outstopThr=1e-8, instopThr=1e-8, log=False):
    """
    TLp-BI, our proposed algorithm

    Parameters
    -----------
    hs : np.ndarray (d, N)
        N measures of size d, x_size times y_size equals d, each measure
        is non-negative with total mass smaller than or equal to 1
    hs : np.ndarray (d, N)
        N measures of size d, x_size times y_size equals d, each measure
        is normalized with total mass equal to 1
    x_size : int
        The length of each image
    y_size : int
        The width of each image
    reg : float
        Entropic regularization term
    eta : float
        The parameter for cost matrix
    outItermax: int
        Max number of iterations for outer loop
    inItermax: int
        Max number of iterations for inner loop
    outstopThr : float
        Stop threshold for outer loop
    instopThr : float
        Stop threshold for inner loop
    log : bool
        Record log if True

    Returns
    -----------
    g : np.ndarray(d, )
        Barycenter of hs

    """

    if weights is None:
        weights = np.ones(hs.shape[1]) / hs.shape[1]
    else:
        assert (len(weights) == hs.shape[1])
    if log:
        log = {'err': [], 'iter': []}
    mean_mass = np.mean(np.sum(hs, axis=0))
    
    g = np.ones(hs.shape[0]) / hs.shape[0]
    g_hat = np.ones(hs_hat.shape[0]) * mean_mass / hs_hat.shape[0]

    outer_g = g.copy()

    u = np.ones(hs.shape)
    v = np.ones(hs.shape)

    outerr = 1
    outcpt = 0
    barycenters = []
    while outstopThr < outerr and outcpt < outItermax:
        print('outer loop cpt', outcpt)
        # update Ks
        Ks = update_Ks(hs, g, reg, x_size, y_size, eta)

        inerr = 1
        incpt = 0
        while inerr > instopThr and incpt < inItermax:
            
            inner_g = g_hat
            # update u
            for i in range(hs_hat.shape[1]):
                u[:, i] = hs_hat[:, i] / np.dot(Ks[:, :, i], v[:, i])
            
            # update barycenter
            g_hat = np.zeros(hs_hat.shape[0])
            for i in range(hs_hat.shape[1]):
                g_hat = g_hat + weights[i] * np.log(np.maximum(1e-19 * np.ones(len(v[:, i])),
                                                       v[:, i] * np.dot(Ks[:, :, i].T, u[:, i])))
            
            g_hat = np.exp(g_hat)
            inerr = np.linalg.norm(g_hat - inner_g)
            
            # update v
            for i in range(hs_hat.shape[1]):
                v[:, i] = g_hat / np.dot(Ks[:, :, i].T, u[:, i])

            if incpt % 10 == 1:
                print('{}-th iteration, inner loop err: {} '.format(incpt, inerr))

            if log:
                log['err'].append(inerr)
                log['iter'].append(incpt)
            incpt = incpt + 1

        g = g_hat / np.sum(g_hat) * mean_mass
        print('{} inner iterations'.format(incpt))
        barycenters.append(g)
        outerr = np.linalg.norm(g - outer_g)
        outer_g = g
        print('{}-th outer loop, outer loop err: {} '.format(outcpt, outerr))
        outcpt += 1

    if log:
        return g, barycenters, log
    else:
        return g, barycenters


