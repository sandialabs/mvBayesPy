import numpy as np
from scipy.optimize import minimize_scalar
from scipy.linalg import toeplitz


def covDiag(resid):
    cov = np.diag(resid.var(axis=0))
    
    return cov


def covMA1(resid, varEqual=True):
    nMV = resid.shape[1]
    
    if nMV == 1:
        return np.array([[1]])
    
    if varEqual is True:
        std = np.std(resid, ddof=1).reshape((1,))
    else:
        std = np.std(resid, ddof=1, axis=0)
    C = std[:, None] * std[None, :]
    covFull = np.cov(resid.T)
    
    def getMA1(coef):
        theta = np.hstack([1, coef, np.repeat(0, nMV-2)])
        return toeplitz(theta)
    
    def minLik(coef):
        covMA1 = C * getMA1(coef)
        D, Q = np.linalg.eigh(covMA1)
        logDetCov = np.sum(np.log(D))
        CovInvMA1 = Q @ np.diag(1/D) @ Q.T
        return logDetCov + np.sum(CovInvMA1 * covFull)
        
    coef = minimize_scalar(
        minLik,
        bounds = (-0.5, 0.5)
    ).x
    
    return C * getMA1(coef), coef


def covAR1(resid, varEqual=True):
    nMV = resid.shape[1]
    
    if nMV == 1:
        return np.array([[1]])
    
    if varEqual is True:
        std = np.std(resid, ddof=1).reshape((1,))
    else:
        std = np.std(resid, ddof=1, axis=0)
    C = std[:, None] * std[None, :]
    covFull = np.cov(resid.T)
    
    def getAR1(coef):
        theta = coef ** np.arange(nMV)
        return toeplitz(theta)
    
    if nMV <= 3:
        def minLik(coef):
            covAR1 = C * getAR1(coef)
            D, Q = np.linalg.eigh(covAR1)
            logDetCov = np.sum(np.log(D))
            CovInvAR1 = Q @ np.diag(1/D) @ Q.T
            return logDetCov + np.sum(CovInvAR1 * covFull)
    else: # compute closed form solution for AR(1)
        def minLik(coef):
            logDetCov = (nMV - 1) * np.log(1 - coef**2)
            theta = np.hstack([1, -coef, np.repeat(0, nMV-2)]) / (1 - coef**2)
            CorInvAR1 = toeplitz(theta)
            for i in range(1, nMV-1):
                CorInvAR1[i, i] += coef**2 / (1 - coef**2)
            CovInvAR1 = CorInvAR1 / C
            return logDetCov + np.sum(CovInvAR1 * covFull)
        
    coef = minimize_scalar(
        minLik,
        bounds = (-1+1e-5, 1-1e-5)
    ).x
    
    return C * getAR1(coef), coef

