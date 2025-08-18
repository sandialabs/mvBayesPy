import numpy as np
import time
import mvBayes as mb
from scipy.stats import multivariate_normal

def mvBayesCV(bayesModel, X, Y, nTrain=None, nTest=None, nRep=1, seed=None, coverageTarget=0.95, idxSamples="default", uqTruncMethod="gaussian", **kwargs):
    """
    Cross-Validation (CV) of a Multivariate Bayesian Regression Model
    
    Parameters:
        bayesModel: A Bayesian regression model-fitting function, with first argument taking an nxp input matrix and second argument taking an n-vector of numeric responses.
        X: A matrix of predictors of dimension nxp, where n is the total number of examples (including training and test sets) and p is the number of inputs (features).
        Y: A response matrix of dimension nxq, where q is the number of multivariate/functional responses.
        nTrain: Number of examples to use in the training set. If None, nTrain = n - nTest; unless nTest is also None, in which case nTrain = ceil(n/2).
        nTest: Number of examples to use in the test set. If None, nTest = n - nTrain.
        nRep: Number of repetitions of CV process.
        seed: Randomization seed, for replication of the train/test split. If None, no seed is set.
        coverageTarget: Level of coverage desired (default: 0.95).
        idxSamples: Which samples to use in CV (default: "all").
        uqTruncMethod: Method to use for UQ truncation ("gaussian" or "empirical").
        **kwargs: Additional arguments to mvBayes, including arguments to bayesModel.
    
    Returns:
        A dictionary containing the out-of-sample RMSE for each replication, fitting and prediction times, and other metrics.
    """
    # Setup
    n, p = X.shape
    alpha = 1 - coverageTarget

    if nTest is None:
        if nTrain is None:
            nTest = n // 2  # half in test set
            nTrain = n - nTest
        elif nTrain >= n:
            raise ValueError("Must have nTrain < nrow(X)")
        else:
            nTest = n - nTrain
    else:
        if nTest >= n:
            raise ValueError("Must have nTest < nrow(X)")
        elif nTrain is None:
            nTrain = n - nTest
        elif nTrain + nTest > n:
            raise ValueError("Must have nTrain + nTest <= n")

    # Get fold indices
    np.random.seed(seed)
    idxTest = [np.random.choice(n, size=nTest, replace=False) for _ in range(nRep)]
    idxTrain = [np.random.choice(np.setdiff1d(np.arange(n), idx), size=nTrain, replace=False) for idx in idxTest]
    np.random.seed(None)  # Reset seed

    # Run CV
    rmse = np.zeros(nRep)
    rSquared = np.zeros(nRep)
    coverage = np.zeros(nRep)
    intervalWidth = np.zeros(nRep)
    intervalScore = np.zeros(nRep)
    fitTime = np.zeros(nRep)
    predictTime = np.zeros(nRep)

    for r in range(nRep):
        # Set up train/test split
        Xtrain, Ytrain = X[idxTrain[r], :], Y[idxTrain[r], :]
        Xtest, Ytest = X[idxTest[r], :], Y[idxTest[r], :]

        # Fit models
        startFit = time.time()
        fit = mb.mvBayes(bayesModel, Xtrain, Ytrain, **kwargs)
        fitTime[r] = time.time() - startFit

        # Predict
        startPred = time.time()
        preds = fit.predict(Xtest, idxSamples=idxSamples)
        predictTime[r] = time.time() - startPred

        Yhat = np.median(preds, axis=0)

        # Calculate RMSE and R-squared
        rmse[r] = np.sqrt(np.mean((Ytest - Yhat) ** 2))
        rSquared[r] = 1 - np.mean((Ytest - Yhat) ** 2) / np.mean((Ytest - np.mean(Ytrain, axis=0)) ** 2)

        # Get truncation error for UQ
        if uqTruncMethod == "gaussian":
            truncErrorVar = np.cov(fit.basisInfo.truncError, rowvar=False)
            truncError = multivariate_normal.rvs(
                mean=np.zeros(truncErrorVar.shape[0]),
                cov=truncErrorVar,
                size=np.prod(preds.shape[:2])
            ).reshape(preds.shape)
        elif uqTruncMethod == "empirical":
            idxResample = np.random.choice(nTrain, size=np.prod(preds.shape[:2]), replace=True)
            truncError = fit.basisInfo.truncError[idxResample, :].reshape(preds.shape)
        preds += truncError
        del truncError

        # Get regression error for UQ
        coefsResidError = np.zeros(preds.shape[:2] + (fit.basisInfo.nBasis, ))
        for k in range(fit.basisInfo.nBasis):
            residSD = np.repeat(
                fit.bmList[k].samples.residSD,
                preds.shape[1]
            ).reshape(preds.shape[:2])
            coefsResidError[:, :, k] = np.random.normal(
                0.0,
                residSD,
                preds.shape[:2]
            )
        residError = coefsResidError @ fit.basisInfo.basis
        del coefsResidError
        preds += residError
        del residError

        # Calculate distance from posterior mean
        distBound = np.zeros(nTest)
        for idx in range(nTest):
            distSamples = np.sqrt(np.mean((preds[:, idx, :] - Yhat[idx, :]) ** 2, axis=1))
            distBound[idx] = np.quantile(distSamples, coverageTarget)
        distTest = np.sqrt(np.mean((Ytest - Yhat) ** 2, axis=1))

        # Calculate UQ metrics
        distRatio = distTest / distBound
        coverage[r] = np.mean(distRatio <= 1)
        intervalWidth[r] = np.exp(np.mean(np.log(distBound)))
        intervalScore[r] = intervalWidth[r] * np.exp(np.mean(np.log(distRatio) * (distRatio > 1)) / alpha)

    # Output results
    out = {
        "rmse": rmse,
        "rSquared": rSquared,
        "coverageTarget": coverageTarget,
        "coverage": coverage,
        "intervalWidth": intervalWidth,
        "intervalScore": intervalScore,
        "fitTime": fitTime,
        "predictTime": predictTime,
        "effectiveArgs": {
            "nTrain": nTrain,
            "nTest": nTest,
            "nRep": nRep,
            "seed": seed,
            "coverageTarget": coverageTarget,
            "idxSamples": idxSamples,
            "uqTruncMethod": uqTruncMethod
        }
    }

    return out


