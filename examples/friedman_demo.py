# %% Setup
import mvBayes as mb
from pyBayesPPR import bppr
import numpy as np

# Generate Data
def f(x):  # Friedman function where first variable is the functional variable
    out = (
        10.0 * np.sin(np.pi * tt * x[0])
        + 20.0 * (x[1] - 0.5) ** 2
        + 10 * x[2]
        + 5.0 * x[3]
    )
    
    return out

tt = np.linspace(0, 1, 50)  # functional variable grid
n = 500  # sample size
p = 9  # number of predictors other (only 4 are used)
X = np.random.rand(n, p)  # training inputs
Y = (
    np.apply_along_axis(f, 1, X) + np.random.normal(size=[n, len(tt)])
)  # training response

ntest = 1000
Xtest = np.random.rand(ntest, p)
Ytest = np.apply_along_axis(f, 1, Xtest) + np.random.normal(size=[ntest, len(tt)])


# %% Fit a multivariate BayesPPR model
mod = mb.mvBayes(
    bppr,
    X,
    Y,
    nBasis=3,
    idxSamplesArg='mcmc_use', # 'mcmc_use' is bppr's equivalent of idxSamples
    # optionally extract posterior samples of residual standard deviation
    residSDExtract = lambda bppr_out: np.sqrt(bppr_out.samples.s2) 
)
mod.basisInfo.plot(idxMV=tt) # Plot PCA decomposition
mod.traceplot()
mod.plot(idxMV=tt)  # Evaluate training data fit
mod.plot(Xtest=Xtest, Ytest=Ytest, idxMV=tt)  # Evaluate test data fit
mod.mvSobol() 
mod.plotSobol(idxMV=tt)

# All posterior predictive samples
Ytest_postSamples = mod.predict(Xtest)
# Posterior predictive mean
Ytest_postMean = np.mean(Ytest_postSamples, axis=0)
# single posterior predictive sample (from MCMC iterations 400-500)
Ytest_postSamples2 = mod.predict(Xtest, idxSamples=np.arange(400, 500), idxSamplesArg='mcmc_use')


# %% Use mvBayesElastic for a joint elastic functional PCA basis
mod = mb.mvBayesElastic(
    bppr,
    X,
    Y,
    nBasis = 3,
    idxSamplesArg='mcmc_use',
    # optionally extract posterior samples of residual standard deviation
    residSDExtract = lambda bppr_out: np.sqrt(bppr_out.samples.s2) 
)
mod.basisInfo.plot()  # Plot PCA decomposition
mod.traceplot()
mod.plot()  # Evaluate training data fit
mod.plot(Xtest=Xtest, Ytest=Ytest)  # Evaluate test data fit
mod.mvSobol()
mod.plotSobol()

# All posterior predictive samples
Ytest_postSamples = mod.predict(Xtest)
