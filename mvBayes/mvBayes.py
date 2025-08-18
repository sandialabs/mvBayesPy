# %% Import modules
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from scipy.stats import qmc
import patsy
from scipy import special as sp

from matplotlib.ticker import ScalarFormatter
import os
import inspect

try:
    from pathos.multiprocessing import ProcessingPool as Pool

    PATHOS_AVAILABLE = True
except ImportError:
    PATHOS_AVAILABLE = False

try:
    import fdasrsf as fs

    FDASRSF_AVAILABLE = True
except ImportError:
    FDASRSF_AVAILABLE = False

try:
    import pyBASS

    PYBASS_AVAILABLE = True
except:
    PYBASS_AVAILABLE = False
    
from .cov import (
    covAR1,
    covMA1,
    covDiag
)

from .cv import cv


# %% Set up PCA basis functions
class basisSetup:
    """
    Compute basis components for a matrix Y. Used in mvBayes.
    """

    def __init__(
        self,
        Y,
        basisType="pca",
        customBasis=None,
        nBasis=None,
        propVarExplained=0.99,
        center=True,
        scale=False,
        thresh=1e-15,
        basisTransform=None
    ):
        """
        :param Y: numpy array of shape (n, nMV) for which a reduced-dimension
                  basis will be computed.
        :param basisType: string stating basis type. Options are `pca`,
                          `pns`, `bspline`, `legendre`, or `custom`.
        :param nBasis: int number of basis components to use (optional).
        :param propVarExplained: float proportion (between 0 and 1) of variation to explain
                                 when choosing number of basis components (if nBasis=None).
        :param center: bool, whether to center Y before basis computations.
        :param scale: bool, whether to scale Y before basis computations.
        :param thresh: 1e-15, eigenvalue thresh for pns
        :return: object with plot method.
        """
        self.Y = Y
        self.nMV = self.Y.shape[1]
        self.basisType = basisType
        self.center = center
        self.scale = scale
        self.Ycenter = 0
        self.Yscale = 1
        if basisType == "pns":
            center = False
            scale = False
        if center:
            self.Ycenter = np.mean(Y, axis=0)
        if scale:
            self.Yscale = np.std(Y, axis=0)
            self.Yscale[self.Yscale == 0] = 1
        Ystandard = (Y - self.Ycenter) / self.Yscale

        if basisType == "pca":
            basisConstruct = PCA()
            self.basisConstruct = basisConstruct
            basisConstruct.fit(Ystandard)
            self.varExplained = basisConstruct.explained_variance_
            basis = basisConstruct.components_
            coefs = basisConstruct.transform(Ystandard)
        elif basisType in ["bspline", "legendre", "custom"]:
            if basisType in ["bspline", "legendre"]:
                if nBasis is None:
                    raise ValueError(
                        f"nBasis must be specified for basisType='{basisType}'"
                        )
                if basisType == "bspline":
                    if nBasis < 3:
                        raise ValueError(
                        "Must have nBasis >= 3 for basisType='bspline'"
                        )
                    customBasis = patsy.bs(np.linspace(0,1,self.nMV), df=nBasis).T
                elif basisType == "legendre":
                    if (nBasis % 2) == 1:
                        print("nBasis must be even for basisType='legendre'. Setting nBasis+=1")
                        nBasis += 1
                    customBasis = basisLegendre(np.linspace(0,1,self.nMV), int(nBasis / 2), 1)
            else: # custom basis
                if customBasis is None:
                    raise ValueError(
                        "Must provide customBasis if basisType=='custom'"
                    )
                if customBasis.shape[1] != self.nMV:
                    raise ValueError(
                        "customBasis.shape[1] != Y.shape[1]"
                        )
            self.basisConstruct = customBasisConstruct(customBasis, Ystandard, basisTransform)
            basis = self.basisConstruct.basis
            coefs = self.basisConstruct.coefs
            self.varExplained = self.basisConstruct.varExplained
        elif basisType == "pns":
            if not FDASRSF_AVAILABLE:
                raise Exception(
                    "Module 'fdasrsf' not available. basisSetup cannot proceed"
                )
            
            n, d = Y.shape
            self.tt = np.linspace(0, 1, d)

            radius = np.mean(np.sqrt((Y**2).sum(axis=1)))
            pnsdat = Y / np.tile(np.sqrt((Y**2).sum(axis=1)), (d, 1)).T

            resmat, PNS = fs.pns.fastpns(pnsdat.T, n_pc="Approx", thresh=thresh)
            self.basisConstruct = pnsBasisConstruct(resmat, PNS, radius, Y)

            basis = self.basisConstruct.basis
            coefs = self.basisConstruct.coefs
            self.varExplained = self.basisConstruct.varExplained

        else:
            raise Exception("Un-supported basisType")

        propVarCumSum = np.cumsum(self.varExplained) / np.sum(self.varExplained)
        if nBasis is None:
            nBasis = min(
                np.where(propVarCumSum > propVarExplained)[0][0] + 1,
                basis.shape[0]
                )
        self.nBasis = nBasis
        self.propVarExplained = propVarCumSum[nBasis - 1]
        self.propVarCumSum = propVarCumSum

        self.basis = basis[: self.nBasis, :]
        self.coefs = coefs[:, : self.nBasis]
        Ytrunc = self.getYtrunc()
        self.truncError = self.Y - Ytrunc

        return

    @property
    def _Y(self):
        """
        Creates a common way to extract object on which basis is computed.
        Can call self._Y or self.Y
        """
        return self.Y

    def getYtrunc(self, Ytest=None, coefs=None, nBasis=None):
        """
        Get a "truncated" reconstruction of Ytest, reconstructed from the first nBasis
        basis components. If Ytest is provided, coefs will be computed. If neither is provided,
        self.coefs will be used. coefs is then used to get the truncated reconstruction.

        Parameters
        ----------
        Ytest : numpy array of shape (n, nMV) for which a truncated reconstruction
            will be computed. If None, uses coefs.
        coefs : numpy array of shape (n, self.nBasis) used to create a truncated
            reconstruction. If None, uses Ytest to compute coefs. If Ytest is also
            None, uses self.coefs.
        nBasis : integer indicating the number of basis components to use in the
            truncation. Max is self.nBasis. If None, uses self.nBasis.

        Returns
        -------
        Ytrunc
            numpy array of shape (n, nMV) containing the truncated reconstruction.

        """
        if coefs is None:
            coefs = self.getCoefs(Ytest)
        if nBasis is None or nBasis > self.nBasis:
            nBasis = self.nBasis
        if self.basisType == "pns":
            PNS = self.basisConstruct.PNS
            radius = self.basisConstruct.radius
            inmat = np.zeros((PNS["radii"].shape[0], coefs.shape[0]))
            inmat[:nBasis, :] = coefs[:, :nBasis].T
            YtruncStandard = fs.pns.fastPNSe2s(inmat, PNS) * radius
        else:
            YtruncStandard = coefs[:, :nBasis] @ self.basis[:nBasis, :]
        return YtruncStandard * self.Yscale + self.Ycenter

    def getCoefs(self, Ytest=None):
        """
        Transform Ytest into basis coefficients

        Parameters
        ----------
        Ytest : numpy array of shape (n, nMV) for which basis coefficients will
        be computed. If Ytest is None, uses self.Y.

        Returns
        -------
        coefs
            numpy array of shape (n, self.nBasis) containing basis coefficients
            corresponding to Ytest.

        """
        if Ytest is None:
            return self.coefs
        else:
            YtestStandard = (Ytest - self.Ycenter) / self.Yscale
            return self.basisConstruct.transform(YtestStandard)[:, :self.nBasis]

    def preprocessY(self, Ytest=None):
        """
        Factory method to perform preprocessing on Y before doing the basis
        decomposition. For basisSetup, simply returns Ytest (if provided) or self.Y (if not).

        Parameters
        ----------
        Ytest : numpy array of shape (n, nMV) on which preprocessing will
        be done. If Ytest is None, uses self.Y.

        Returns
        -------
        numpy array of shape (n, nMV) containing the preprocessed version of Ytest

        """
        if Ytest is None:
            return self._Y
        else:
            return Ytest

    def plot(
        self,
        nBasis=None,
        propVarExplained=None,
        nPlot=None,
        idxMV=None,
        xscale='linear',
        xlabel='Multivariate Index',
        file=None,
        title=None,
        **kwargs,
    ):
        """
        Plot of basis components and percent variance explained
        * top left - Y
        * top right - basis components scaled by coefs
        * bottom left - Truncation error due to dimension reduction
        * bottom right - Percent variance explained by basis components. Colors
        correspond to top right plot

        :param nBasis: int (<=self.nBasis) number of basis components to plot.
            If None, propVarExplained will be used to choose how many basis components
            to plot. If propVarExplained is also None, self.nBasis components will be used.
        :param propVarExplained: float proportion (between 0 and 1) of variation to explain
            when choosing number of basis components (if nBasis=None) to plot. If both
            propVarExplained and nBasis are None, self.nBasis will be used.
        :param nPlot: int number of observations to plot. If None, uses min(n, 1000)
        :param file: str where to save the file. Shown, not saved, if None.
        :param title: str title for the top of the plot. If None, uses "Basis Decomposition of Y"
        :return: None
        """

        propVar = self.varExplained / np.sum(self.varExplained)

        if nBasis is None:
            if propVarExplained is None:
                nBasis = self.nBasis
            else:
                nBasis = np.where(np.cumsum(propVar) >= propVarExplained)[0][0] + 1
        elif nBasis > len(self.varExplained):
            nBasis = self.nBasis

        n = self._Y.shape[0]

        if nPlot is None:
            nPlot = min(n, 1000)
        elif nPlot > n:
            print(
                "nPlot should be at most n, where n=len(Xtest) (or n=len(X) if Xtest is None). Using nPlot=n."
            )
            nPlot = n
        idxPlot = np.random.choice(n, nPlot, replace=False)

        if idxMV is None:
            idxMV = list(range(self.nMV))

        fig = plt.figure(figsize=(8, 6))

        cmap = plt.get_cmap("tab20")

        fig.add_subplot(2, 2, 1)
        plt.plot(idxMV, self._Y[idxPlot].T, color="darkblue", alpha=0.5)
        plt.xlim(np.min(idxMV), np.max(idxMV))
        plt.xscale(xscale)
        plt.ylabel("Y")
        plt.xlabel(xlabel)

        fig.add_subplot(2, 2, 2)
        for k in range(nBasis):
            basisScaled = (
                np.outer(self.coefs[idxPlot, k], self.basis[k, :]) * self.Yscale
            )
            plt.plot(idxMV, basisScaled.T, color=cmap(k % 20), alpha=0.5)
        plt.xlim(np.min(idxMV), np.max(idxMV))
        plt.xscale(xscale)
        plt.ylabel("Basis Projection")
        plt.xlabel(xlabel)
        truncPlotYlim = plt.ylim()

        fig.add_subplot(2, 2, 3)
        plt.plot(idxMV, self.truncError[idxPlot].T, color="silver", alpha=0.5)
        plt.xlim(np.min(idxMV), np.max(idxMV))
        plt.ylim(truncPlotYlim)
        plt.xscale(xscale)
        plt.ylabel("Truncation Error")
        plt.xlabel(xlabel)

        fig.add_subplot(2, 2, 4)
        if nBasis < len(self.varExplained):
            plt.scatter(
                np.repeat(nBasis, len(self.varExplained) - nBasis),
                100 * np.cumsum(propVar[nBasis:]),
                color="silver",
            )
        for k in range(nBasis):
            plt.scatter(k, 100 * propVar[k], color=cmap(k % 20))
        plt.xticks(range(nBasis + 1), list(range(1, nBasis + 1)) + ["T"])
        plt.ylabel("%Variance")
        plt.xlabel("Component")
        plt.title(
            f"{np.round(100*self.propVarExplained, 1)}% Variance Explained"
        )

        if title is None:
            fig.suptitle("Basis Projection")
        else:
            fig.suptitle(title)
        fig.tight_layout()

        if file is None:
            plt.show()
        else:
            plt.savefig(file, **kwargs)

        plt.close(fig)


def legendre(N, X):
    matrixReturn = np.zeros((N + 1, X.shape[0]))
    for i in enumerate(X):
        currValues = sp.lpmn(N, N, i[1])
        matrixReturn[:, i[0]] = np.array([j[N] for j in currValues[0]])
    return matrixReturn


def basisLegendre(fDomain, nLegendre, pFourier):
    basis = np.zeros((2 * nLegendre, fDomain.shape[0]))
    for i in range(0, 2 * nLegendre):
        fDomainScaled = 2 * (fDomain / pFourier) - 1
        tmp = legendre(i + 1, fDomainScaled)
        basis[i, :] = tmp[0, :]

    return basis


def isOrthogonal(basis, tol=1e-10):
    """
    Check if a matrix is orthogonal.
    
    Parameters:
        basis (ndarray): The basis to check.
        tol (float): Tolerance for numerical precision.
        
    Returns:
        bool: True if the basis is orthogonal, False otherwise.
    """    
    return np.allclose(basis @ basis.T, np.eye(basis.shape[0]), atol=tol)


def orthogonalize(basis):
    Q, R = np.linalg.qr(basis.T)
    newBasis = Q.T
    return newBasis


class customBasisConstruct:
    def __init__(
        self,
        customBasis,
        Ystandard,
        basisTransform=None
    ):        
        self.basisTransform = basisTransform
        if self.basisTransform is None:
            varTotal = np.var(Ystandard, axis=0).sum()
            if isOrthogonal(customBasis):
                self.basis = customBasis.copy()
            else:
                self.basis = orthogonalize(customBasis)
        else:
            D, Q = np.linalg.eigh(self.basisTransform)
            self.basisTransformSqrt = Q @ np.diag(np.sqrt(D)) @ Q.T
            varTotal = np.var(Ystandard @ self.basisTransformSqrt, axis=0).sum()
            if isOrthogonal(customBasis @ self.basisTransformSqrt):
                self.basis = customBasis.copy()
            else:
                covSqrt = Q @ np.diag(np.sqrt(1/D)) @ Q.T
                self.basis = orthogonalize(customBasis @ self.basisTransformSqrt) @ covSqrt

        self.coefs = self.transform(Ystandard)
        self.varExplained = np.var(self.coefs, axis=0)
        if customBasis.shape[0] < customBasis.shape[1]:
            self.varExplained = np.hstack(
                [
                    self.varExplained,
                    varTotal - np.sum(self.varExplained),
                ]
            )
        return

    def transform(self, Ystandard):
        if self.basisTransform is None:
            return Ystandard @ self.basis.T
        else:
            return Ystandard @ self.basisTransform @ self.basis.T 


class pnsBasisConstruct:
    def __init__(
            self,
            resmat,
            PNS,
            radius,
            Y,
    ):
        self.basis = np.zeros((resmat.shape[0], Y.shape[1]))
        self.coefs = resmat.T

        self.PNS = PNS
        self.radius = radius

        varPNS = np.sum(np.abs(resmat) ** 2, axis=1) / Y.shape[0]
        self.varExplained = varPNS

        return
    
    def transform(self, Y):
        n, d = Y.shape
        tt = np.linspace(0, 1, d)
        psi = np.zeros((d, n))
        binsize = np.mean(np.diff(tt))
        for k in range(0, n):
            psi[:, k] = np.sqrt(np.gradient(Y[k, :], binsize))

        pnsdat = psi / np.tile(np.sqrt((psi**2).sum(axis=0)), (d, 1))
        out = fs.pns.PNSs2e(pnsdat, self.PNS)
        return out


# %%
class mvBayes:
    """
    Structure for multivariate response Bayesian model using a basis decomposition.
    """

    def __init__(
        self,
        bayesModel,
        X,
        Y,
        basisType="pca",
        customBasis=None,
        nBasis=None,
        propVarExplained=0.99,
        nCores=1,
        center=True,
        scale=False,
        samplesExtract=None,
        residSDExtract=None,
        idxSamplesArg="idxSamples",
        thresh=1e-15,
        **kwargs,
    ):
        """
        :param bayesModel: Function for fitting a Bayesian model, with `X` the first
            argument and univariate response `y` the second argument. The object
            output by bayesModel (`bm`) needs a `predict` method with required `Xtest`
            argument and an optional argument (specified by `idxSamplesArg`) to down-select
            posterior samples. Optionally, `bm` can also have a `samples` attribute containing
            posterior samples (used for traceplots) and `samples.residSD` containing
            posterior samples of the standard deviation of residuals (optionally used in
            self.predict)
        :param X: numpy array of shape (n, p) continaing predictors (features).
        :param Y: numpy array of shape (n, nMV) containing responses.
        :param basisType: str Type of basis components to use. Options are `pca`, `pns`, 
                              `bspline`, `legendre`, or custom.
        :param nBasis: int number of basis components to use. If None, uses propVarExplained.
        :param propVarExplained: float proportion (between 0 and 1) of variation to
            explain when choosing number of basis functions (if nBasis=None).
        :param nCores: int (<=nBasis) number of threads to use when fitting independent
            Bayesian models.
        :param center: bool whether to center the responses before transformation
            (False not recommended in most cases)
        :param scale: bool whether to scale the responses before transformation
        :param samplesExtract: function taking the output of bayesModel (`bm`) and extracting
            posterior samples of all parameters of interest. If None, mvBayes tries to access
            `bm.samples`; if unsuccessful, an object called `samples` is created with attribute
            `residSD`.
        :param residSDExtract: function taking the output of bayesModel (`bm`) and extracting
            posterior samples of the residual standard deviation (`residSD`). If None, mvBayes
            tries to access `bm.samples.residSD`; if unsuccessful, `residSD` is the  standard
            deviation of the residuals.
        :param idxSamplesArg: str Name of an optional argument of `bm.predict` controlling
            which posterior samples are used for posterior prediction.
        :param thresh: float eigenvalue thresh for pns
        :param kwargs: additional arguments to bayesModel.

        :return: object of class mvBayes, with predict and plot methods.
        """
        self.X = X
        self.Y = Y
        self.nMV = self.Y.shape[1]
        self.bayesModel = bayesModel

        self.basisInfo = basisSetup(
            self.Y, basisType, customBasis, nBasis, propVarExplained, center, scale, thresh
        )

        self.samplesExtract = samplesExtract
        self.residSDExtract = residSDExtract
        self.idxSamplesArg = idxSamplesArg

        self.fit(nCores, **kwargs)

        return

    def fit(self, nCores=1, **kwargs):
        """
        Fit bayesModel for each basis component

        Parameters
        ----------
        nCores : int Number of cores to use when fitting the model
        **kwargs : additional arguments to bayesModel

        Returns
        -------
        None

        """
        nCores = self.nCoresAdjust(nCores)

        def fitBayesModel(k):
            try:
                out = self.bayesModel(self.X, self.basisInfo.coefs[:, k], **kwargs)
                return out
            except Exception as e:
                print(f"Error fitting model {k}: {e}")
                return None

        print(
            "\rStarting mvBayes with {:d} components, using {:d} cores.".format(
                self.basisInfo.nBasis, nCores
            )
        )

        if nCores == 1:
            bmList = [fitBayesModel(k) for k in range(self.basisInfo.nBasis)]
        else:
            with Pool(processes=nCores) as pool:
                bmList = pool.map(fitBayesModel, range(self.basisInfo.nBasis))

        self.bmList = bmList

        self._getSamples()

        self._getResidSD(nCores)

        self.nSamples = len(self.bmList[0].samples.residSD)

        return

    def nCoresAdjust(self, nCores):
        """
        Adjust the number of cores based on availability of 'pathos' module
        and number of available cores (calculated by os.cpu_count())

        Parameters
        ----------
        nCores : TYPE
            DESCRIPTION.

        Returns
        -------
        nCores : TYPE
            DESCRIPTION.

        """
        nCores = min(nCores, self.basisInfo.nBasis)
        nCores_Available = os.cpu_count()
        if nCores > 1 and not PATHOS_AVAILABLE:
            print("Parallel processing module 'pathos' not available. Setting nCores=1.")
            nCores = 1
        elif nCores > nCores_Available:
            print(
                f"Only {nCores_Available} cores are available. Using all available cores."
            )
            nCores = nCores_Available

        return nCores

    def _getSamples(self):
        for k, bm in enumerate(self.bmList):
            if self.samplesExtract is None:
                if not hasattr(bm, "samples"):
                    if k == 0:
                        print(
                            "Generating 'samples' attribute, since it was absent in 'bmList[0]'"
                        )
                    bm.samples = bayesModelSamples()
                    continue
                if isinstance(bm.samples, tuple) and hasattr(bm.samples, "_fields"):
                    bm.samples = bm.samples._asdict()
                if isinstance(bm.samples, dict):
                    newSamples = bayesModelSamples()
                    for key, val in bm.samples.items():
                        setattr(newSamples, key, val)
                    bm.samples = newSamples
            else:
                bm.samples = self.samplesExtract(bm)

        return

    def _getResidSD(self, nCores):
        if self.residSDExtract is None:
            if not hasattr(self.bmList[0].samples, "residSD"):
                print("Approximating 'residSD', since 'residSDExtract' is None")
                _, postCoefs = self.predict(self.X, nCores=nCores, returnPostCoefs=True)
                del _
                for k, bm in enumerate(self.bmList):
                    resid = self.basisInfo.coefs[:, k] - postCoefs[:, :, k]
                    bm.samples.residSD = np.std(resid, axis=1)
        else:
            for k, bm in enumerate(self.bmList):
                bm.samples.residSD = self.residSDExtract(bm)

        return

    def predict(
        self,
        Xtest,
        idxSamples="default",
        addResidError=False,
        addTruncError=False,
        returnPostCoefs=False,
        returnMeanOnly=False,
        nCores=1,
        idxSamplesArg=None,
        **kwargs,
    ):
        """
        Predict the multivariate response at new inputs.

        :param Xtest: numpy array of shape (nTest, p) containing predictors (features).
            p and order of columns must match self.X.
        :param idxSamples: {list, tuple, np.ndarray, "default", "final"} which posterior
            samples to use. Default is to provide no idxSamples argument to bayesModel's
            predict function. "final" equates to [self.nSamples-1]. All inputs are
            converted to np.ndarray before passing to the bayesModel predict method
            ***Only works if bayesModel's predict method has argument idxSamples,
            which is a np.ndarray***
        :param addResidError: bool whether to add in Gaussian error with standard
            deviation residSD when predicting.
        :param addTruncError: bool whether to add in truncation error from the basis
            reconstruction (sampled uniformly from the rows of self.basisInfo.truncError).
        :param nCores: int number of cores to use while predicting. In almost all cases,
            use nCores=1 to avoid conflicts with parallel computation in bayesModel.
        :param idxSamplesArg: str Name of an optional argument of `bm.predict` controlling
        which posterior samples are used for posterior prediction. If None, self.idxSamplesArg
        is used.

        :return: a numpy array of predictions with shape (nSamples, n, nMV), with
            first dimension corresponding to posterior samples, second dimension
            corresponding to observations, and third dimension corresponding to multivariate
            response.
        """
        if idxSamplesArg is None:
            idxSamplesArg = self.idxSamplesArg
        
        if isinstance(idxSamples, str) and idxSamples == "default":
            pass
        elif (
            idxSamplesArg
            not in inspect.signature(self.bmList[0].predict).parameters.keys()
        ):
            print(
                f"'{self.idxSamplesArg}' is not an argument of the bayesModel predict function...setting idxSamples='default'"
            )
            idxSamples = "default"
        else:
            if isinstance(idxSamples, str) and idxSamples == "final":
                idxSamples = np.array([self.nSamples - 1])
            elif isinstance(idxSamples, int):
                idxSamples = np.array([idxSamples])
            elif isinstance(idxSamples, (list, tuple)):
                idxSamples = np.array(idxSamples)
            elif not isinstance(idxSamples, np.ndarray):
                try:
                    idxSamples = np.array(idxSamples)
                except Exception:
                    raise ValueError(
                        "'idxSamples' must be 'default', 'final', np.ndarray or coercible to np.ndarray."
                    )
            kwargs[idxSamplesArg] = idxSamples

        nCores = self.nCoresAdjust(nCores)

        def predictBayesModel(k):
            return self.bmList[k].predict(Xtest, **kwargs)

        if nCores == 1:
            postCoefs = [predictBayesModel(k) for k in range(self.basisInfo.nBasis)]
        else:
            with Pool(processes=nCores) as pool:
                postCoefs = pool.map(predictBayesModel, range(self.basisInfo.nBasis))

        postCoefs = np.dstack(postCoefs)

        if addResidError:
            for k in range(self.basisInfo.nBasis):
                residError = np.random.normal(
                    scale=self.bmList[k].samples.residSD[:, np.newaxis],
                    size=postCoefs.shape[:2],
                )
                postCoefs[:, :, k] += residError

        if self.basisInfo.basisType == "pns":
            PNS = self.basisInfo.basisConstruct.PNS
            N = postCoefs.shape[0] * postCoefs.shape[1]
            nBasis = self.basisInfo.nBasis
            inmat = np.zeros((PNS["radii"].shape[0], N))
            inmat[:nBasis, :] = np.reshape(postCoefs, (N, nBasis), order='F').T
            tmp = fs.pns.fastPNSe2s(
                inmat, self.basisInfo.basisConstruct.PNS
            ) * self.basisInfo.basisConstruct.radius
            YstandardPost = np.reshape(tmp, (postCoefs.shape[0], postCoefs.shape[1], tmp.shape[1]), order='F')
        else:
            YstandardPost = postCoefs @ self.basisInfo.basis
        Ypost = YstandardPost * self.basisInfo.Yscale + self.basisInfo.Ycenter
        del YstandardPost

        if addTruncError:
            idx = np.random.choice(
                self.Y.shape[0], size=np.prod(Ypost.shape[:2]), replace=True
            )
            truncError = self.basisInfo.truncError[idx, :].reshape(Ypost.shape)
            Ypost += truncError

        if returnMeanOnly:
            Ypost = np.mean(Ypost, axis=0)
            postCoefs = np.mean(postCoefs, axis=0)

        if returnPostCoefs:
            return Ypost, postCoefs
        else:
            return Ypost


    def getMSE(
            self,
            resid=None,
            Xtest=None,
            Ytest=None,
            scale=True
        ):
        if resid is None:
            if Xtest is None:
                Xtest = self.X
            if Ytest is None:
                Ytest = self.Y
            
            Ypred = self.predict(Xtest, returnMeanOnly=True)
            resid = Ytest - Ypred
            
        if scale is True:
            resid /= self.basisInfo.Yscale
            
        MSE = np.mean(resid**2)
            
        return MSE
    
    
    def getNegLogLik(self, resid):
        if (
            not hasattr(self.basisInfo.basisConstruct, "basisTransform") or
            self.basisInfo.basisConstruct.basisTransform is None or
            not hasattr(self.basisInfo.basisConstruct, "logDet") or
            self.basisInfo.basisConstruct.logDet is None
        ):
            negLogLike = self.basisInfo.nMV * (np.log(np.mean(resid**2)) + 1)
        else:
            ssResid = [r.T @ self.basisInfo.basisConstruct.basisTransform @ r for r in resid]
            negLogLike = self.basisInfo.basisConstruct.logDet + np.mean(ssResid)
        
        return negLogLike
        
    
    def updateBasis(self, coefsPred, Ystandard, cov=None):
        if coefsPred.shape[1] > Ystandard.shape[1] or coefsPred.shape[0] != Ystandard.shape[0]:
            raise ValueError("X and Y must have the same nrows, ncols of Y must be larger.")
            
        if cov is None:
            basisTransform = None
            logDet = None
            U, _, Vt = np.linalg.svd(coefsPred.T @ Ystandard, full_matrices=False)
            newBasis = U @ Vt
        else:
            D, Q = np.linalg.eigh(cov)
            basisTransform = Q @ np.diag(1/D) @ Q.T
            logDet = np.sum(np.log(D))

            covSqrt = Q @ np.diag(np.sqrt(D)) @ Q.T
            covInvSqrt = Q @ np.diag(np.sqrt(1/D)) @ Q.T
            U, _, Vt = np.linalg.svd(coefsPred.T @ Ystandard @ covInvSqrt, full_matrices=False)
            newBasis = U @ Vt @ covSqrt
        
        self.basisInfo = basisSetup(
            self.Y,
            basisType="custom",
            customBasis=newBasis,
            nBasis=self.basisInfo.nBasis,
            center=self.basisInfo.center,
            scale=self.basisInfo.scale,
            basisTransform=basisTransform
        )
        self.basisInfo.basisConstruct.logDet = logDet
        
        return


    def superDR(self, nIterations=1, covStructure="independent", varEqual=True, nCores=1, **kwargs):
        """
        Run iterative Supervised Dimension Reduction (SuperDR) algorithm to refine basis
        and regression model fit.

        :param nIterations: int number of iterations of the algorithm to run.
        :param corStructure: str correlation structure for the multivariate response.
            Options include "independent", "AR1" and "MA1".
        :param varEqual: bool indicating whether or the residual variance of the 
            response is assumed to be equal across the multivariate index.

        :return: None
        """
        Ystandard = (self.basisInfo._Y - self.basisInfo.Ycenter) / self.basisInfo.Yscale        
        
        Ypred, coefsPred = self.predict(
            self.X,
            returnPostCoefs = True,
            returnMeanOnly = True
        )
        resid = self.basisInfo._Y - Ypred
        
        if not hasattr(self, "MSE"):
            self.MSE = [self.getMSE(resid)]
            
        if not hasattr(self, "negLogLik"):
            self.negLogLik = [self.getNegLogLik(resid)]

        for it in range(nIterations):
            if covStructure == "AR1":
                cov, self.covParam = covAR1(resid, varEqual=varEqual)
            elif covStructure == "MA1":
                cov, self.covParam = covMA1(resid, varEqual=varEqual)
            if covStructure == "independent":
                if varEqual:
                    cov = None
                else:
                    cov = covDiag(resid)
                self.covParam = None
                
            self.updateBasis(
                coefsPred,
                Ystandard,
                cov
            )
        
            self.fit(nCores, **kwargs)
            
            Ypred, coefsPred = self.predict(
                self.X,
                returnPostCoefs = True,
                returnMeanOnly = True
            )
            resid = self.basisInfo._Y - Ypred
            
            self.MSE = np.concatenate([self.MSE, [self.getMSE(resid)]])
            self.negLogLik = np.concatenate([self.negLogLik, [self.getNegLogLik(resid)]])
        
        return


    def traceplot(self, modelParams=None, labels=None, title=None, file=None, **kwargs):
        """
        Trace plots of model parameters

        :param modelParams: str or list of strings specifying names of model parameters
            to plot. These should be attributes of `samples` from the bayesModel object.
            If None, selects "plottable" attributes of `samples`, including residSD.
        :param labels: string or list of strings labeling each model parameter.
            Default is to use modelParams.
        :param file: str specifying a file path to save the plot. Default
            is not to save but to instead call plt.show().

        :return: None
        """
        bmList = self.bmList
        nBasis = self.basisInfo.nBasis

        if modelParams is None:

            def isModelParam(obj):
                if (
                    obj is None
                    or callable(obj)
                    or isinstance(obj, (str, bytes, bytearray))
                ):
                    return False
                try:
                    array = np.array(obj)
                    return array.ndim in [0, 1]
                except Exception:
                    return False

            modelParams = [
                attr
                for attr in dir(bmList[0].samples)
                if (
                    not (attr.startswith("__") and attr.endswith("__"))
                    and isModelParam(getattr(bmList[0].samples, attr))
                )
            ]
        elif isinstance(modelParams, str):
            modelParams = [modelParams]

        if isinstance(labels, str):
            labels = [labels]
        elif labels is None:
            labels = modelParams

        nParams = len(modelParams)
        if nParams > 8:
            print("Currently, must have len(modelParams) <= 8. Plotting the first 8.")
            modelParams = modelParams[:8]
            labels = labels[:8]
            nParams = 8

        nrow = int(np.ceil(nParams / 2))
        if nParams == 1:
            ncol = 1
        else:
            ncol = 2

        fig = plt.figure(figsize=(8, 6))
        cmap = plt.get_cmap("tab20")

        for j in range(nParams):
            fig.add_subplot(nrow, ncol, j + 1)
            for k in range(nBasis):
                obj = bmList[k].samples
                try:
                    param = getattr(obj, modelParams[j])
                except AttributeError:
                    raise Exception(f"No attribute named {modelParams[j]}")
                plt.plot(param, color=cmap(k % 20))
            plt.ylabel(labels[j])
            plt.xlabel("MCMC iteration")

        if title is not None:
            fig.suptitle(title)
        fig.tight_layout()

        if file is None:
            plt.show()
        else:
            plt.savefig(file, **kwargs)

        plt.close(fig)
        return

    def plot(
        self,
        Xtest=None,
        Ytest=None,
        idxSamples='final',
        nPlot=None,
        idxMV=None,
        xscale='linear',
        xlabel='Multivariate Index',
        title=None,
        file=None,
        **kwargs,
    ):
        """
        Plot the model fit for Xtest (or self.X if Xtest is None)
        * top left - Residuals (black) overlayed on centered response (slate blue)
        * top right - basis components scaled by residuals of coefficients. Colors
            correspond to the bottom plots.
        * bottom left - R-squared for predicting each basis component.
        * bottom right - % Residual variance explained by each basis component,
            including truncation error components stacked on the right (grey).

        Parameters
        ----------
        Xtest : numpy array of shape (nTest, p) containing predictors for which
            predictions will be obtained and plots will be made. If None,
            Xtest is self.X
        Ytest : numpy array of shape (nTest, nMV) containing responses for which
            accuracy will be assessed and plots will be made. If None, Ytest is self.Y.
            A warning is provided if only one of Xtest and Ytest is None (not recommended
            in most cases)
        :param nPlot: int number of observations to plot. If None, uses min(nTest, 1000)
        :param file: str where to save the file. Shown, not saved, if None.
        :param title: str title for the top of the plot. If None, no title is provided.
        :return: None

        """
        useXtrain = Xtest is None
        useYtrain = Ytest is None
        if useXtrain:
            Xtest = self.X.copy()
        elif useYtrain:
            print(
                "Model output at user-specified Xtest is being compared to training responses Y."
            )

        if useYtrain:
            Ytest = self.basisInfo._Y.copy()
            coefs = self.basisInfo.coefs
            truncError = self.basisInfo.truncError
        else:
            if useXtrain:
                print(
                    "Model output at training inputs X is being compared to user-specified Ytest."
                )

            coefs = self.basisInfo.getCoefs(Ytest)
            Ytest = self.basisInfo.preprocessY(Ytest)
            Ytrunc = self.basisInfo.getYtrunc(coefs=coefs)
            truncError = Ytest - Ytrunc

        if nPlot is None:
            nPlot = min(Xtest.shape[0], 1000)
        elif nPlot > Xtest.shape[0]:
            print(
                "nPlot should be at most n=len(Xtest) (or n=len(X) if Xtest is None). Using nPlot=n."
            )
            nPlot = Xtest.shape[0]
        idxPlot = np.random.choice(Xtest.shape[0], nPlot, replace=False)

        if idxMV is None:
            idxMV = list(range(self.basisInfo.nMV))

        if self.basisInfo.basisType == "pns":
            Ycentered = Ytest - Ytest.mean(axis=0)
        else:
            Ycentered = Ytest - self.basisInfo.Ycenter

        # Get predictions and residuals
        Ypred, coefsPred = self.predict(
            Xtest, returnPostCoefs=True, returnMeanOnly=True, idxSamples=idxSamples
        )  # posterior predictive draws
        R = Ytest - Ypred  # residuals
        RbasisCoefs = coefs - coefsPred  # residuals of coefficients

        # Create plot
        fig = plt.figure(figsize=(8, 6))
        cmap = plt.get_cmap("tab20")

        # Plot Residuals on top of Y
        fig.add_subplot(2, 2, 1)
        mseOverall = self.getMSE(R)
        if not self.basisInfo.center and not self.basisInfo.scale:
            legendLab = "Y"
        else:
            legendLab = "Y (Centered)"
        plt.plot(idxMV, Ycentered[idxPlot].T, color=(0.7, 0.7, 1.0), alpha=0.5)
        plt.plot(idxMV, R[idxPlot].T, color="black", alpha=0.5)
        plt.plot(
            [], [], color="black", alpha=0.5, label="Residual", linewidth=2
        )  # for legend label
        plt.plot(
            [], [], color=(0.7, 0.7, 1.0), alpha=0.5, label=legendLab, linewidth=2
        )  # for legend label
        plt.xlim(np.min(idxMV), np.max(idxMV))
        plt.xscale(xscale)
        plt.ylabel("Residuals")
        plt.xlabel(xlabel)
        plt.title(f"Overall MSE = {mseOverall:.4g}")
        plt.legend()

        # Plot each basis, scaled by residuals
        fig.add_subplot(2, 2, 2)
        RbasisScaled = []
        mseBasis = np.zeros(self.basisInfo.nBasis)
        varBasis = np.zeros(self.basisInfo.nBasis)
        if self.basisInfo.basisType == "pns":
            mseTrunc = np.sum(self.basisInfo.varExplained[self.basisInfo.nBasis : self.basisInfo.nMV])
            for k in range(self.basisInfo.nBasis):
                PNS = self.basisInfo.basisConstruct.PNS
                N = coefsPred.shape[0]
                inmat = np.zeros((PNS["radii"].shape[0], N))
                inmat[k, :] = coefsPred[:, k]
                tmp = fs.pns.fastPNSe2s(
                    inmat, self.basisInfo.basisConstruct.PNS
                ) * self.basisInfo.basisConstruct.radius
            
                RbasisScaled.append(Ytest - tmp)
                mseBasis[k] = np.mean(RbasisCoefs[:, k] ** 2)
                varBasis[k] = np.mean(coefs[:, k] ** 2)
        else:
            mseTrunc = np.sum(self.basisInfo.varExplained[self.basisInfo.nBasis : self.basisInfo.nMV])*(Ytest.shape[0]-1)/Ytest.shape[0]
            for k in range(self.basisInfo.nBasis):
                RbasisScaled.append(
                    np.outer(RbasisCoefs[:, k], self.basisInfo.basis[k, :])
                    * self.basisInfo.Yscale
                )
                mseBasis[k] = np.mean(RbasisCoefs[:, k] ** 2)

                basisScaled = (
                    np.outer(coefs[:, k], self.basisInfo.basis[k, :])
                    * self.basisInfo.Yscale
                )
                varBasis[k] = self.basisInfo.varExplained[k]*(Ytest.shape[0]-1)/(Ytest.shape[0])
        mseOrder = np.argsort(mseBasis)[::-1]

        plt.plot(idxMV, truncError[idxPlot].T, color="silver", alpha=0.5)
        for k in mseOrder:
            plt.plot(idxMV, RbasisScaled[k][idxPlot].T, color=cmap(k % 20), alpha=0.5)
        plt.xlim(np.min(idxMV), np.max(idxMV))
        plt.xscale(xscale)
        plt.ylabel("Basis Projection")
        plt.xlabel(xlabel)

        # R^2 plot
        fig.add_subplot(2, 2, 3)
        plotColors = [cmap(k % 20) for k in range(self.basisInfo.nBasis)]

        r2Basis = 1 - mseBasis / varBasis
        varOverall = np.sum(self.basisInfo.varExplained)*(Ytest.shape[0]-1)/Ytest.shape[0]
        r2Overall = 1 - mseOverall / varOverall

        plt.scatter(
            range(1, self.basisInfo.nBasis + 1), r2Basis, color=plotColors, s=50
        )
        plt.xlabel("Component")
        plt.xticks(
            range(1, self.basisInfo.nBasis + 1),
            list(range(1, self.basisInfo.nBasis + 1)),
        )
        plt.ylabel("$R^2$")
        plt.title(f"Overall $R^2$ = {r2Overall:.3g}")
        plt.axhline(y=r2Overall, linestyle="--", color="silver")
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

        # Residual variance plot
        fig.add_subplot(2, 2, 4)
        if hasattr(self.basisInfo.basisConstruct, "basisTransformSqrt"):
            basisTransformSqrt = self.basisInfo.basisConstruct.basisTransformSqrt
            mseOverall = np.mean((R @ basisTransformSqrt)**2, axis=0).sum()
            for k in range(self.basisInfo.nBasis):
                mseBasis[k] = np.mean((RbasisScaled[k] @ basisTransformSqrt)**2, axis=0).sum()
            mseTrunc = np.mean((truncError @ basisTransformSqrt)**2, axis=0).sum()

        mseTruncProp = self.basisInfo.varExplained[
            self.basisInfo.nBasis : self.basisInfo.nMV
        ] / np.sum(
            self.basisInfo.varExplained[self.basisInfo.nBasis:self.basisInfo.nMV]
        )
        mseTruncCS = np.cumsum(mseTruncProp * mseTrunc / mseOverall)
        mseExplained = mseBasis / mseOverall
        plt.scatter(
            range(1, self.basisInfo.nBasis + 1),
            100 * mseExplained,
            color=plotColors,
            s=50,
        )
        plt.scatter(
            np.repeat(
                self.basisInfo.nBasis + 1,
                len(self.basisInfo.varExplained) - self.basisInfo.nBasis,
            ),
            100 * mseTruncCS,
            color="silver",
            s=50,
        )
        plt.xlabel("Component")
        plt.ylabel("%Residual Variance")
        plt.xticks(
            range(1, self.basisInfo.nBasis + 2),
            list(range(1, self.basisInfo.nBasis + 1)) + ["T"],
        )

        # Finishing touches
        if title is not None:
            fig.suptitle(title)
        fig.tight_layout()

        if file is None:
            plt.show()
        else:
            plt.savefig(file, **kwargs)

        plt.close(fig)

        return

    def mvSobol(
        self, totalSobol=True, idxSamples="final", nMC=None, showPlot=False, **kwargs
    ):
        """
        Compute Sobol' indices via Monte Carlo sampling (except in special cases
        where a closed-form solution is available).

        Parameters
        ----------
        totalSobol : bool whether to compute the total Sobol' index (summing all interactions)
            in addition to the first-order Sobol' index
        :param idxSamples: {list, tuple, np.ndarray, "default", "final"} which posterior
            samples to use to compute Sobol' indices. Default is to use only the final sample.
            This is passed to self.predict (see self.predict documentation for details)
            and is ONLY available if the bayesModel object has a predict method with argument
            idxSamples of type np.ndarray
        nMC : int number of Monte Carlo samples to take in computing the Sobol' indices.
            Default is to use 2**12.
        showPlot : bool whether to call self.plotSobol after computing the indices
        **kwargs : additional arguments to the bayesModel object's predict method

        Returns
        -------
        None.

        """
        p = self.X.shape[1]

        if self.basisInfo.basisType == "pns" and nMC is None:
            nMC = 2**12

        useBASS = False
        if self.bayesModel.__name__ == "bass" and nMC is None:
            if isinstance(self, mvBayesElastic):
                print(
                    "mvBayesElastic does not support closed-form Sobol' indices. Using Monte Carlo."
                )
            elif not PYBASS_AVAILABLE:  # tried to use BASS, but not available
                print(
                    "Module 'pyBASS' not available. Using Monte Carlo to Compute Sobol' indices."
                )
            else:
                useBASS = True

        if useBASS:
            # Generate class mimicking pyBass.BassBasis
            class BassBasis:
                def __init__(self, xx, basis, nbasis, bm_list):
                    self.xx = xx
                    self.basis = basis
                    self.nbasis = nbasis
                    self.bm_list = bm_list
                    return

            # Generate BassBasis object
            myBassBasis = BassBasis(
                self.X,
                self.basisInfo.basis.T,
                self.basisInfo.nBasis,
                self.bmList,
            )

            if totalSobol:
                maxOrder = min(p, self.bmList[0].prior.maxInt)
            else:
                maxOrder = 1

            if isinstance(idxSamples, str) and idxSamples == "final":
                idxSamples = [self.nSamples - 1]
            elif isinstance(idxSamples, str) and idxSamples == "default": # use all posterior samples
                idxSamples = list(range(self.nSamples))
            elif type(idxSamples) is np.ndarray:
                idxSamples = list(idxSamples.astype(int))
            elif type(idxSamples) is int:
                idxSamples = [idxSamples]
            nSamples = len(idxSamples)

            # Compute Sobol' Indices
            sobol = pyBASS.sobolBasis(myBassBasis)
            firstOrder = np.zeros((nSamples, p, self.basisInfo.nMV))
            if totalSobol:
                totalOrder = np.zeros((nSamples, p, self.basisInfo.nMV))
            else:
                totalOrder = None
            varTotal = np.zeros((nSamples, self.basisInfo.nMV))
            for idx in range(nSamples):
                sobol.decomp(maxOrder, mcmc_use=idxSamples[idx])
                firstOrder[idx] = sobol.S_var[:p].copy()
                if totalSobol:
                    totalOrder[idx] = sobol.T_var.copy()
                varTotal = sobol.S_var[0] / sobol.S[0]

            self.firstOrderSobol = firstOrder
            if totalSobol:
                self.totalOrderSobol = totalOrder
            self.varTotal = np.max(
                np.vstack([varTotal, np.sum(np.mean(firstOrder, axis=0), axis=0)]),
                axis=0,
            )
        else:
            if nMC is None:
                nMC = 2**12

            # Generate random samples of parameters according to Saltelli
            # (2010) method. I coded it myself b/c other methods are
            # insufficient. SALib only works for univariate output and is
            # unstable. scipy gives first- and total-order Sobol' indices
            # normalized by total variance, so total variance is unknown.
            qrng = qmc.Sobol(d=2 * p, scramble=True)
            baseSequence = qrng.random(nMC)
            A = baseSequence[:, :p].copy()
            B = baseSequence[:, p:(2 * p)].copy()
            del baseSequence
            AB = np.zeros([p * nMC, p])
            for j in range(p):
                idx = np.delete(np.arange(p), j)
                AB[(j * nMC):((j + 1) * nMC), idx] = A[:, idx].copy()
                AB[(j * nMC):((j + 1) * nMC), j] = B[:, j].copy()
            saltelliSequence = np.vstack([A, B, AB])
            del A
            del B
            del AB

            xmin = np.min(self.X, axis=0)
            xrange = np.max(self.X, axis=0) - xmin
            saltelliSequence *= xrange
            saltelliSequence += xmin

            # Evaluate model at those param values
            saltelliMC = self.predict(saltelliSequence, idxSamples=idxSamples, **kwargs)
            nSamples = saltelliMC.shape[0]

            # Transform the samples
            saltelliMC = self.transformSaltelliMC(saltelliMC)

            # Estimate Sobol' Indices
            modA = saltelliMC[:, :nMC, :].copy()
            modB = saltelliMC[:, nMC : (2 * nMC), :].copy()
            modAB = [
                saltelliMC[:, ((2 + j) * nMC) : ((3 + j) * nMC), :].copy()
                for j in range(p)
            ]

            varTotal = np.var(saltelliMC, axis=1)
            del saltelliMC

            firstOrder = np.zeros((nSamples, p, self.nMV))
            if totalSobol:
                totalOrder = np.zeros((nSamples, p, self.nMV))
            else:
                totalOrder = None
            for j in range(p):
                firstOrder[:, j, :] = np.mean(modB * (modAB[j] - modA), axis=1)

                if totalSobol:
                    totalOrder[:, j, :] = 0.5 * np.mean((modA - modAB[j]) ** 2, axis=1)

            self.firstOrderSobol = firstOrder
            if totalSobol:
                self.totalOrderSobol = totalOrder
            self.varTotal = np.max(
                np.vstack([varTotal, np.sum(np.mean(firstOrder, axis=0), axis=0)]),
                axis=0,
            )

        if showPlot:
            self.plotSobol(totalSobol=totalSobol)

        return

    def transformSaltelliMC(self, saltelliMC):
        """
        Factory method to transform Monte Carlo samples. For basisSetup, just
        returns the saltelliMC argument

        Parameters
        ----------
        saltelliMC : np.ndarray containing posterior samples of predictions at
            Saltelli sample locations

        Returns
        -------
        saltelliMC : np.ndarray (same as input argument)

        """

        meanS = np.mean(saltelliMC, axis=1)
        saltelliMC -= meanS

        return saltelliMC

    def plotSobol(
        self,
        totalSobol=True,
        labels=None,
        idxMV=None,
        xscale='linear',
        xlabel='Multivariate Index',
        yOverlay=None,
        yOverlayLabel="Overlay",
        waterfall=False,
        title=None,
        file=None,
        **kwargs,
    ):
        """
        Make plots of the Sobol' indices previously calculated by self.sobol.
        * left - First-order Relative Sensitivity, i.e., first-order Sobol'
            index normalized to sum to one across predictors at each multivariate
            index
        * center - First-order Sensitivity, i.e., first-order Sobol' index on
            original variance scale
        * right - Total Sensitivity, i.e., total Sobol' index, summing interactions
            for each predictor. Only shown if totalSobol is True

        Parameters
        ----------
        totalSobol : bool whether to plot total Sobol' indices in addition to
            first-order indices (only works if totalSobol was true when self.sobol)
            was called.
        labels : list of strings labeling the input parameters
        idxMV : list or np.array of length nMV containing indices for the x-axis
            (e.g., time for functional data). If None, list(range(self.nMV)) is used
        yOverlay : list or np.ndarray of length nMV containing a multivariate
            response to overlay on top of the Sobol' plots (e.g., self.Y.mean(axis=0))
        waterfall: bool whether to plot sobol as a waterfall (functional pie-chart)
                   default (False)

        Returns
        -------
        None.

        """
        if not hasattr(self, "firstOrderSobol"):
            raise Exception(
                "Sobol' indices have not been computed. Need to run sobol before plotSobol."
            )

        p = np.shape(self.X)[1]

        if idxMV is None:
            idxMV = list(range(self.nMV))

        if labels is None:
            labels = ["X" + str(j + 1) for j in range(p)]

        lty = np.resize(["solid", "dashed", "dotted", "dashdot"], len(labels))
        lty = list(lty) + ["solid"]
                    
        firstOrder = np.mean(self.firstOrderSobol, axis=0)  # posterior mean
        firstOrderRel = firstOrder / self.varTotal  # posterior mean

        # Set up plot
        fig, axs = plt.subplots(1, 2 + totalSobol, figsize=(9, 5))

        # plot first order relative sobol' indices
        map = cm.Paired(np.linspace(0, 1, 12))
        map = np.resize(map, (p, 4))
        rgb = np.ones((p + 1, 4))
        rgb[:p] = map
        del map
        rgb[-1, :3] = np.array([153, 153, 153]) / 255
        labels = list(labels) + ["Higher-Order"]
        if waterfall:
            idxMV = np.array(idxMV)
            ord = idxMV.argsort()
            meanX = np.vstack(
                [
                    firstOrderRel,
                    1.0 - np.sum(firstOrderRel, axis=0),
                ]
            )

            sens = np.cumsum(meanX, axis=0).T

            for j in range(p + 1):
                x2 = np.concatenate((idxMV[ord], np.flip(idxMV[ord])))
                if j == 0:
                    inBetween = np.concatenate(
                        (np.zeros(idxMV[ord].shape[0]), np.flip(sens[ord, j]))
                    )
                else:
                    inBetween = np.concatenate(
                        (sens[ord, j - 1], np.flip(sens[ord, j]))
                    )

                axs[0].fill(x2, inBetween, color=rgb[j], label=labels[j])
        else:
            for j in range(p):
                axs[0].plot(
                    idxMV,
                    firstOrderRel[j],
                    linewidth=3,
                    linestyle=lty[j],
                    color=rgb[j],
                    label=labels[j],
                )
            axs[0].plot(
                idxMV,
                1.0 - np.sum(firstOrderRel, axis=0),
                linewidth=3,
                color=rgb[j],
                label=labels[p],
            )

        axs[0].set(
            xlabel=xlabel,
            ylabel="Relative First-Order Sobol' Index",
            title="First-Order Relative Sensitivity",
            ylim=[0, 1],
            xlim=[np.min(idxMV), np.max(idxMV)],
            xscale=xscale,
        )

        if yOverlay is not None:
            ax2 = axs[0].twinx()
            ax2.plot(idxMV, yOverlay, color="black", linestyle="--", alpha=0.7)

        # plot first order sobol' indices
        if waterfall:
            sens_var = np.vstack([np.cumsum(firstOrder, axis=0), self.varTotal]).T

            for j in range(p + 1):
                x2 = np.concatenate((idxMV[ord], np.flip(idxMV[ord])))
                if j == 0:
                    inBetween = np.concatenate(
                        (np.zeros(idxMV[ord].shape[0]), np.flip(sens_var[ord, j]))
                    )
                else:
                    inBetween = np.concatenate(
                        (sens_var[ord, j - 1], np.flip(sens_var[ord, j]))
                    )

                axs[1].fill(x2, inBetween, color=rgb[j], label=labels[j])
            axs[1].set_ylim(0, inBetween.max() + 3)
        else:
            for j in range(p):
                axs[1].plot(
                    idxMV,
                    firstOrder[j],
                    linewidth=3,
                    linestyle=lty[j],
                    color=rgb[j],
                    label=labels[j],
                )
            axs[1].plot(
                idxMV,
                self.varTotal - np.sum(firstOrder, axis=0),
                linewidth=3,
                color=rgb[p],
                label=labels[p],
            )
            axs[1].set_ylim(0, np.max(firstOrder) * 1.05)

        axs[1].set(
            xlabel=xlabel,
            ylabel="First-Order Sobol' Index",
            title="First-Order Sensitivity",
            xlim=[np.min(idxMV), np.max(idxMV)],
            xscale=xscale,
        )
            
        lines, labels = axs[1].get_legend_handles_labels()
        if yOverlay is not None:
            ax2 = axs[1].twinx()
            ax2.plot(idxMV, yOverlay, color="black", linestyle="--", alpha=0.7, label=yOverlayLabel)
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines += lines2
            labels += labels2
        axs[1].legend(lines, labels, loc="upper left")
        
        # plot total order sobol' indices
        if totalSobol:
            if not hasattr(self, "totalOrderSobol"):
                print(
                    "Total Order Sobol' indices have not been computed and will not be plotted."
                )
            else:
                totalOrder = np.mean(self.totalOrderSobol, axis=0)  # posterior mean
                for j in range(p):
                    axs[2].plot(
                        idxMV,
                        totalOrder[j],
                        linewidth=3,
                        linestyle=lty[j],
                        color=rgb[j],
                        label=labels[j],
                    )
                axs[2].set(
                    xlabel=xlabel,
                    ylabel="Total-Order Sobol' Index",
                    title="Total Sensitivity",
                    ylim=[0, np.max(totalOrder) * 1.05],
                    xlim=[np.min(idxMV), np.max(idxMV)],
                    xscale=xscale,
                )

        if yOverlay is not None:
            ax2 = axs[1 + totalSobol].twinx()
            ax2.plot(idxMV, yOverlay, color="black", linestyle="--", alpha=0.7)

        if title is not None:
            fig.suptitle(title)
        fig.tight_layout()

        if file is None:
            plt.show()
        else:
            plt.savefig(file, **kwargs)

        plt.close(fig)
        return


class bayesModelSamples:  # empty samples class
    pass
        

# %% Elastic FDA Basis setup
class basisSetupElastic(basisSetup):
    """
    Compute basis components for a matrix Y, using the elastic functional
    data analysis framework. Used in mvBayesElastic.
    """

    def __init__(
        self,
        Y,
        warpData=None,
        basisType="jfpcah",
        nBasis=None,
        propVarExplained=0.99,
    ):
        """
        :param Y: numpy array of shape (n, nMV) for which a reduced-dimension
            basis will be computed.
        :param warpData: used to generate joint aligned/warping function basis
        :param basisType: str stating basis type. Options are 'jfpca', 'jfpcah', or 'pns'
        :param nBasis: int number of basis components to use (optional).
        :param propVarExplained: float proportion (between 0 and 1) of variation to explain
                                 when choosing number of basis components (if nBasis=None).
        :return: object with plot method.
        """
        if not FDASRSF_AVAILABLE:
            raise Exception(
                "Module 'fdasrsf' not available. basisSetupElastic cannot proceed"
            )

        if warpData is None:
            print("warpData not provided. Computing default warpData from Y")
            idxMV = np.linspace(0.0, 1.0, Y.shape[1])
            warpData = fs.fdawarp(Y.T, idxMV)
            warpData.srsf_align()

        self.warpData = warpData
        self.basisType = basisType
        self.Yscale = 1.0
        self.center = True
        self.scale = False

        if basisType == "jfpca":
            basisConstruct = fs.fdajpca(warpData)
            basisConstruct.calc_fpca(no=warpData.time.shape[0], parallel=True)
            basis = basisConstruct.U.T
            self.varExplained = basisConstruct.latent
            self.Ycenter = basisConstruct.mu_g
            self.YjointElastic = basisConstruct.g.T
        elif basisType == "jfpcah":
            if nBasis is not None:
                propVarExplained = 0.999  # would set to 1.0, this breaks fdasrsf
            basisConstruct = fs.fdajpcah(warpData)
            basisConstruct.calc_fpca(
                var_exp=propVarExplained, parallel=True, srsf=False
            )
            no_q = basisConstruct.U.shape[1]
            Psi_q = basisConstruct.U @ basisConstruct.Uz[0:no_q, :]
            Psi_h = basisConstruct.U1 @ basisConstruct.Uz[no_q:, :]
            basis = np.vstack((Psi_q, Psi_h)).T
            del Psi_q
            del Psi_h
            self.varExplained = basisConstruct.eigs
            self.Ycenter = np.concatenate((basisConstruct.mqn, basisConstruct.mh))
            self.YjointElastic = np.concatenate(
                (basisConstruct.qn1, basisConstruct.C * basisConstruct.h)
            ).T
            nBasis = basis.shape[0]
        else:
            raise Exception("un-supported basisType")

        propVarCumSum = np.cumsum(self.varExplained) / np.sum(self.varExplained)
        if nBasis is None:
            nBasis = np.where(propVarCumSum > propVarExplained)[0][0] + 1
        elif nBasis > len(self.varExplained):
            nBasis = len(self.varExplained)
            print(
                f"User-specified 'nBasis' larger than the number of jfpcah basis. Setting nBasis={nBasis}."
            )

        self.nBasis = nBasis
        self.propVarExplained = propVarCumSum[nBasis - 1]

        self.basisConstruct = basisConstruct
        coefs = basisConstruct.coef
        self.nMV = self.YjointElastic.shape[1]

        self.basis = basis[: self.nBasis, :]
        self.coefs = coefs[:, : self.nBasis]
        YjointElasticTrunc = self.getYtrunc()
        self.truncError = self.YjointElastic - YjointElasticTrunc

        return

    @property
    def _Y(self):
        """
        Creates a common way to extract object on which basis is computed.
        For basisSetupElastic, call self._Y or self.YjointElastic
        """
        return self.YjointElastic

    def getCoefs(self, Ytest=None):
        """
        Transform Ytest into basis coefficients

        Parameters
        ----------
        Ytest : numpy array of shape (n, nMV) for which basis coefficients will
            be computed. If Ytest is None, uses self.Y.

        Returns
        -------
        coefs
            numpy array of shape (n, self.nBasis) containing basis coefficients
            corresponding to Ytest.

        """
        if Ytest is None:
            return self.coefs
        else:
            self.basisConstruct.project(Ytest.T)
            return self.basisConstruct.new_coef[:, : self.nBasis]
    
    def getYtrunc(self, Ytest=None, coefs=None, nBasis=None):
        """
        Get a "truncated" reconstruction of Ytest, reconstructed from the first nBasis
        basis components. If Ytest is provided, coefs will be computed. If neither is provided,
        self.coefs will be used. coefs is then used to get the truncated reconstruction.

        Parameters
        ----------
        Ytest : numpy array of shape (n, nMV) for which a truncated reconstruction
            will be computed. If None, uses coefs.
        coefs : numpy array of shape (n, self.nBasis) used to create a truncated
            reconstruction. If None, uses Ytest to compute coefs. If Ytest is also
            None, uses self.coefs.
        nBasis : integer indicating the number of basis components to use in the
            truncation. Max is self.nBasis. If None, uses self.nBasis.

        Returns
        -------
        Ytrunc
            numpy array of shape (n, nMV) containing the truncated reconstruction.

        """
        if coefs is None:
            coefs = self.getCoefs(Ytest)
        if nBasis is None or nBasis > self.nBasis:
            nBasis = self.nBasis
        if self.basisType == "jfpcah":
            no_q = self.basisConstruct.U.shape[1]
            Psi_q = self.basisConstruct.U @ self.basisConstruct.Uz[0:no_q, :]
            Psi_h = self.basisConstruct.U1 @ self.basisConstruct.Uz[no_q:, :]
            qhat = coefs @ Psi_q.T
            hhat = coefs @ Psi_h.T
            YtruncStandard = np.concatenate((qhat, hhat), axis=1)
        else:
            YtruncStandard = np.dot(coefs[:, :nBasis], self.basis[:nBasis, :])
        return YtruncStandard * self.Yscale + self.Ycenter

    def preprocessY(self, Ytest=None, projectNew=False):
        """
        Method to perform preprocessing on Y before doing the basis
        decomposition. For basisSetupElastic, this involves computing
        aligned and warping functions, scaling, and concatenating.

        Parameters
        ----------
        Ytest : numpy array of shape (n, nMV) on which preprocessing will
        be done. If Ytest is None, uses self.Y.

        Returns
        -------
        numpy array of shape (n, nMV) containing the preprocessed version of Ytest

        """
        if Ytest is None:
            return self.YjointElastic
        else:
            if projectNew or not hasattr(self.basisConstruct, "new_coef"):
                self.basisConstruct.project(Ytest.T)

            if self.basisType == "jfpca":
                YtestPreprocessed = self.basisConstruct.new_g.T
            elif self.basisType == "jfpcah":
                YtestPreprocessed = np.concatenate(
                    (
                        self.basisConstruct.new_qn1,
                        self.basisConstruct.C * self.basisConstruct.new_h,
                    )
                ).T

            return YtestPreprocessed


# %% Elastic FDA mvBayes
class mvBayesElastic(mvBayes):
    """
    Structure for functional/multivariate response Bayes model using a
    basis decomposition. Performs elastic functional data analysis
    before calculating a lower-dimensional reconstruction.
    """

    def __init__(
        self,
        bayesModel,
        X,
        Y,
        warpData=None,
        basisType="jfpcah",
        nBasis=None,
        propVarExplained=0.99,
        nCores=1,
        samplesExtract=None,
        residSDExtract=None,
        idxSamplesArg="idxSamples",
        **kwargs,
    ):
        """
        :param bayesModel: Function for fitting a Bayesian model, with `X` the
            first argument and univariate response `y` the second argument.
            Optionally, the object output by bayesModel can have a `samples`
            attribute containing posterior samples (used for traceplots) and
            `samples.residSD` containing posterior samples of the standard
            deviation of residuals (optionally used in self.predict).
            bayesModel also needs a `predict` method with required `Xtest`
            argument and optional idxSamples argument to down-select posterior
            samples.
        :param X: numpy array of shape (n, p) continaing predictors (features).
        :param Y: numpy array of shape (n, nMV) containing responses.
        :param warpData: fs.fdawarp object to generate joint aligned/warping
                         function basis. If None, this is calculated using
                         default settings and aligning to the Karcher mean.
        :param basisType: str Type of basis components to use. Options are
                              `jfpca` or `jfpcah
        :param nBasis: int number of basis components to use. If None, uses
                           propVarExplained.
        :param propVarExplained: float proportion (between 0 and 1) of
                                       variation to explain when choosing
                                       number of basis functions
                                       (if nBasis=None).
        :param nCores: int (<=nBasis) number of threads to use when fitting
                                      independent Bayesian models.
        :param kwargs: additional arguments to bayesModel.

        :return: object of class mvBayes, with predict and plot functions.
        """
        self.X = X
        self.Y = Y
        self.nMV = self.Y.shape[1]
        self.bayesModel = bayesModel

        basisInfo = basisSetupElastic(Y, warpData, basisType, nBasis, propVarExplained)
        self.basisInfo = basisInfo

        self.samplesExtract = samplesExtract
        self.residSDExtract = residSDExtract
        self.idxSamplesArg = idxSamplesArg

        self.fit(nCores, **kwargs)

        return

    def transformSaltelliMC(self, saltelliMC):
        """
        Method to transform Monte Carlo samples. For basisSetupElastic, this
        computes samples of the aligned curves given samples of the aligned and
        warping functions.

        Parameters
        ----------
        saltelliMC : np.ndarray containing posterior samples of predictions at
            Saltelli sample locations

        Returns
        -------
        saltelliMC : np.ndarray with shape matching self.Y rather than 
                                self.basisInfo.Y

        """
        C = self.basisInfo.basisConstruct.C
        nSamples = saltelliMC.shape[0]
        N = saltelliMC.shape[1]

        if self.basisInfo.nMV % 2 == 0:
            M = int(self.basisInfo.nMV / 2)
            time = np.linspace(0, 1, M)
            postSamples = np.zeros((nSamples, N, M))
            for ii in range(nSamples):
                if self.basisInfo.basisType == "jfpca":
                    gamtmp = fs.geometry.v_to_gam((saltelliMC[ii, :, M:] / C).T)
                elif self.basisInfo.basisType == "jfpcah":
                    gamtmp = fs.geometry.h_to_gam((saltelliMC[ii, :, M:] / C).T)
                for jj in range(N):
                    ftmp = saltelliMC[ii, jj, :M]
                    postSamples[ii, jj, :] = fs.warp_f_gamma(time, ftmp, gamtmp[:, jj])
        else:
            M = int((self.basisInfo.nMV - 1) / 2)
            time = np.linspace(0, 1, M)
            postSamples = np.zeros((nSamples, N, M))
            mididx = self.basisInfo.basisConstruct.id
            for ii in range(nSamples):
                if self.basisInfo.basisType == "jfpca":
                    gamtmp = fs.geometry.v_to_gam((saltelliMC[ii, :, (M + 1):] / C).T)
                elif self.basisInfo.basisType == "jfpcah":
                    gamtmp = fs.geometry.h_to_gam((saltelliMC[ii, :, (M + 1):] / C).T)
                for jj in range(N):
                    ftmp = saltelliMC[ii, jj, : (M + 1)]
                    ftmp = fs.utility_functions.cumtrapzmid(
                        time,
                        ftmp[0:M] * np.fabs(ftmp[0:M]),
                        np.sign(ftmp[M]) * (ftmp[M] * ftmp[M]),
                        mididx,
                    )
                    postSamples[ii, jj, :] = fs.warp_f_gamma(time, ftmp, gamtmp[:, jj])

        return postSamples
