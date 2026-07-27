import numpy as np
import os
import itertools
from .mvBayes import mvBayes

try:
    from joblib import Parallel, delayed
    joblibAvailable = True
except Exception:
    joblibAvailable = False


class mvBayesBinned:
    """
    Composition-based wrapper around multiple mvBayes models fit within bins of
    predictor space.

    The user specifies internal bin breaks for one or more predictor dimensions
    via binBreaks, and this class fits one mvBayes model per bin in the full
    Cartesian product of bins across the selected predictor dimensions.

    This version raises an error if any bin is empty.

    Notes
    -----
    - This is a sibling class to mvBayes, not a subclass.
    - Internal component models are stored in self.modelDict, keyed by bin tuples.
    - Bin tuples contain integer bin indices, one per binned predictor dimension.
    - Since bins may have different basis decompositions, returnPostCoefs is not
      supported globally.
    - binBreaks contains internal break points only. If a dimension has m breaks,
      then it defines m+1 bins, with outer bins extending to -inf and +inf.
    - Extra keyword arguments passed to mvBayesBinned are forwarded to every
      internal mvBayes fit unless overridden via binOverrides.
    """

    def __init__(
        self,
        bayesModel,
        X,
        Y,
        binBreaks,
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
        binLabels=None,
        minBinSize=2,
        right=False,
        binOverrides=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        bayesModel : callable
            Same as in mvBayes.
        X : np.ndarray of shape (n, p)
            Predictor matrix.
        Y : np.ndarray of shape (n, nMV)
            Multivariate response matrix.
        binBreaks : dict
            Dictionary mapping predictor column indices to 1D arrays of internal
            bin break points.

            Example:
                {0: np.array([1.0, 2.0]),
                 2: np.array([-1.0, 0.0, 1.0])}

            Here X[:, 0] is split into 3 bins and X[:, 2] into 4 bins, with
            outer bins extending to -inf and +inf.
        basisType, customBasis, nBasis, propVarExplained, nCores, center, scale,
        samplesExtract, residSDExtract, idxSamplesArg, thresh :
            Same meaning as in mvBayes.
        binLabels : dict or None
            Optional dictionary mapping bin tuples to display labels.
            If None, default labels are generated.
        minBinSize : int
            Minimum number of observations required in each bin.
        right : bool
            Passed to np.digitize. If False, bins are left-closed/right-open in
            the interior; if True, right-closed/left-open.
        binOverrides : dict or None
            Optional dictionary mapping bin tuples to dictionaries of keyword
            overrides for the corresponding mvBayes fit.
        **kwargs :
            Additional keyword arguments passed to every internal mvBayes fit,
            unless overridden for a specific bin via binOverrides.
        """
        self.bayesModel = bayesModel
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        self.nMV = self.Y.shape[1]

        self.binBreaks = self._validateBinBreaks(binBreaks)
        self.binDims = list(self.binBreaks.keys())
        self.binBreakList = [np.asarray(self.binBreaks[d]) for d in self.binDims]
        self.nBinsPerDim = [len(breakPoints) + 1 for breakPoints in self.binBreakList]

        self.basisType = basisType
        self.customBasis = customBasis
        self.nBasis = nBasis
        self.propVarExplained = propVarExplained
        self.center = center
        self.scale = scale
        self.samplesExtract = samplesExtract
        self.residSDExtract = residSDExtract
        self.idxSamplesArg = idxSamplesArg
        self.thresh = thresh

        self.minBinSize = minBinSize
        self.right = right

        self.fitKwargs = dict(kwargs)

        self.binTuples = self._assignBins(self.X)
        self.binKeys = self._getAllBinKeys()
        self.nBins = len(self.binKeys)

        self.binCountDict = {
            binKey: int(np.sum(self._getBinMask(self.binTuples, binKey))) for binKey in self.binKeys
        }
        self.binMaskDict = {
            binKey: self._getBinMask(self.binTuples, binKey) for binKey in self.binKeys
        }

        self._validateBins()

        if binLabels is None:
            self.binLabelDict = {
                binKey: f"bin_{k}" for k, binKey in enumerate(self.binKeys)
            }
        else:
            self._validateBinLabelDict(binLabels)
            self.binLabelDict = dict(binLabels)

        self.binOverrides = {} if binOverrides is None else dict(binOverrides)
        self._validateBinOverrides()

        self.modelDict = {}

        self.fit(nCores=nCores)

    def _validateBinBreaks(self, binBreaks):
        """
        Validate the bin break specification.

        binBreaks maps predictor column indices to arrays of internal break points.
        If a dimension has m break points, then it defines m+1 bins, with outer
        bins extending to -inf and +inf.
        """
        if not isinstance(binBreaks, dict) or len(binBreaks) == 0:
            raise ValueError("binBreaks must be a non-empty dict mapping column indices to break points.")

        validatedBinBreaks = {}
        for key, val in binBreaks.items():
            if not isinstance(key, (int, np.integer)):
                raise ValueError("All keys in binBreaks must be integer column indices.")

            breakPoints = np.asarray(val)
            if breakPoints.ndim != 1:
                raise ValueError(f"Break points for column {key} must be a 1D array.")
            if len(breakPoints) == 0:
                raise ValueError(f"Break points for column {key} must contain at least one value.")
            if not np.all(np.diff(breakPoints) > 0):
                raise ValueError(f"Break points for column {key} must be strictly increasing.")

            validatedBinBreaks[int(key)] = breakPoints

        return dict(sorted(validatedBinBreaks.items(), key=lambda x: x[0]))

    def _validateBinLabelDict(self, binLabels):
        if not isinstance(binLabels, dict):
            raise ValueError("binLabels must be a dict mapping bin tuples to labels.")
        missingKeys = [binKey for binKey in self.binKeys if binKey not in binLabels]
        extraKeys = [key for key in binLabels if key not in self.binKeys]
        if missingKeys:
            raise ValueError(f"binLabels is missing bin keys: {missingKeys}")
        if extraKeys:
            raise ValueError(f"binLabels has keys not present among bins: {extraKeys}")

    def _validateBinOverrides(self):
        if not isinstance(self.binOverrides, dict):
            raise ValueError("binOverrides must be a dict mapping bin tuples to dicts of overrides.")
        extraKeys = [key for key in self.binOverrides if key not in self.binKeys]
        if extraKeys:
            raise ValueError(
                f"binOverrides contains keys that are not valid bins: {extraKeys}"
            )
        badVals = [key for key, val in self.binOverrides.items() if not isinstance(val, dict)]
        if badVals:
            raise ValueError(
                f"Each value in binOverrides must be a dict of keyword overrides. "
                f"Invalid keys: {badVals}"
            )

    def _digitizeOneDim(self, xCol, breakPoints):
        """
        Convert one predictor column into integer bin indices in {0, ..., nBreaks}.
        """
        return np.digitize(xCol, breakPoints, right=self.right)

    def _getBinMask(self, binTuples, binKey):
        """
        Return Boolean mask indicating which entries of binTuples equal binKey.
        """
        return np.array([bt == binKey for bt in binTuples], dtype=bool)
    
    def _assignBins(self, X):
        """
        Assign each row of X to a multidimensional bin tuple.
        """
        X = np.asarray(X)
        n = X.shape[0]
    
        oneDimBinIndexList = []
        for dim, breakPoints in zip(self.binDims, self.binBreakList):
            if dim < 0 or dim >= X.shape[1]:
                raise ValueError(f"binBreaks references invalid column index {dim}.")
            oneDimBinIndexList.append(self._digitizeOneDim(X[:, dim], breakPoints))
    
        if len(oneDimBinIndexList) == 1:
            binTuples = [(int(idx),) for idx in oneDimBinIndexList[0]]
        else:
            binTuples = [tuple(int(arr[i]) for arr in oneDimBinIndexList) for i in range(n)]
    
        binTupleArray = np.empty(n, dtype=object)
        binTupleArray[:] = binTuples
        return binTupleArray
    
    def _getAllBinKeys(self):
        """
        Get all possible bin tuples from the Cartesian product of per-dimension bins.
        """
        return list(itertools.product(*[range(nBins) for nBins in self.nBinsPerDim]))

    def _validateBins(self):
        """
        Validate minimum bin size.
        """
        smallBins = [
            binKey for binKey, count in self.binCountDict.items()
            if count < self.minBinSize
        ]
        if smallBins:
            smallCounts = {binKey: self.binCountDict[binKey] for binKey in smallBins}
            raise ValueError(
                f"Some bins have fewer than minBinSize={self.minBinSize}: {smallCounts}"
            )

    def _getBinFitArgs(self, binKey):
        """
        Construct mvBayes arguments for one bin, including overrides.
        """
        binMask = self.binMaskDict[binKey]
        Xbin = self.X[binMask]
        Ybin = self.Y[binMask]

        fitArgs = dict(
            bayesModel=self.bayesModel,
            X=Xbin,
            Y=Ybin,
            basisType=self.basisType,
            customBasis=self.customBasis,
            nBasis=self.nBasis,
            propVarExplained=self.propVarExplained,
            nCores=1,
            center=self.center,
            scale=self.scale,
            samplesExtract=self.samplesExtract,
            residSDExtract=self.residSDExtract,
            idxSamplesArg=self.idxSamplesArg,
            thresh=self.thresh,
        )
        fitArgs.update(self.fitKwargs)
        if binKey in self.binOverrides:
            fitArgs.update(self.binOverrides[binKey])

        return fitArgs

    def nCoresAdjust(self, nCores):
        """
        Adjust requested cores based on available bins and hardware.
        """
        nCores = min(nCores, self.nBins)
        nCoresAvailable = os.cpu_count()

        if nCores > 1 and not joblibAvailable:
            print("Parallel processing module 'joblib' not available. Setting nCores=1.")
            nCores = 1
        elif nCoresAvailable is not None and nCores > nCoresAvailable:
            print(f"Only {nCoresAvailable} cores are available. Using all available cores.")
            nCores = nCoresAvailable

        return nCores

    def fit(self, nCores=1):
        """
        Fit one internal mvBayes model per bin.
        """
        nCores = self.nCoresAdjust(nCores)

        def fitOneBin(binKey):
            try:
                fitArgs = self._getBinFitArgs(binKey)
                model = mvBayes(**fitArgs)
                return binKey, model
            except Exception as e:
                print(
                    f"Error fitting bin {self.binLabelDict[binKey]}, "
                    f"key={binKey}: {e}"
                )
                return binKey, None

        print(f"\rStarting mvBayesBinned with {self.nBins} bins, using {nCores} cores.")

        if nCores == 1:
            results = [fitOneBin(binKey) for binKey in self.binKeys]
        else:
            results = Parallel(n_jobs=nCores)(
                delayed(fitOneBin)(binKey) for binKey in self.binKeys
            )

        self.modelDict = {binKey: model for binKey, model in results}

        failedBins = [binKey for binKey, model in self.modelDict.items() if model is None]
        if failedBins:
            raise RuntimeError(f"Fits failed for bins: {failedBins}")

        nSamplesList = [getattr(model, "nSamples", None) for model in self.modelDict.values()]
        nSamplesList = [s for s in nSamplesList if s is not None]
        self.nSamples = min(nSamplesList) if nSamplesList else None

        return

    def predictBin(self, Xtest):
        """
        Return the bin tuple for each row of Xtest.
        """
        Xtest = np.asarray(Xtest)
        return self._assignBins(Xtest)

    def getComponent(self, binKey):
        """
        Return one internal mvBayes component.

        Parameters
        ----------
        binKey : tuple or str
            If tuple, matched against bin keys in self.modelDict.
            If str, matched against labels in self.binLabelDict.
        """
        if isinstance(binKey, str):
            matchingKeys = [key for key, label in self.binLabelDict.items() if label == binKey]
            if len(matchingKeys) == 0:
                raise ValueError(f"Unknown bin label: {binKey}")
            if len(matchingKeys) > 1:
                raise ValueError(f"Bin label is not unique: {binKey}")
            binKey = matchingKeys[0]

        if binKey not in self.modelDict:
            raise ValueError(f"Unknown bin key: {binKey}")

        return self.modelDict[binKey]

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
        returnBinTuple=False,
        **kwargs,
    ):
        """
        Predict the multivariate response at new inputs by routing each row of Xtest
        to the appropriate bin-specific mvBayes model.
    
        Parameters
        ----------
        Xtest : np.ndarray of shape (nTest, p)
            Predictor matrix.
        idxSamples, addResidError, addTruncError, returnMeanOnly, nCores,
        idxSamplesArg, **kwargs :
            Passed through to component mvBayes models.
        returnPostCoefs : bool
            Not supported because different bins may have different basis decompositions.
        returnBinTuple : bool
            Whether to also return bin tuples for Xtest rows.
    
        Returns
        -------
        Ypost : np.ndarray
            If returnMeanOnly=False: shape (nPost, nTest, nMV)
            If returnMeanOnly=True: shape (nTest, nMV)
        """
        if returnPostCoefs:
            raise NotImplementedError(
                "returnPostCoefs is not supported in mvBayesBinned because "
                "different bin models may have different basis decompositions."
            )
    
        Xtest = np.asarray(Xtest)
        nTest = Xtest.shape[0]
        binTuples = self._assignBins(Xtest)
        usedBins = list(dict.fromkeys(binTuples))
    
        if returnMeanOnly:
            Ypost = np.zeros((nTest, self.nMV))
        else:
            if self.nSamples is None:
                raise ValueError(
                    "self.nSamples is None, so the number of posterior draws cannot be inferred."
                )
    
            if isinstance(idxSamples, str):
                if idxSamples == "default":
                    nPost = self.nSamples
                elif idxSamples == "final":
                    nPost = 1
                else:
                    raise ValueError(
                        "idxSamples must be 'default', 'final', int, list, tuple, np.ndarray, "
                        "or coercible to np.ndarray."
                    )
            elif isinstance(idxSamples, int):
                nPost = 1
            elif isinstance(idxSamples, (list, tuple, np.ndarray)):
                nPost = len(np.asarray(idxSamples))
            else:
                try:
                    nPost = len(np.asarray(idxSamples))
                except Exception:
                    raise ValueError(
                        "idxSamples must be 'default', 'final', int, list, tuple, np.ndarray, "
                        "or coercible to np.ndarray."
                    )
    
            Ypost = np.zeros((nPost, nTest, self.nMV))
    
        def predictOneBin(binKey):
            rows = np.where(self._getBinMask(binTuples, binKey))[0]
            pred = self.modelDict[binKey].predict(
                Xtest[rows],
                idxSamples=idxSamples,
                addResidError=addResidError,
                addTruncError=addTruncError,
                returnPostCoefs=False,
                returnMeanOnly=returnMeanOnly,
                nCores=1,
                idxSamplesArg=idxSamplesArg,
                **kwargs,
            )
            return binKey, rows, pred
    
        nCores = self.nCoresAdjust(nCores)
    
        if nCores == 1:
            results = [predictOneBin(binKey) for binKey in usedBins]
        else:
            results = Parallel(n_jobs=nCores)(
                delayed(predictOneBin)(binKey) for binKey in usedBins
            )
    
        for binKey, rows, pred in results:
            if returnMeanOnly:
                Ypost[rows, :] = pred
            else:
                if pred.shape[0] != Ypost.shape[0]:
                    raise ValueError(
                        "Different bin models returned different numbers of posterior "
                        "samples than expected from idxSamples/self.nSamples."
                    )
                Ypost[:, rows, :] = pred
    
        if returnBinTuple:
            return Ypost, binTuples
        return Ypost

    def getMSE(self, resid=None, Xtest=None, Ytest=None, scale=False):
        """
        Compute mean squared error for stitched binned predictions.
        """
        if resid is None:
            if Xtest is None:
                Xtest = self.X
            if Ytest is None:
                Ytest = self.Y

            Ypred = self.predict(Xtest, returnMeanOnly=True)
            resid = Ytest - Ypred

        if scale:
            print(
                "Warning: scale=True is ignored in mvBayesBinned because there is no "
                "single global response scaling across bins."
            )

        return np.mean(resid ** 2)

    def traceplot(self, binKey=None, *args, **kwargs):
        """
        Traceplot for one or more bin-specific mvBayes models.
    
        Parameters
        ----------
        binKey : None, tuple, str, or iterable of tuple/str
            - If None, plot traceplots for all bins.
            - If tuple, use that bin key.
            - If str, interpret as a bin label.
            - If iterable, plot for each specified bin.
        """
        if binKey is None:
            binKeyList = self.binKeys
        elif isinstance(binKey, (tuple, str)):
            binKeyList = [binKey]
        else:
            binKeyList = list(binKey)
    
        for oneBinKey in binKeyList:
            component = self.getComponent(oneBinKey)
            component.traceplot(*args, **kwargs)
    
        return

    def describeBins(self):
        """
        Print the interval definition for each bin.
        """
        print("Bins:")
        for binKey in self.binKeys:
            intervalDescList = []
            for j, dim in enumerate(self.binDims):
                breakPoints = self.binBreaks[dim]
                idx = binKey[j]

                if idx == 0:
                    intervalStr = (
                        f"X[:, {dim}] in (-inf, {breakPoints[0]}]"
                        if self.right else
                        f"X[:, {dim}] in (-inf, {breakPoints[0]})"
                    )
                elif idx == len(breakPoints):
                    intervalStr = (
                        f"X[:, {dim}] in ({breakPoints[-1]}, inf)"
                        if self.right else
                        f"X[:, {dim}] in [{breakPoints[-1]}, inf)"
                    )
                else:
                    intervalStr = (
                        f"X[:, {dim}] in ({breakPoints[idx - 1]}, {breakPoints[idx]}]"
                        if self.right else
                        f"X[:, {dim}] in [{breakPoints[idx - 1]}, {breakPoints[idx]})"
                    )

                intervalDescList.append(intervalStr)

            print(
                f"  key={binKey}, label={self.binLabelDict[binKey]}, "
                f"nTrain={self.binCountDict[binKey]} :: " + "; ".join(intervalDescList)
            )

    def plot(
        self,
        Xtest=None,
        Ytest=None,
        nPlot=None,
        idxMV=None,
        xscale="linear",
        xlabel="Multivariate Index",
        title=None,
        file=None,
        **kwargs,
    ):
        """
        Simple plot for binned model: observed vs predicted and residuals by bin.
        """
        import matplotlib.pyplot as plt

        if Xtest is None:
            Xtest = self.X
        if Ytest is None:
            Ytest = self.Y

        Xtest = np.asarray(Xtest)
        Ytest = np.asarray(Ytest)

        Ypred, binTuples = self.predict(
            Xtest,
            returnMeanOnly=True,
            returnBinTuple=True
        )
        resid = Ytest - Ypred

        if nPlot is None:
            nPlot = min(Xtest.shape[0], 1000)
        else:
            nPlot = min(Xtest.shape[0], nPlot)

        idxPlot = np.random.choice(Xtest.shape[0], nPlot, replace=False)

        if idxMV is None:
            idxMV = np.arange(Ytest.shape[1])

        fig = plt.figure(figsize=(8, 6))
        cmap = plt.get_cmap("tab20")

        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(idxMV, Ytest[idxPlot].T, color="slateblue", alpha=0.25)
        ax1.plot(idxMV, Ypred[idxPlot].T, color="black", alpha=0.25)
        ax1.set_xscale(xscale)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel("Response")
        ax1.set_title("Observed (blue) vs Predicted (black)")

        ax2 = fig.add_subplot(2, 1, 2)
        uniquePlotBins = list(dict.fromkeys(binTuples[idxPlot]))
        for k, binKey in enumerate(uniquePlotBins):
            plotMask = self._getBinMask(binTuples[idxPlot], binKey)
            rows = idxPlot[plotMask]
            ax2.plot(idxMV, resid[rows].T, color=cmap(k % 20), alpha=0.25)

        ax2.set_xscale(xscale)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("Residual")
        ax2.set_title(f"Residuals by bin; overall MSE = {np.mean(resid**2):.4g}")

        if title is not None:
            fig.suptitle(title)
        fig.tight_layout()

        if file is None:
            plt.show()
        else:
            plt.savefig(file, **kwargs)

        plt.close(fig)

    def summary(self):
        """
        Print summary of the binned fit.
        """
        print("mvBayesBinned summary")
        print(f"  nBins: {self.nBins}")
        print(f"  nMV: {self.nMV}")
        print(f"  binDims: {self.binDims}")
        for binKey in self.binKeys:
            model = self.modelDict.get(binKey, None)
            nBasis = model.basisInfo.nBasis if model is not None else None
            print(
                f"  Bin key={binKey}: label={self.binLabelDict[binKey]}, "
                f"nTrain={self.binCountDict[binKey]}, nBasis={nBasis}"
            )
            