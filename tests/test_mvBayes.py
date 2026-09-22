import importlib
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

import mvBayes

from tests.conftest import (
    DEFAULT_NSAMPLES,
    mockBayesModel,
    mockBayesModelDictSamples,
    mockBayesModelNamedTupleSamples,
    mockBayesModelNoIdx,
    mockBayesModelObjectSamples,
)

# `mvBayes/__init__.py` does `from .mvBayes import *`, so the `mvBayes.mvBayes`
# package attribute is the class, not the submodule. Resolve the submodule
# explicitly so patching module-level globals works on every Python version.
mvBayesModule = importlib.import_module("mvBayes.mvBayes")


# --- initialization and fit ---------------------------------------------------


def test_initialization():
    X = np.random.rand(100, 10)
    Y = np.random.rand(100, 3)
    model = mvBayes.mvBayes(mockBayesModel, X, Y)

    assert model.X is X
    assert model.Y is Y
    assert model.nMV == 3
    assert model.basisInfo is not None


def test_fit():
    X = np.random.rand(100, 10)
    Y = np.random.rand(100, 3)
    model = mvBayes.mvBayes(mockBayesModel, X, Y)

    assert len(model.bmList) == model.basisInfo.nBasis
    assert hasattr(model.bmList[0], "samples")
    assert hasattr(model.bmList[0].samples, "residSD")
    assert model.nSamples == DEFAULT_NSAMPLES


# --- predict ------------------------------------------------------------------


def test_predict_shape(smallXY):
    X, Y = smallXY
    model = mvBayes.mvBayes(mockBayesModel, X, Y, nBasis=3)
    pred = model.predict(X)

    assert pred.shape == (DEFAULT_NSAMPLES, X.shape[0], Y.shape[1])


def test_predict_returnMeanOnly_and_postCoefs(smallXY):
    X, Y = smallXY
    model = mvBayes.mvBayes(mockBayesModel, X, Y, nBasis=3)

    pred, postCoefs = model.predict(
        X, returnMeanOnly=True, returnPostCoefs=True
    )
    assert pred.shape == (X.shape[0], Y.shape[1])
    assert postCoefs.shape == (X.shape[0], 3)


def test_predict_addResidError_and_truncError(smallXY):
    X, Y = smallXY
    model = mvBayes.mvBayes(mockBayesModel, X, Y, nBasis=3)

    pred = model.predict(X, addResidError=True, addTruncError=True)
    assert pred.shape == (DEFAULT_NSAMPLES, X.shape[0], Y.shape[1])


@pytest.mark.parametrize(
    "idxSamples",
    ["final", 0, [0, 1], np.array([0, 2])],
)
def test_predict_idxSamples_variants(smallXY, idxSamples):
    X, Y = smallXY
    model = mvBayes.mvBayes(mockBayesModel, X, Y, nBasis=3)

    pred = model.predict(X, idxSamples=idxSamples)
    expectedN = 1 if isinstance(idxSamples, (int, str)) else len(idxSamples)
    assert pred.shape == (expectedN, X.shape[0], Y.shape[1])


def test_predict_idxSamples_fallback_when_unsupported(smallXY, capsys):
    X, Y = smallXY
    # Model whose predict has no idxSamples arg -> resets to "default"
    model = mvBayes.mvBayes(mockBayesModelNoIdx, X, Y, nBasis=3)

    pred = model.predict(X, idxSamples=[0, 1])
    captured = capsys.readouterr()
    assert "not an argument" in captured.out
    assert pred.shape == (DEFAULT_NSAMPLES, X.shape[0], Y.shape[1])


# --- getMSE / getNegLogLik ----------------------------------------------------


def test_getMSE(smallXY):
    X, Y = smallXY
    model = mvBayes.mvBayes(mockBayesModel, X, Y, nBasis=3)

    mse = model.getMSE()
    assert np.isscalar(mse) or np.ndim(mse) == 0
    assert mse >= 0

    resid = np.random.randn(*Y.shape)
    mseResid = model.getMSE(resid=resid, scale=False)
    assert mseResid == pytest.approx(np.mean(resid**2))


def test_getNegLogLik(smallXY):
    X, Y = smallXY
    model = mvBayes.mvBayes(mockBayesModel, X, Y, nBasis=3)

    resid = np.random.randn(*Y.shape)
    nll = model.getNegLogLik(resid)
    assert np.isfinite(nll)


# --- updateBasis / superDR ----------------------------------------------------


def test_updateBasis_no_cov(smallXY):
    X, Y = smallXY
    model = mvBayes.mvBayes(mockBayesModel, X, Y, nBasis=3)

    Ystandard = (model.basisInfo._Y - model.basisInfo.Ycenter) / model.basisInfo.Yscale
    _, coefsPred = model.predict(X, returnPostCoefs=True, returnMeanOnly=True)

    model.updateBasis(coefsPred, Ystandard, cov=None)
    assert model.basisInfo.basis.shape == (3, Y.shape[1])
    assert model.basisInfo.basisConstruct.logDet is None


@pytest.mark.parametrize(
    "covStructure,varEqual",
    [
        ("independent", True),
        ("independent", False),
        ("AR1", True),
        ("MA1", True),
    ],
)
def test_superDR(smallXY, covStructure, varEqual):
    X, Y = smallXY
    model = mvBayes.mvBayes(mockBayesModel, X, Y, nBasis=3)

    model.superDR(nIterations=1, covStructure=covStructure, varEqual=varEqual)

    # One initial value plus one per iteration
    assert len(model.MSE) == 2
    assert len(model.negLogLik) == 2
    assert np.all(np.isfinite(model.MSE))


# --- samples extraction / hooks -----------------------------------------------


@pytest.mark.parametrize(
    "modelFn",
    [
        mockBayesModelDictSamples,
        mockBayesModelNamedTupleSamples,
        mockBayesModelObjectSamples,
    ],
)
def test_getSamples_branches(smallXY, modelFn):
    X, Y = smallXY
    model = mvBayes.mvBayes(modelFn, X, Y, nBasis=3)

    assert hasattr(model.bmList[0].samples, "residSD")
    assert hasattr(model.bmList[0].samples, "someParam")


def test_samplesExtract_and_residSDExtract_hooks(smallXY):
    X, Y = smallXY

    class RawSamples:
        def __init__(self):
            self.residSD = np.abs(np.random.rand(DEFAULT_NSAMPLES)) + 0.1

    def rawModel(X, y, **kwargs):
        class MockModel:
            def __init__(self):
                self._raw = RawSamples()

            def predict(self, Xtest, idxSamples=None):
                nSamples = (
                    DEFAULT_NSAMPLES if idxSamples is None else len(np.atleast_1d(idxSamples))
                )
                return np.random.rand(nSamples, len(Xtest))

        return MockModel()

    model = mvBayes.mvBayes(
        rawModel,
        X,
        Y,
        nBasis=3,
        samplesExtract=lambda bm: bm._raw,
        residSDExtract=lambda bm: bm._raw.residSD,
    )
    assert hasattr(model.bmList[0].samples, "residSD")


# --- nCoresAdjust -------------------------------------------------------------


def test_nCoresAdjust(smallXY):
    X, Y = smallXY
    model = mvBayes.mvBayes(mockBayesModel, X, Y, nBasis=3)

    # Clamped to nBasis (3), even if more requested, with joblib available
    with patch.object(mvBayesModule, "JOBLIB_AVAILABLE", True):
        with patch("os.cpu_count", return_value=16):
            assert model.nCoresAdjust(10) == 3
            assert model.nCoresAdjust(2) == 2

        # Clamp to available CPUs
        with patch("os.cpu_count", return_value=2):
            assert model.nCoresAdjust(3) == 2

    # joblib unavailable -> forced to 1
    with patch.object(mvBayesModule, "JOBLIB_AVAILABLE", False):
        with patch("os.cpu_count", return_value=16):
            assert model.nCoresAdjust(3) == 1


# --- plotting smoke tests -----------------------------------------------------


def test_plot_smoke(smallXY):
    X, Y = smallXY
    model = mvBayes.mvBayes(mockBayesModel, X, Y, nBasis=3)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmpfile:
        model.plot(file=tmpfile)


def test_plot_with_test_data(smallXY):
    X, Y = smallXY
    model = mvBayes.mvBayes(mockBayesModel, X, Y, nBasis=3)
    Xtest = np.random.rand(15, X.shape[1])
    Ytest = np.random.rand(15, Y.shape[1])

    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmpfile:
        model.plot(Xtest=Xtest, Ytest=Ytest, file=tmpfile)


def test_traceplot_smoke(smallXY):
    X, Y = smallXY
    model = mvBayes.mvBayes(mockBayesModel, X, Y, nBasis=3)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmpfile:
        model.traceplot(file=tmpfile)


# --- Sobol' sensitivity (Monte Carlo path) ------------------------------------


def test_mvSobol_monte_carlo(smallXY):
    X, Y = smallXY
    model = mvBayes.mvBayes(mockBayesModel, X, Y, nBasis=3)

    # Mock model is not "bass", so mvSobol uses the Monte Carlo estimator
    model.mvSobol(totalSobol=True, nMC=2**6)

    p = X.shape[1]
    assert model.firstOrderSobol.shape == (1, p, Y.shape[1])
    assert model.totalOrderSobol.shape == (1, p, Y.shape[1])
    assert model.varTotal.shape == (Y.shape[1],)


def test_mvSobol_first_order_only(smallXY):
    X, Y = smallXY
    model = mvBayes.mvBayes(mockBayesModel, X, Y, nBasis=3)

    model.mvSobol(totalSobol=False, nMC=2**6)
    assert hasattr(model, "firstOrderSobol")
    assert not hasattr(model, "totalOrderSobol")


@pytest.mark.parametrize("waterfall", [False, True])
def test_plotSobol_smoke(smallXY, waterfall):
    X, Y = smallXY
    model = mvBayes.mvBayes(mockBayesModel, X, Y, nBasis=3)
    model.mvSobol(totalSobol=True, nMC=2**6)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmpfile:
        model.plotSobol(waterfall=waterfall, file=tmpfile)


def test_plotSobol_before_mvSobol_raises(smallXY):
    X, Y = smallXY
    model = mvBayes.mvBayes(mockBayesModel, X, Y, nBasis=3)

    with pytest.raises(Exception):
        model.plotSobol()


if __name__ == "__main__":
    pytest.main()
