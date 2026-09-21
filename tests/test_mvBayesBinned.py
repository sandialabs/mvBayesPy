import tempfile

import numpy as np
import pytest

from mvBayes import mvBayesBinned

from tests.conftest import DEFAULT_NSAMPLES, mockBayesModel


@pytest.fixture
def binnedModel():
    np.random.seed(0)
    n = 80
    X = np.random.rand(n, 3)
    Y = np.random.rand(n, 8)
    binBreaks = {0: np.array([0.5])}  # two bins on predictor 0
    model = mvBayesBinned(
        mockBayesModel, X, Y, binBreaks=binBreaks, nBasis=3, minBinSize=2
    )
    return model, X, Y


def test_binned_fit(binnedModel):
    model, X, Y = binnedModel

    assert set(model.modelDict.keys()) == {(0,), (1,)}
    assert model.nMV == Y.shape[1]


def test_binned_predict(binnedModel):
    model, X, Y = binnedModel
    pred = model.predict(X)

    assert pred.shape == (DEFAULT_NSAMPLES, X.shape[0], Y.shape[1])


def test_binned_predictBin(binnedModel):
    model, X, Y = binnedModel
    binTuples = model.predictBin(X)

    assert len(binTuples) == X.shape[0]
    # rows with X[:,0] < 0.5 go to bin 0, else bin 1
    assigned = np.array([bt[0] for bt in binTuples])
    expected = (X[:, 0] >= 0.5).astype(int)
    assert np.array_equal(assigned, expected)


def test_binned_getComponent(binnedModel):
    model, X, Y = binnedModel

    comp = model.getComponent((0,))
    assert hasattr(comp, "predict")

    with pytest.raises(ValueError):
        model.getComponent((99,))


def test_binned_getMSE(binnedModel):
    model, X, Y = binnedModel
    mse = model.getMSE()

    assert np.isscalar(mse) or np.ndim(mse) == 0
    assert mse >= 0


def test_binned_summary_and_describe(binnedModel, capsys):
    model, X, Y = binnedModel

    model.summary()
    model.describeBins()
    captured = capsys.readouterr()
    assert "nBins" in captured.out


if __name__ == "__main__":
    pytest.main()
