import pytest
import numpy as np
from unittest.mock import patch
import mvBayes


def mock_bayesModel(X, y, **kwargs):
    class MockModel:
        def predict(self, Xtest, idxSamples=None):
            return np.random.rand(len(y))

    return MockModel()


# Test cases
def test_initialization():
    X = np.random.rand(100, 10)
    Y = np.random.rand(100, 3)
    model = mvBayes.mvBayes(mock_bayesModel, X, Y)

    assert model.X is X
    assert model.Y is Y
    assert model.nMV == 3
    assert model.basisInfo is not None

    # TODO: Add test when bayesModel already has 'samples'


def test_fit():
    X = np.random.rand(100, 10)
    Y = np.random.rand(100, 3)
    model = mvBayes.mvBayes(mock_bayesModel, X, Y)

    assert len(model.bmList) == model.basisInfo.nBasis
    assert hasattr(model.bmList[0], "samples")
    assert hasattr(model.bmList[0].samples, "residSD")


# def test_nCoresAdjust():
#     X = np.random.rand(100, 10)
#     Y = np.random.rand(100, 5)
#     model = mvBayes.mvBayes(mock_bayesModel, X, Y, nBasis=4)

#     assert model.nCoresAdjust(1) == 1

#     # Hit the CPU max
#     with patch("os.cpu_count", return_value=3):
#         with patch("mvBayes.mvBayes.PATHOS_AVAILABLE", new=True):
#             assert model.nCoresAdjust(10) == 3

#     # Hit the nBasis max
#     with patch("os.cpu_count", return_value=10):
#         with patch("mvBayes.mvBayes.PATHOS_AVAILABLE", new=True):
#             assert model.nCoresAdjust(10) == 4

#     with patch("os.cpu_count", return_value=10):
#         with patch("mvBayes.mvBayes.PATHOS_AVAILABLE", new=True):
#             assert model.nCoresAdjust(2) == 2


if __name__ == "__main__":
    pytest.main()
