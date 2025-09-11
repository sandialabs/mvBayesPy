import pytest
import numpy as np
from mvBayes import basisSetup
import tempfile


def test_initialization():
    Y = np.random.rand(100, 10)
    bs = basisSetup(Y, basisType="pca", nBasis=5, center=True, scale=True)

    assert bs.Y.shape == (100, 10)
    assert bs.nMV == 10
    assert bs.basisType == "pca"
    assert bs.nBasis == 5
    assert bs.propVarExplained <= 1.0
    assert bs.basis.shape == (5, 10)
    assert bs.coefs.shape == (100, 5)
    assert bs.truncError.shape == (100, 10)


def test_getYtrunc():
    Y = np.random.rand(100, 10)
    bs = basisSetup(Y, basisType="pca", nBasis=5, center=True, scale=True)
    Ytrunc = bs.getYtrunc()

    assert Ytrunc.shape == (100, 10)
    assert np.allclose(Y - Ytrunc, bs.truncError)
    assert np.allclose(Y - bs.getYtrunc(Y), bs.truncError)


def test_getCoefs():
    Y = np.random.rand(100, 10)
    bs = basisSetup(Y, basisType="pca", nBasis=5, center=True, scale=True)
    coefs = bs.getCoefs()

    assert coefs.shape == (100, 5)
    assert np.allclose(coefs, bs.coefs)
    assert np.allclose(coefs, bs.getCoefs(Y))


def test_preprocessY():
    Y = np.random.rand(100, 10)
    bs = basisSetup(Y, basisType="pca", nBasis=5, center=True, scale=True)
    Y_preprocessed = bs.preprocessY()

    assert np.allclose(Y, Y_preprocessed)
    assert np.allclose(Y, bs.preprocessY(Y))


def test_plot():
    Y = np.random.rand(100, 10)
    bs = basisSetup(Y, basisType="pca", nBasis=5, center=True, scale=True)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmpfile:
        bs.plot(file=tmpfile)


if __name__ == "__main__":
    pytest.main()
