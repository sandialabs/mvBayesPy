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


def test_legendre_values():
    from mvBayes.mvBayes import legendre, legendreP

    x = np.linspace(-1, 1, 11)

    # P_0, ..., P_3
    expected = np.vstack([
        np.ones_like(x),
        x,
        (3 * x**2 - 1) / 2,
        (5 * x**3 - 3 * x) / 2,
    ])
    assert np.allclose(legendreP(3, x), expected)

    # associated Legendre functions of degree 2, orders 0, 1, 2
    expected = np.vstack([
        (3 * x**2 - 1) / 2,
        -3 * x * np.sqrt(1 - x**2),
        3 * (1 - x**2),
    ])
    assert np.allclose(legendre(2, x), expected)


def test_legendre_basis():
    from mvBayes.mvBayes import basisLegendre

    fDomain = np.linspace(0, 1, 20)
    basis = basisLegendre(fDomain, 3, 1)

    # rows are the Legendre polynomials of degree 1, ..., 6 on [-1, 1]
    x = 2 * fDomain - 1
    assert basis.shape == (6, 20)
    assert np.allclose(basis[0, :], x)
    assert np.allclose(basis[1, :], (3 * x**2 - 1) / 2)


def test_legendre_basisSetup():
    Y = np.random.rand(100, 10)
    bs = basisSetup(Y, basisType="legendre", nBasis=4, center=True, scale=True)

    assert bs.basisType == "legendre"
    assert bs.nBasis == 4
    assert bs.basis.shape == (4, 10)
    assert bs.coefs.shape == (100, 4)
    assert np.allclose(Y - bs.getYtrunc(), bs.truncError)


def test_plot():
    Y = np.random.rand(100, 10)
    bs = basisSetup(Y, basisType="pca", nBasis=5, center=True, scale=True)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmpfile:
        bs.plot(file=tmpfile)


if __name__ == "__main__":
    pytest.main()
