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


# --- bspline and custom bases -------------------------------------------------


def test_bspline_basisSetup():
    Y = np.random.rand(100, 12)
    bs = basisSetup(Y, basisType="bspline", nBasis=5, center=True, scale=True)

    assert bs.basisType == "bspline"
    assert bs.nBasis == 5
    assert bs.basis.shape == (5, 12)
    assert bs.coefs.shape == (100, 5)
    assert np.allclose(Y - bs.getYtrunc(), bs.truncError)


def test_custom_basisSetup():
    from mvBayes.mvBayes import isOrthogonal

    Y = np.random.rand(100, 10)
    customBasis = np.random.rand(4, 10)
    bs = basisSetup(Y, basisType="custom", customBasis=customBasis, nBasis=4)

    assert bs.basisType == "custom"
    assert bs.nBasis == 4
    assert bs.basis.shape == (4, 10)
    assert bs.coefs.shape == (100, 4)
    # The stored basis is orthogonalized
    assert isOrthogonal(bs.basis)
    assert np.allclose(Y - bs.getYtrunc(), bs.truncError)


def test_getCoefs_and_getYtrunc_with_Ytest():
    Y = np.random.rand(100, 10)
    bs = basisSetup(Y, basisType="pca", nBasis=5, center=True, scale=True)

    Ytest = np.random.rand(20, 10)
    coefs = bs.getCoefs(Ytest)
    assert coefs.shape == (20, 5)

    Ytrunc = bs.getYtrunc(Ytest)
    assert Ytrunc.shape == (20, 10)
    # getYtrunc(Ytest) should equal getYtrunc(coefs=getCoefs(Ytest))
    assert np.allclose(Ytrunc, bs.getYtrunc(coefs=coefs))


def test_custom_basisTransform():
    from mvBayes.mvBayes import customBasisConstruct

    np.random.seed(0)
    Ystandard = np.random.rand(50, 6)
    customBasis = np.random.rand(3, 6)
    # A valid SPD transform
    A = np.random.rand(6, 6)
    basisTransform = A @ A.T + np.eye(6)

    cbc = customBasisConstruct(customBasis, Ystandard, basisTransform=basisTransform)
    assert cbc.basis.shape == (3, 6)
    coefs = cbc.transform(Ystandard)
    assert coefs.shape == (50, 3)


# --- orthogonality helpers ----------------------------------------------------


def test_isOrthogonal_and_orthogonalize():
    from mvBayes.mvBayes import isOrthogonal, orthogonalize

    # An identity-derived orthonormal matrix
    Q = np.linalg.qr(np.random.rand(5, 5))[0]
    assert isOrthogonal(Q[:3, :])

    # A generally non-orthogonal matrix
    A = np.random.rand(3, 5)
    assert not isOrthogonal(A)
    assert isOrthogonal(orthogonalize(A))


# --- error / edge paths -------------------------------------------------------


def test_bspline_requires_nBasis():
    Y = np.random.rand(50, 10)
    with pytest.raises(ValueError):
        basisSetup(Y, basisType="bspline")


def test_bspline_min_nBasis():
    Y = np.random.rand(50, 10)
    with pytest.raises(ValueError):
        basisSetup(Y, basisType="bspline", nBasis=2)


def test_legendre_requires_nBasis():
    Y = np.random.rand(50, 10)
    with pytest.raises(ValueError):
        basisSetup(Y, basisType="legendre")


def test_custom_requires_customBasis():
    Y = np.random.rand(50, 10)
    with pytest.raises(ValueError):
        basisSetup(Y, basisType="custom")


def test_custom_shape_mismatch():
    Y = np.random.rand(50, 10)
    badBasis = np.random.rand(3, 7)  # wrong number of columns
    with pytest.raises(ValueError):
        basisSetup(Y, basisType="custom", customBasis=badBasis, nBasis=3)


def test_unsupported_basisType():
    Y = np.random.rand(50, 10)
    with pytest.raises(Exception):
        basisSetup(Y, basisType="bogus")


def test_legendre_odd_nBasis_increment(capsys):
    Y = np.random.rand(50, 10)
    bs = basisSetup(Y, basisType="legendre", nBasis=3)
    captured = capsys.readouterr()

    assert "nBasis must be even" in captured.out
    assert bs.nBasis == 4


if __name__ == "__main__":
    pytest.main()
