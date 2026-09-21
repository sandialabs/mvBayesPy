import numpy as np
import pytest

from mvBayes.cov import covAR1, covDiag, covMA1


def _isSymmetricPSD(cov, tol=1e-8):
    if not np.allclose(cov, cov.T, atol=tol):
        return False
    eigvals = np.linalg.eigvalsh(cov)
    return np.all(eigvals > -tol)


def test_covDiag():
    np.random.seed(0)
    resid = np.random.randn(50, 6)
    cov = covDiag(resid)

    assert cov.shape == (6, 6)
    assert np.allclose(cov, np.diag(np.diag(cov)))  # diagonal
    assert np.allclose(np.diag(cov), resid.var(axis=0))


@pytest.mark.parametrize("nMV", [4, 8])
@pytest.mark.parametrize("varEqual", [True, False])
def test_covAR1_shape_and_psd(nMV, varEqual):
    np.random.seed(1)
    resid = np.random.randn(60, nMV)
    cov, coef = covAR1(resid, varEqual=varEqual)

    assert cov.shape == (nMV, nMV)
    assert _isSymmetricPSD(cov)
    assert np.isscalar(coef) or np.ndim(coef) == 0
    assert -1.0 < float(coef) < 1.0


def test_covAR1_small_nMV_branch():
    # nMV <= 3 uses the generic (non-closed-form) likelihood branch
    np.random.seed(2)
    resid = np.random.randn(40, 3)
    cov, coef = covAR1(resid)

    assert cov.shape == (3, 3)
    assert _isSymmetricPSD(cov)


def test_covAR1_single_column():
    resid = np.random.randn(10, 1)
    cov, coef = covAR1(resid)

    assert np.array_equal(cov, np.array([[1]]))
    assert coef == 0


@pytest.mark.parametrize("varEqual", [True, False])
def test_covMA1_shape_and_psd(varEqual):
    np.random.seed(3)
    resid = np.random.randn(60, 5)
    cov, coef = covMA1(resid, varEqual=varEqual)

    assert cov.shape == (5, 5)
    assert _isSymmetricPSD(cov)
    # MA(1): entries beyond the first off-diagonal are zero
    assert np.allclose(np.triu(cov, k=2), 0.0)


def test_covMA1_single_column():
    resid = np.random.randn(10, 1)
    cov, coef = covMA1(resid)

    assert np.array_equal(cov, np.array([[1]]))
    assert coef == 0


if __name__ == "__main__":
    pytest.main()
