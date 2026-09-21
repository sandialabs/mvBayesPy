"""Smoke tests for the elastic-FDA and PNS code paths.

These require the optional ``fdasrsf`` dependency and are skipped automatically
when it is not installed (e.g. in CI).
"""

import numpy as np
import pytest

import mvBayes

from tests.conftest import DEFAULT_NSAMPLES, mockBayesModel

pytest.importorskip("fdasrsf")


def _bumps(n, nMV, seed=0):
    """Generate smooth, strictly-positive-ish curves suitable for elastic FDA."""
    rng = np.random.default_rng(seed)
    tt = np.linspace(0, 1, nMV)
    Y = np.zeros((n, nMV))
    for i in range(n):
        center = rng.uniform(0.3, 0.7)
        width = rng.uniform(0.05, 0.15)
        amp = rng.uniform(1.0, 2.0)
        Y[i] = amp * np.exp(-((tt - center) ** 2) / (2 * width**2))
    return Y


def test_basisSetup_pns():
    from mvBayes import basisSetup

    Y = _bumps(30, 20, seed=1)
    bs = basisSetup(Y, basisType="pns", nBasis=3)

    assert bs.basisType == "pns"
    assert bs.nBasis == 3
    assert bs.coefs.shape[0] == 30


def test_mvBayesElastic_smoke():
    Y = _bumps(30, 20, seed=2)
    X = np.random.rand(30, 3)

    mod = mvBayes.mvBayesElastic(mockBayesModel, X, Y, nBasis=3)

    pred = mod.predict(X)
    assert pred.shape[0] == DEFAULT_NSAMPLES
    assert pred.shape[1] == X.shape[0]


if __name__ == "__main__":
    pytest.main()
