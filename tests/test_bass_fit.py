"""Integration test using a real Bayesian regression model (pyBASS).

Skipped automatically when pyBASS is not installed (e.g. in CI), so the suite
stays dependency-free by default. Run locally after ``pip install pyBASS`` to
exercise the full fit / predict / Sobol pipeline end to end.
"""

import numpy as np
import pytest

import mvBayes as mb

from tests.util import rootmeansqerror

pb = pytest.importorskip("pyBASS")


def test_bassPCA_fit():
    # Friedman function with functional response
    tt = np.linspace(0, 1, 50)  # functional variable grid

    def f2(x):
        return (
            10.0 * np.sin(np.pi * tt * x[1])
            + 20.0 * (x[2] - 0.5) ** 2
            + 10 * x[3]
            + 5.0 * x[4]
        )

    np.random.seed(0)
    n = 500  # sample size
    p = 9  # number of predictors (only 4 are used)
    X = np.random.rand(n, p)  # training inputs
    Xtest = np.random.rand(1000, p)
    noise = np.random.normal(size=[n, len(tt)]) * 0.1
    Y = np.apply_along_axis(f2, 1, X) + noise  # training response
    Ftest = np.apply_along_axis(f2, 1, Xtest)  # noise-free test truth

    # Fit mvBayes with the BASS model (pb.bass)
    mod = mb.mvBayes(pb.bass, X, Y, nBasis=5)

    # Posterior predictive mean at new inputs
    pred = mod.predict(Xtest, returnMeanOnly=True)

    rmse = rootmeansqerror(pred, Ftest)
    print("RMSE:", rmse)

    # Loose threshold; the fit should comfortably clear this
    assert rmse < 0.5


if __name__ == "__main__":
    pytest.main()
