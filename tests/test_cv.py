import numpy as np
import pytest

from mvBayes.cv import _avg_crps_sorted, _crps_sorted_1d, mvCV

from tests.conftest import mockBayesModel


def test_crps_identical_samples_is_zero():
    # If all samples equal the truth, CRPS is exactly 0
    assert _crps_sorted_1d(np.zeros(20), 0.0) == pytest.approx(0.0)


def test_crps_nonnegative_and_penalizes_bias():
    np.random.seed(0)
    samples = np.random.randn(200)
    crps_good = _crps_sorted_1d(samples, 0.0)
    crps_bad = _crps_sorted_1d(samples, 5.0)  # forecast far from truth

    assert crps_good >= 0.0
    assert crps_bad > crps_good


def test_avg_crps_averages_dimensions():
    np.random.seed(1)
    samples = np.random.rand(50, 4)
    y_true = np.random.rand(4)

    avg = _avg_crps_sorted(samples, y_true)
    manual = np.mean(
        [_crps_sorted_1d(samples[:, j], y_true[j]) for j in range(4)]
    )
    assert avg == pytest.approx(manual)


EXPECTED_KEYS = {
    "rmse",
    "rSquared",
    "crps",
    "coverageTarget",
    "coverage",
    "intervalWidth",
    "intervalScore",
    "fitTime",
    "predictTime",
    "effectiveArgs",
}


@pytest.mark.parametrize("uqTruncMethod", ["gaussian", "empirical"])
def test_mvCV_runs_and_returns_metrics(uqTruncMethod):
    np.random.seed(2)
    X = np.random.rand(40, 3)
    Y = np.random.rand(40, 8)

    out = mvCV(
        mockBayesModel,
        X,
        Y,
        nTrain=25,
        nTest=10,
        nRep=2,
        seed=1,
        uqTruncMethod=uqTruncMethod,
        nBasis=3,
    )

    assert EXPECTED_KEYS.issubset(out.keys())
    for key in ["rmse", "rSquared", "crps", "coverage", "intervalWidth", "intervalScore"]:
        assert len(out[key]) == 2
    assert np.all(out["rmse"] >= 0)
    assert np.all((out["coverage"] >= 0) & (out["coverage"] <= 1))
    assert out["effectiveArgs"]["nTrain"] == 25
    assert out["effectiveArgs"]["nTest"] == 10


def test_mvCV_default_split():
    np.random.seed(3)
    X = np.random.rand(30, 2)
    Y = np.random.rand(30, 6)

    out = mvCV(mockBayesModel, X, Y, nRep=1, seed=0, nBasis=2)
    # Default: nTest = n // 2, nTrain = n - nTest
    assert out["effectiveArgs"]["nTest"] == 15
    assert out["effectiveArgs"]["nTrain"] == 15


def test_mvCV_invalid_uqTruncMethod():
    np.random.seed(4)
    X = np.random.rand(30, 2)
    Y = np.random.rand(30, 6)

    with pytest.raises(ValueError):
        mvCV(
            mockBayesModel,
            X,
            Y,
            nTrain=20,
            nTest=8,
            uqTruncMethod="bogus",
            nBasis=2,
        )


def test_mvCV_invalid_split_raises():
    X = np.random.rand(20, 2)
    Y = np.random.rand(20, 4)

    with pytest.raises(ValueError):
        mvCV(mockBayesModel, X, Y, nTrain=25, nBasis=2)  # nTrain >= n

    with pytest.raises(ValueError):
        mvCV(mockBayesModel, X, Y, nTest=25, nBasis=2)  # nTest >= n


if __name__ == "__main__":
    pytest.main()
