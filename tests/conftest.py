"""Shared fixtures and mock Bayesian models for the mvBayes test suite.

The real workflow fits an arbitrary univariate Bayesian regression model to each
basis coefficient. To keep the tests fast and dependency-free, we substitute a
lightweight mock whose ``predict`` returns random posterior draws with the shape
mvBayes expects: ``(nSamples, nTest)``. The mock optionally honors an
``idxSamples`` argument so that the ``idxSamples`` code paths in
``mvBayes.predict`` are reachable.
"""

import collections

import numpy as np
import pytest


DEFAULT_NSAMPLES = 5


def _n_from_idx(idxSamples):
    """Number of posterior samples implied by an ``idxSamples`` argument."""
    if idxSamples is None:
        return DEFAULT_NSAMPLES
    return len(np.atleast_1d(idxSamples))


def mockBayesModel(X, y, **kwargs):
    """Minimal mock bayesModel supporting ``idxSamples`` in ``predict``.

    ``predict`` returns an array of shape ``(nSamples, nTest)`` of random values,
    mimicking posterior predictive draws of a single basis coefficient.
    """

    class MockModel:
        def predict(self, Xtest, idxSamples=None):
            nSamples = _n_from_idx(idxSamples)
            return np.random.rand(nSamples, len(Xtest))

    return MockModel()


def mockBayesModelNoIdx(X, y, **kwargs):
    """Mock whose ``predict`` does NOT accept ``idxSamples``.

    Used to exercise the fallback in ``mvBayes.predict`` that detects the missing
    argument and resets ``idxSamples`` to ``"default"``.
    """

    class MockModel:
        def predict(self, Xtest):
            return np.random.rand(DEFAULT_NSAMPLES, len(Xtest))

    return MockModel()


def mockBayesModelDictSamples(X, y, **kwargs):
    """Mock exposing ``samples`` as a dict (exercises ``_getSamples`` dict path)."""

    class MockModel:
        def __init__(self):
            self.samples = {
                "residSD": np.abs(np.random.rand(DEFAULT_NSAMPLES)) + 0.1,
                "someParam": np.random.rand(DEFAULT_NSAMPLES),
            }

        def predict(self, Xtest, idxSamples=None):
            nSamples = _n_from_idx(idxSamples)
            return np.random.rand(nSamples, len(Xtest))

    return MockModel()


_SampleTuple = collections.namedtuple("_SampleTuple", ["residSD", "someParam"])


def mockBayesModelNamedTupleSamples(X, y, **kwargs):
    """Mock exposing ``samples`` as a namedtuple (``_getSamples`` namedtuple path)."""

    class MockModel:
        def __init__(self):
            self.samples = _SampleTuple(
                residSD=np.abs(np.random.rand(DEFAULT_NSAMPLES)) + 0.1,
                someParam=np.random.rand(DEFAULT_NSAMPLES),
            )

        def predict(self, Xtest, idxSamples=None):
            nSamples = _n_from_idx(idxSamples)
            return np.random.rand(nSamples, len(Xtest))

    return MockModel()


def mockBayesModelObjectSamples(X, y, **kwargs):
    """Mock exposing ``samples`` as an object that already has ``residSD``."""

    class Samples:
        def __init__(self):
            self.residSD = np.abs(np.random.rand(DEFAULT_NSAMPLES)) + 0.1
            self.someParam = np.random.rand(DEFAULT_NSAMPLES)

    class MockModel:
        def __init__(self):
            self.samples = Samples()

        def predict(self, Xtest, idxSamples=None):
            nSamples = _n_from_idx(idxSamples)
            return np.random.rand(nSamples, len(Xtest))

    return MockModel()


@pytest.fixture
def rng_seed():
    """Seed numpy's global RNG for deterministic tests, then reset."""
    np.random.seed(0)
    yield
    np.random.seed(None)


@pytest.fixture
def smallXY(rng_seed):
    """Small (X, Y) training data: X is (40, 3), Y is (40, 8)."""
    X = np.random.rand(40, 3)
    Y = np.random.rand(40, 8)
    return X, Y
