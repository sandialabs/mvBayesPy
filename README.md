# mvBayes

A python implementation of the multivariate Bayesian regression (mvBayes) framework. Decomposes a multivariate/functional response using a user-specified orthogonal basis decomposition, and then models each basis component independently using an arbitrary user-specified (univariate) Bayesian regression model. Includes prediction and plotting methods.

## Installation
Use
```bash
pip install "git+https://cee-gitlab.sandia.gov/statistics/mvBayes/#egg=mvBayes&subdirectory=mvBayesPy"
```

## Examples
* [Friedman Example](examples/friedman_example.py) - An extension of the "Friedman function" to functional response. The Bayesian regression model here is BASS (Bayesian Adaptive Smoothing Splines, see https://github.com/lanl/pyBASS)


## References


************

Author: Gavin Q. Collins

