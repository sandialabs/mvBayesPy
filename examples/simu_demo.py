#%%
import numpy as np
import pyBASS as pb
import mvBayes as mb
import fdasrsf as fs
from scipy.stats import norm

np.random.seed(251014)

def f(x):
    out = (
        norm.pdf(tt, np.sin(2 * np.pi * x[0] ** 2) / 4 - x[0] / 10 + 0.55, 0.05) * x[1]
    )
    return out


M = 99
tt = np.linspace(0, 1, M)
n = 100
p = 3
x_train = np.random.rand(n, p)
x_test = np.random.rand(1000, p)
e = np.random.normal(size=n * M)
y_train = (
    np.apply_along_axis(f, 1, x_train) + e.reshape(n, M) * 0
)  # with higher error, smooth first
y_test = np.apply_along_axis(f, 1, x_test)

x_true = np.array([0.1028, 0.5930])
# x_true = [.3,.2]
f_obs = f(x_true)
y_obs = f_obs + np.random.normal(size=M) * 0.05

tt = np.linspace(0, 1, M)
out = fs.fdawarp(y_train.T, tt)
out.multiple_align_functions(y_obs, parallel=True, lam=0.01)
gam_train = out.gam
psi_train = fs.geometry.gam_to_psi(gam_train)
ftilde_train = out.fn
qtilde_train = out.qn

gam_obs = np.linspace(0, 1, M)
psi_obs = fs.geometry.gam_to_psi(gam_obs)

# emu_ftilde = mb.mvBayes(
#     pb.bass,
#     x_train,
#     ftilde_train.T,
#     nBasis=4,
#     idxSamplesArg="mcmc_use",
#     # optionally extract posterior samples of residual standard deviation
#     residSDExtract=lambda bass_out: np.sqrt(bass_out.samples.s2),
# )

#%%
emu_vv = mb.mvBayes(
    pb.bass,
    x_train,
    psi_train.T,
    idxSamplesArg="mcmc_use",
    basisType="pns",
    # optionally extract posterior samples of residual standard deviation
    residSDExtract=lambda bass_out: np.sqrt(bass_out.samples.s2),
)

# %%
emu_vv.plot()

