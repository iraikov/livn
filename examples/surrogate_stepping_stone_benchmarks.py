"""surrogate_stepping_stone_benchmarks.py
=======================================

Demonstration of progressive "stepping stone" benchmarks for evaluating
surrogate models of biophysical neural network responses.

Each benchmark adds a progressively more difficult representational demand,
starting with exact linear fits, through structured basis decompositions,
to arbitrary nonlinear and multi-function approximation.

They are "stepping stones" because each level isolates a necessary but
not sufficient condition for universal approximation: linearity, basis
decomposition, pointwise nonlinearity, multi-function
generalization. A surrogate failing at an earlier level cannot
"qualify" for the later ones, and a surrogate passing all levels has
demonstrated the core representational prerequisites without requiring
the exponentially larger and more complex stimulus libraries and
decoder complexity that a full universal approximation test would
require.

LIVN workflow context
---------------------

In the full LIVN pipeline, a surrogate model is first trained on
the biophysical model's responses to systematic stimulus patterns.
The trained, frozen surrogate is then evaluated on a suite of downstream
function-approximation tasks.  For each task a lightweight linear decoder
is fitted on top of the frozen surrogate representations and compared
against the same decoder fitted on the ground-truth biophysical responses.

The performance retention ratio (R^2_surrogate / R^2_ground_truth)
quantifies how much task-relevant information the surrogate has preserved.

Benchmark demo simplification
-----------------------------

Because no biophysical simulator is present in this demo, a small
frozen random MLP (the "biophysical proxy") stands in for the biophysical model.

The biophysical proxy maps 2-D stimulus parameters to a 16-D "neural response" through
fixed random weights and tanh activations.  This preserves the key challenge:
the surrogate must learn a compressed, approximately invertible representation
of a nonlinear high-dimensional mapping.  All downstream logic is identical to
what would be used with the real biophysical models.

Benchmark workflow
------------------

The biophysical proxy has fixed random weights and is never trained;
it is evaluated (forward pass only) to produce "ground-truth" responses.

    Stimulus x (2-D)
         |
         +---> [Biophys. proxy *]  -- (z_gt, x) train surrogate once -->  [Surrogate]
                    |                                                           |
               z_gt (16-D)                                               z_sur (16-D)
                    |                                                           |
              [Linear decoder GT] <--------- train per benchmark -------> [Linear decoder S]
                    |                                                           |
                R^2_GT (upper bound)                                       R^2_S (surrogate)

    * Biophys. proxy: frozen random MLP, weights fixed at init with Glorot rule.

Performance retention metric = R^2_sur / R^2_GT

Gradient-fitting comparison
----------------------------

Standard surrogate training minimises a loss on function values alone.
Gradient-enriched Sobolev training co-trains on analytical partial
derivatives, forcing the model to also learn the gradient field.  A separate
direct-comparison experiment (bypassing the proxy pipeline) illustrates cases
where gradient-enriched should perform better:

  - Rosenbrock function (narrow curved valley; gradient direction is critical
    for downstream optimisation)
  - Anisotropic ridge (gradient magnitude differs by ~1000x between axes;
    isotropic models systematically misattribute the steep direction)
  - Saddle-point landscape (gradient sign topology matters near the saddle)

Benchmark tasks
---------------

  LinearApproximationBenchmark     -- f(x) = Ax + b, sanity check
  BasisFunctionBenchmark           -- structured signals (Fourier / Legendre / RBF)
  NonlinearApproximationBenchmark  -- standard + adversarial nonlinear functions
  MultiFunctionSuiteBenchmark      -- single decoder across diverse function types

Usage
-----
    python surrogate_stepping_stone_benchmarks.py \\
        --benchmarks all --seed 42 --n-train 800 --output-dir figures/surrogate_benchmarks

"""

import argparse
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

logger = logging.getLogger("surrogate_stepping_stone_benchmarks")

plt.style.use("ggplot")
plt.rc("font", size=10)
plt.rc("axes", titlesize=11, labelsize=10)
plt.rc("legend", fontsize=9)

class BiophysicalModelProxy:
    """
    Frozen random MLP that stands in for the biophysical neural model.

    In the real LIVN pipeline this is replaced by a biophysical neuron network model.
    The proxy preserves the key structural property:
    it maps low-dimensional stimulus parameters to high-dimensional nonlinear
    responses through a composition of random affine transformations and tanh
    activations, creating a compression problem for any surrogate.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_sizes: Tuple[int, ...] = (64, 32),
        output_dim: int = 16,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        dims = [input_dim] + list(hidden_sizes) + [output_dim]
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])  # Glorot initialisation
            self.weights.append(rng.normal(0.0, scale, (dims[i], dims[i + 1])))
            self.biases.append(rng.normal(0.0, 0.1, (dims[i + 1],)))

    def respond(self, X: np.ndarray) -> np.ndarray:
        """Map stimulus array X (n, input_dim) to responses (n, output_dim)."""
        h: np.ndarray = X
        for idx, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ W + b
            if idx < len(self.weights) - 1:
                h = np.tanh(h)
        return h


# ---------------------------------------------------------------------------
# Surrogate Models
# ---------------------------------------------------------------------------

#   (a) ProxySurrogates: trained on (x, proxy_response) pairs; used in the
#       three-phase pipeline benchmarks.
#   (b) DirectSurrogates: trained on (x, f(x)) directly; used only in the
#       gradient-accuracy comparison for adversarial functions.


class _BaseModel:
    """Minimal interface shared by all surrogate wrappers."""

    name: str = "Base"

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# Proxy surrogates (multi-output: x -> 16-D response)


class LinearProxySurrogate(_BaseModel):
    """
    Ridge regression surrogate for the biophysical model (proxy).

    Maps stimulus x directly to neural response z via a linear model.  Included
    as the simplest possible surrogate to establish a lower performance bound.
    Because the biophysical model itself is non-linear, a linear surrogate will under-fit
    and reveal how much information is lost when the mapping is restricted to a
    single affine transformation.
    """

    name = "Linear (Ridge)"

    def __init__(self, alpha: float = 1.0) -> None:
        self._scaler = StandardScaler()
        self._model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, Z: np.ndarray) -> None:
        self._model.fit(self._scaler.fit_transform(X), Z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._scaler.transform(X))


class MLPProxySurrogate(_BaseModel):
    """
    Shallow MLP surrogate for the biophysical model proxy.

    A two-hidden-layer MLP is the simplest surrogate model suitable for non-linear biophysics:
    it has enough expressive capacity to approximate non-linear dynamics  and serves as a
    representative of the MLP-family surrogate architectures that are present in the
    surrogate literature.  Its performance on the stepping stone benchmarks quantifies
    how much task-relevant information is preserved when a shallow MLP is used
    as a biophysical model surrogate.
    """

    name = "Shallow MLP"

    def __init__(
        self,
        hidden: Tuple[int, ...] = (64, 64),
        max_iter: int = 500,
        seed: int = 0,
    ) -> None:
        self._scaler = StandardScaler()
        self._model = MLPRegressor(
            hidden_layer_sizes=hidden,
            max_iter=max_iter,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )

    def fit(self, X: np.ndarray, Z: np.ndarray) -> None:
        self._model.fit(self._scaler.fit_transform(X), Z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._scaler.transform(X))


# Direct surrogates (scalar output: x -> f(x))

class DirectLinearSurrogate(_BaseModel):
    """Ridge regression trained directly on scalar-valued benchmark functions."""

    name = "Linear (Ridge)"

    def __init__(self, alpha: float = 1.0) -> None:
        self._model = Pipeline(
            [("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


class DirectPolynomialSurrogate(_BaseModel):
    """Degree-2 polynomial Ridge regression trained directly on benchmark functions."""

    def __init__(self, degree: int = 2, alpha: float = 1.0) -> None:
        self.name = f"Polynomial deg={degree}"
        self._model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


class DirectMLPSurrogate(_BaseModel):
    """Shallow MLP trained directly on scalar-valued benchmark functions."""

    name = "Shallow MLP"

    def __init__(
        self,
        hidden: Tuple[int, ...] = (64, 64),
        max_iter: int = 1000,
        seed: int = 0,
    ) -> None:
        self._model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("mlp", MLPRegressor(
                    hidden_layer_sizes=hidden,
                    max_iter=max_iter,
                    random_state=seed,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=30,
                )),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


class GradientEnrichedSurrogate(_BaseModel):
    """
    Gradient-enriched Sobolev-trained MLP surrogate.

    Standard surrogate training minimises a loss on function values f(x) alone,
    which could produce gradient estimates that diverge from the true gradient field.
    
    Sobolev training adds analytical partial derivatives as additional
    supervised targets via the augmented target vector
    [f(x), df/dx_0, ..., df/dx_{d-1}].  This forces the model to learn not only
    the correct output values but also the correct first-order geometry of the
    function landscape.

    For functions with narrow curved valleys (Rosenbrock), strong gradient
    anisotropy (anisotropic ridge), or sign-changing gradient topology
    (saddle landscape), gradient co-training can improve gradient
    cosine similarity.

    Implementation: a MultiOutputRegressor wrapping a shared MLPRegressor base
    is trained on targets [f(x), df/dx_0, ..., df/dx_{d-1}].  At prediction time
    only the first output column is returned as the function estimate; the
    remaining columns give the gradient estimate used for the gradient accuracy
    metric.
    """

    name = "Gradient-Enriched MLP (Sobolev)"

    def __init__(
        self,
        hidden: Tuple[int, ...] = (64, 64),
        max_iter: int = 2000,
        seed: int = 0,
    ) -> None:
        base = MLPRegressor(
            hidden_layer_sizes=hidden,
            max_iter=max_iter,
            random_state=seed,
            early_stopping=False,
        )
        self._model = MultiOutputRegressor(base)
        self._scaler = StandardScaler()

    def fit(self, X: np.ndarray, y_aug: np.ndarray) -> None:
        """
        y_aug : array of shape (n, 1+d).
        Column 0 holds f(x); columns 1..d hold the partial derivatives.
        """
        self._model.fit(self._scaler.fit_transform(X), y_aug)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._scaler.transform(X))[:, 0]

    def predict_gradient(self, X: np.ndarray) -> np.ndarray:
        """Return the surrogate's estimate of grad f at each row of X."""
        return self._model.predict(self._scaler.transform(X))[:, 1:]


class DirectGPRSurrogate(_BaseModel):
    """
    Gaussian Process Regressor trained directly on scalar-valued benchmark functions.

    GPR with an RBF kernel produces globally smooth predictions with calibrated
    uncertainty and is a widely used surrogate in engineering design.  Its
    single length-scale parameter makes it well-suited to isotropic functions
    but causes it to fail on anisotropic functions: the optimised length scale
    is a compromise between directions with very different gradient magnitudes,
    leading to over-smoothing in steep directions.  Serves as a reference
    point for isotropic surrogate behavior.

    Training is restricted to at most GPR_MAX_TRAIN points due to the O(n^3)
    complexity of exact GP inference.
    """

    name = "Gaussian Process (RBF)"
    GPR_MAX_TRAIN: int = 300

    def __init__(self, seed: int = 0) -> None:
        kernel = RBF(length_scale_bounds=(1e-2, 10.0)) + WhiteKernel()
        self._model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            random_state=seed,
            normalize_y=True,
        )
        self._scaler = StandardScaler()
        self._rng = np.random.default_rng(seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) > self.GPR_MAX_TRAIN:
            idx = self._rng.choice(len(X), self.GPR_MAX_TRAIN, replace=False)
            X, y = X[idx], y[idx]
        self._model.fit(self._scaler.fit_transform(X), y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._scaler.transform(X))



def compute_analytical_gradient(
    func: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Central-difference Jacobian of func evaluated at each row of X.

    Returns an array of shape (n, d) where entry [i, j] approximates
    df/dx_j at X[i].  Not defined for discontinuous functions (step_disc) -
    results at the discontinuity boundary will be numerically large.
    """
    n, d = X.shape
    G = np.zeros((n, d), dtype=float)
    for j in range(d):
        Xp = X.copy()
        Xm = X.copy()
        Xp[:, j] += eps
        Xm[:, j] -= eps
        G[:, j] = (func(Xp) - func(Xm)) / (2.0 * eps)
    return G


def build_sobolev_targets(
    func: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Build augmented target matrix [f(x), df/dx_0, ..., df/dx_{d-1}] for
    Sobolev training.  Returns array of shape (n, 1+d).
    """
    y = func(X).reshape(-1, 1)
    G = compute_analytical_gradient(func, X, eps=eps)
    return np.concatenate([y, G], axis=1)


def gradient_cosine_similarity(
    model: _BaseModel,
    X_test: np.ndarray,
    true_grads: np.ndarray,
    eps: float = 1e-4,
) -> float:
    """
    Mean cosine similarity between the model's gradient estimate and true_grads.

    If model exposes predict_gradient(), that is used directly (GradientEnriched).
    Otherwise the gradient is estimated via forward finite differences on
    model.predict().  A value close to +1 indicates the model's gradient field
    agrees with the true gradient direction across the test distribution.
    """
    if hasattr(model, "predict_gradient"):
        est = model.predict_gradient(X_test)  # type: ignore[attr-defined]
    else:
        n, d = X_test.shape
        est = np.zeros((n, d), dtype=float)
        y0 = model.predict(X_test)
        for j in range(d):
            Xp = X_test.copy()
            Xp[:, j] += eps
            est[:, j] = (model.predict(Xp) - y0) / eps

    norms_e = np.linalg.norm(est, axis=1)
    norms_t = np.linalg.norm(true_grads, axis=1)
    dot = np.einsum("ij,ij->i", est, true_grads)
    denom = norms_e * norms_t
    cos = np.where(denom > 1e-12, dot / denom, 0.0)
    return float(np.mean(cos))


def train_linear_decoder(
    Z_tr: np.ndarray,
    y_tr: np.ndarray,
    Z_te: np.ndarray,
    y_te: np.ndarray,
) -> Tuple[float, float]:
    """
    Fit Ridge regression from representation Z to target y.
    Returns (r2_train, r2_test).  y may be 1-D or 2-D (multi-output).
    """
    dec = Ridge(alpha=1.0)
    dec.fit(Z_tr, y_tr)
    r2_tr = float(r2_score(y_tr, dec.predict(Z_tr)))
    r2_te = float(r2_score(y_te, dec.predict(Z_te)))
    return r2_tr, r2_te


# Benchmark Function Definitions


# Linear functions
def make_linear_function(
    A: np.ndarray, b: float
) -> Callable[[np.ndarray], np.ndarray]:
    """Return f(X) = X @ A + b (A is a column vector, b is scalar)."""
    def _f(X: np.ndarray) -> np.ndarray:
        return (X @ A + b).ravel()
    return _f


# Basis function signals
def fourier_signal(
    X: np.ndarray, coeffs: np.ndarray, freqs: np.ndarray
) -> np.ndarray:
    """
    Superposition of Fourier components along the first input dimension.
    f(x) = sum_k coeffs[k] * sin(freqs[k] * x_0 + pi*k/K)
    """
    out = np.zeros(len(X))
    K = len(coeffs)
    for k in range(K):
        out += coeffs[k] * np.sin(freqs[k] * X[:, 0] + np.pi * k / K)
    return out


def legendre_signal(X: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """
    Linear combination of Legendre polynomials of degree 1..K along x_0.
    Input is clipped to [-1, 1] as required by the Legendre basis.
    """
    from numpy.polynomial.legendre import legval  # type: ignore

    out = np.zeros(len(X))
    x = np.clip(X[:, 0], -1.0, 1.0)
    for k, c in enumerate(coeffs):
        deg_coeffs = np.zeros(k + 2)
        deg_coeffs[k + 1] = 1.0
        out += c * legval(x, deg_coeffs)
    return out


def rbf_signal(
    X: np.ndarray, coeffs: np.ndarray, centers: np.ndarray, width: float = 0.3
) -> np.ndarray:
    """
    Weighted sum of isotropic Gaussian RBF kernels.
    f(x) = sum_k coeffs[k] * exp(-||x - centers[k]||^2 / (2*width^2))
    """
    out = np.zeros(len(X))
    for c, mu in zip(coeffs, centers):
        out += c * np.exp(-np.sum((X - mu) ** 2, axis=1) / (2.0 * width ** 2))
    return out


# Standard nonlinear functions


def quadratic_bowl(X: np.ndarray) -> np.ndarray:
    return X[:, 0] ** 2 + X[:, 1] ** 2


def gaussian_bump(X: np.ndarray) -> np.ndarray:
    return np.exp(-X[:, 0] ** 2 - X[:, 1] ** 2)


def trig_product(X: np.ndarray) -> np.ndarray:
    return np.sin(X[:, 0]) * np.cos(X[:, 1])


def rational_func(X: np.ndarray) -> np.ndarray:
    return X[:, 0] / (1.0 + X[:, 1] ** 2)


def step_disc(X: np.ndarray) -> np.ndarray:
    return np.where(X[:, 0] ** 2 + X[:, 1] ** 2 < 0.5, 1.0, 0.0)


# Adversarial functions


def runge_function(X: np.ndarray) -> np.ndarray:
    # The Runge function is the classic illustration of limitations
    # of polynomial interpolation: any polynomial surrogate trained on uniformly
    # spaced data exhibits wild oscillations near x = +/-1, even though the
    # function itself is smooth.  Standard MLPs are less susceptible, but a
    # polynomial-feature surrogate (degree >= 5) will diverge near boundaries.
    # This benchmark verifies that the surrogate does not implicitly rely on
    # polynomial-like global approximation.
    return 1.0 / (1.0 + 25.0 * X[:, 0] ** 2)


def high_frequency_oscillation(X: np.ndarray) -> np.ndarray:
    # MLPs trained by gradient descent exhibit a strong "spectral
    # bias" (also called the F-principle): they learn low-frequency components
    # first and fit high-frequency content only slowly, if at all.  A function
    # dominated by a high spatial frequency (10 cycles per unit interval) exposes
    # this bias clearly.  Gradient-enriched training does not resolve spectral
    # bias; the appropriate remedy is Fourier feature encoding or longer training.
    # This benchmark therefore demonstrates a structural limitation shared by
    # both standard and gradient-enriched MLP variants.
    return np.sin(10.0 * np.pi * X[:, 0])


def rosenbrock(X: np.ndarray) -> np.ndarray:
    # The Rosenbrock function has a narrow curved valley whose floor
    # follows the parabola y = x^2.  A standard surrogate trained on function
    # values can achieve an acceptable R^2 by capturing the large-scale shape
    # (steep walls, shallow floor), yet its gradient estimates along the valley
    # are unreliable because the surrogate has no information about the precise
    # curvature.  A gradient-enriched (Sobolev-trained) surrogate is forced to
    # match the analytical partial derivatives during training, which encodes
    # the valley curvature directly.  This translates into higher gradient
    # cosine similarity and, consequently, more reliable surrogate-guided
    # optimisation, adjoint sensitivity analysis, and active learning.
    a, b = 1.0, 100.0
    return (a - X[:, 0]) ** 2 + b * (X[:, 1] - X[:, 0] ** 2) ** 2


def anisotropic_ridge(X: np.ndarray) -> np.ndarray:
    # The gradient of this function differs by three orders of
    # magnitude between the two input dimensions (exponent coefficients 100:0.1).
    # Isotropic surrogates, i.e. RBF kernels with a single length scale and MLPs with
    # uniform weight initialisation will under-weight the steep direction and
    # over-smooth it, producing gradient estimates that are systematically biased
    # toward the shallow direction.  Gradient-enriched training reveals the true
    # gradient anisotropy during fitting, allowing the model's internal
    # representation to adapt to the two vastly different length scales.  This
    # benchmark exposes a weakness that is shared by GPR with an isotropic kernel
    # and by MLPs trained without gradient supervision.
    return np.exp(-100.0 * X[:, 0] ** 2 - 0.1 * X[:, 1] ** 2)


def saddle_landscape(X: np.ndarray) -> np.ndarray:
    # A saddle point at the origin has opposing gradient signs in the
    # two input directions (positive along x_0, negative along x_1),
    # and the cubic perturbation introduces asymmetry.  Surrogates
    # trained on function values alone can fit the values acceptably
    # while producing gradient estimates that point in the wrong
    # direction near the saddle. Gradient-enriched surrogates are
    # expected to better capture the sign structure of the gradient
    # field around the saddle.
    return X[:, 0] ** 2 - X[:, 1] ** 2 + 0.5 * X[:, 0] ** 3


STANDARD_FUNCTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "quadratic": quadratic_bowl,
    "gaussian": gaussian_bump,
    "trig_product": trig_product,
    "rational": rational_func,
    "step_disc": step_disc,
}

ADVERSARIAL_FUNCTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "runge": runge_function,
    "high_frequency": high_frequency_oscillation,
    "rosenbrock": rosenbrock,
    "anisotropic_ridge": anisotropic_ridge,
    "saddle": saddle_landscape,
}

# Functions for which gradient accuracy is a primary metric
GRADIENT_SENSITIVE: List[str] = ["rosenbrock", "anisotropic_ridge", "saddle"]

# Results Container


@dataclass
class BenchmarkRecord:
    """One row in the summary table."""

    benchmark: str
    surrogate: str
    function: str
    r2_gt: float  # ground-truth decoder upper bound
    r2_sur: float  # surrogate decoder performance
    retention: float  # r2_sur / r2_gt (clamped to [-inf, 1])
    grad_cosine_gt: float = float("nan")  # gradient cosine sim (GT or NaN)
    grad_cosine_sur: float = float("nan")  # gradient cosine sim (surrogate or NaN)
    passed: bool = False



class LinearApproximationBenchmark:
    """
    Validates that the surrogate-and-decoder pipeline can recover simple
    linear functions f(x) = a*x + b.

    This is the simplest possible function approximation task.  Any
    surrogate that preserves basic linear information about the inputs
    should pass easily.  A failure here indicates a fundamental
    problem with the stimulus encoding, surrogate training, or
    decoder, not a limitation of approximation capacity.

    Three random linear projections are tested (varying input dimension and
    projection direction) so that the result is not an artifact of a lucky
    choice of direction.
    """

    name = "LinearApproximation"
    PASS_THRESHOLD = 0.90
    N_TEST = 300

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def _make_functions(self) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
        funcs = {}
        for tag in ["proj_a", "proj_b", "proj_c"]:
            A = self._rng.standard_normal(2)
            A = A / np.linalg.norm(A)
            b = float(self._rng.uniform(-1.0, 1.0))
            funcs[tag] = make_linear_function(A, b)
        return funcs

    def run(
        self,
        proxy: BiophysicalModelProxy,
        surrogates: List[_BaseModel],
        n_train: int,
    ) -> List[BenchmarkRecord]:
        rng = self._rng
        X_all = rng.uniform(-1.0, 1.0, (n_train + self.N_TEST, 2))
        X_tr, X_te = X_all[:n_train], X_all[n_train:]
        Z_gt_tr = proxy.respond(X_tr)
        Z_gt_te = proxy.respond(X_te)

        records: List[BenchmarkRecord] = []
        funcs = self._make_functions()

        for sur in surrogates:
            t0 = time.perf_counter()
            sur.fit(X_tr, Z_gt_tr)
            logger.debug("  %s surrogate fit in %.1fs", sur.name, time.perf_counter() - t0)
            Z_sur_tr = sur.predict(X_tr)
            Z_sur_te = sur.predict(X_te)

            for fname, func in funcs.items():
                y_tr = func(X_tr)
                y_te = func(X_te)
                r2_gt_tr, r2_gt = train_linear_decoder(Z_gt_tr, y_tr, Z_gt_te, y_te)
                r2_sur_tr, r2_sur = train_linear_decoder(Z_sur_tr, y_tr, Z_sur_te, y_te)
                retention = r2_sur / r2_gt if abs(r2_gt) > 1e-6 else float("nan")
                rec = BenchmarkRecord(
                    benchmark=self.name,
                    surrogate=sur.name,
                    function=fname,
                    r2_gt=r2_gt,
                    r2_sur=r2_sur,
                    retention=retention,
                    passed=r2_sur >= self.PASS_THRESHOLD,
                )
                records.append(rec)
                logger.info(
                    "  [%s] %s | %s : R^2_GT=%.3f  R^2_sur=%.3f  "
                    "retention=%.2f  %s",
                    self.name, sur.name, fname, r2_gt, r2_sur, retention,
                    "PASS" if rec.passed else "FAIL",
                )

        return records


class BasisFunctionBenchmark:
    """
    Tests whether the surrogate representation preserves the structure of
    signals expressed as linear combinations of known basis functions.

    In neural response analysis, input signals are often decomposed into
    Fourier modes, polynomial expansions, or localised RBF kernels.  A
    surrogate that discards basis-function structure will fail as a substrate
    for frequency-selective or spatially-localised downstream decoders.  This
    benchmark tests three basis families: Fourier, Legendre polynomial, and
    Gaussian RBF.  For each family, five random signals are generated from the
    same basis; the surrogate+decoder pipeline is evaluated on its ability to
    decode each signal's value from the frozen surrogate representation.

    Pass criterion: mean R^2 across all basis types and signals >= 0.80 with
    a linear decoder.  A linear decoder is deliberately used here: if the
    surrogate representation is linear in the basis coefficients, a linear
    decoder suffices; a failure indicates that the relevant information is
    entangled or lost in the representation.
    """

    name = "BasisFunction"
    PASS_THRESHOLD = 0.80
    N_TEST = 300
    N_SIGNALS = 5

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def _make_basis_functions(
        self,
    ) -> Dict[str, List[Callable[[np.ndarray], np.ndarray]]]:
        rng = self._rng
        basis: Dict[str, List[Callable[[np.ndarray], np.ndarray]]] = {}

        # Fourier signals: random coefficients, fixed low frequencies
        freqs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0])
        fourier_sigs = []
        for _ in range(self.N_SIGNALS):
            c = rng.standard_normal(len(freqs))
            c = c / np.linalg.norm(c)
            _c, _f = c.copy(), freqs.copy()
            fourier_sigs.append(
                lambda X, _c=_c, _f=_f: fourier_signal(X, _c, _f)
            )
        basis["fourier"] = fourier_sigs

        # Legendre signals: random coefficients, degrees 1..7
        legendre_sigs = []
        for _ in range(self.N_SIGNALS):
            c = rng.standard_normal(7)
            c = c / np.linalg.norm(c)
            _c = c.copy()
            legendre_sigs.append(lambda X, _c=_c: legendre_signal(X, _c))
        basis["legendre"] = legendre_sigs

        # RBF signals: random centers in [-1,1]^2, random coefficients
        centers = rng.uniform(-0.9, 0.9, (6, 2))
        rbf_sigs = []
        for _ in range(self.N_SIGNALS):
            c = rng.standard_normal(6)
            _c, _ctr = c.copy(), centers.copy()
            rbf_sigs.append(
                lambda X, _c=_c, _ctr=_ctr: rbf_signal(X, _c, _ctr)
            )
        basis["rbf"] = rbf_sigs

        return basis

    def run(
        self,
        proxy: BiophysicalModelProxy,
        surrogates: List[_BaseModel],
        n_train: int,
    ) -> List[BenchmarkRecord]:
        rng = self._rng
        X_all = rng.uniform(-1.0, 1.0, (n_train + self.N_TEST, 2))
        X_tr, X_te = X_all[:n_train], X_all[n_train:]
        Z_gt_tr = proxy.respond(X_tr)
        Z_gt_te = proxy.respond(X_te)
        basis_funcs = self._make_basis_functions()

        records: List[BenchmarkRecord] = []

        for sur in surrogates:
            sur.fit(X_tr, Z_gt_tr)
            Z_sur_tr = sur.predict(X_tr)
            Z_sur_te = sur.predict(X_te)

            for basis_name, sig_list in basis_funcs.items():
                r2_gt_list, r2_sur_list = [], []
                for idx, func in enumerate(sig_list):
                    y_tr = func(X_tr)
                    y_te = func(X_te)
                    _, r2_gt = train_linear_decoder(Z_gt_tr, y_tr, Z_gt_te, y_te)
                    _, r2_sur = train_linear_decoder(Z_sur_tr, y_tr, Z_sur_te, y_te)
                    r2_gt_list.append(r2_gt)
                    r2_sur_list.append(r2_sur)

                r2_gt_mean = float(np.mean(r2_gt_list))
                r2_sur_mean = float(np.mean(r2_sur_list))
                retention = (
                    r2_sur_mean / r2_gt_mean if abs(r2_gt_mean) > 1e-6 else float("nan")
                )
                rec = BenchmarkRecord(
                    benchmark=self.name,
                    surrogate=sur.name,
                    function=basis_name,
                    r2_gt=r2_gt_mean,
                    r2_sur=r2_sur_mean,
                    retention=retention,
                    passed=r2_sur_mean >= self.PASS_THRESHOLD,
                )
                records.append(rec)
                logger.info(
                    "  [%s] %s | %s : R^2_GT=%.3f  R^2_sur=%.3f  "
                    "retention=%.2f  %s",
                    self.name, sur.name, basis_name,
                    r2_gt_mean, r2_sur_mean, retention,
                    "PASS" if rec.passed else "FAIL",
                )

        return records


class NonlinearApproximationBenchmark:
    """
    Tests surrogate quality across a suite of standard nonlinear functions and
    a set of adversarial functions specifically designed to expose surrogate
    weaknesses.

    Standard suite: five smooth-to-discontinuous functions (quadratic,
    Gaussian, trigonometric product, rational, step discontinuity) evaluated
    via the three-phase pipeline.  These progress in difficulty: quadratic and
    Gaussian are well-behaved, trig_product introduces oscillations, rational
    introduces a soft singularity, and step_disc is discontinuous.

    Adversarial suite: five functions chosen to expose specific failure modes.
    For gradient-sensitive functions (Rosenbrock, anisotropic ridge, saddle), a
    direct comparison is run: standard MLP and gradient-enriched MLP are each
    trained directly on (x, f(x)) and (x, f(x), grad f(x)) respectively, and
    gradient cosine similarity is measured on a held-out test set.  This
    comparison demonstrates that gradient co-training is necessary when the
    surrogate will be used for gradient-based operations.

    Pass criteria:
      - Standard functions: R^2_sur >= 0.60
      - Adversarial functions: R^2_sur >= 0.40 (harder targets)
      - Gradient-sensitive: gradient cosine similarity (Sobolev) >= 0.70
    """

    name = "NonlinearApproximation"
    PASS_THRESHOLD_STANDARD = 0.60
    PASS_THRESHOLD_ADVERSARIAL = 0.40
    PASS_THRESHOLD_GRAD_COS = 0.70
    N_TEST = 300

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def _run_pipeline(
        self,
        funcs: Dict[str, Callable[[np.ndarray], np.ndarray]],
        proxy: BiophysicalModelProxy,
        surrogates: List[_BaseModel],
        X_tr: np.ndarray,
        X_te: np.ndarray,
        Z_gt_tr: np.ndarray,
        Z_gt_te: np.ndarray,
        pass_threshold: float,
    ) -> List[BenchmarkRecord]:
        records: List[BenchmarkRecord] = []
        for sur in surrogates:
            sur.fit(X_tr, Z_gt_tr)
            Z_sur_tr = sur.predict(X_tr)
            Z_sur_te = sur.predict(X_te)
            for fname, func in funcs.items():
                y_tr = func(X_tr)
                y_te = func(X_te)
                _, r2_gt = train_linear_decoder(Z_gt_tr, y_tr, Z_gt_te, y_te)
                _, r2_sur = train_linear_decoder(Z_sur_tr, y_tr, Z_sur_te, y_te)
                retention = r2_sur / r2_gt if abs(r2_gt) > 1e-6 else float("nan")
                rec = BenchmarkRecord(
                    benchmark=self.name,
                    surrogate=sur.name,
                    function=fname,
                    r2_gt=r2_gt,
                    r2_sur=r2_sur,
                    retention=retention,
                    passed=r2_sur >= pass_threshold,
                )
                records.append(rec)
                logger.info(
                    "  [%s] %s | %s : R^2_GT=%.3f  R^2_sur=%.3f  "
                    "retention=%.2f  %s",
                    self.name, sur.name, fname, r2_gt, r2_sur, retention,
                    "PASS" if rec.passed else "FAIL",
                )
        return records

    def _run_gradient_comparison(
        self,
        X_tr: np.ndarray,
        X_te: np.ndarray,
        n_train: int,
        seed: int,
    ) -> List[BenchmarkRecord]:
        """
        Direct gradient-accuracy comparison: train standard MLP and
        gradient-enriched MLP directly on each gradient-sensitive function
        (bypassing the proxy pipeline).  Report function R^2 and gradient
        cosine similarity for both models.
        """
        records: List[BenchmarkRecord] = []
        std_mlp = DirectMLPSurrogate(seed=seed)
        sob_mlp = GradientEnrichedSurrogate(seed=seed)

        for fname in GRADIENT_SENSITIVE:
            func = ADVERSARIAL_FUNCTIONS[fname]
            y_tr = func(X_tr)
            y_te = func(X_te)
            true_grads_te = compute_analytical_gradient(func, X_te)

            # Standard MLP
            std_mlp.fit(X_tr, y_tr)
            r2_std = float(r2_score(y_te, std_mlp.predict(X_te)))
            gc_std = gradient_cosine_similarity(std_mlp, X_te, true_grads_te)

            # Sobolev MLP
            y_aug_tr = build_sobolev_targets(func, X_tr)
            sob_mlp.fit(X_tr, y_aug_tr)
            r2_sob = float(r2_score(y_te, sob_mlp.predict(X_te)))
            gc_sob = gradient_cosine_similarity(sob_mlp, X_te, true_grads_te)

            # Record standard MLP as the "surrogate" column, GT=Sobolev as reference
            rec_std = BenchmarkRecord(
                benchmark=f"{self.name}_grad_comparison",
                surrogate=std_mlp.name,
                function=fname,
                r2_gt=r2_sob,
                r2_sur=r2_std,
                retention=r2_std / r2_sob if abs(r2_sob) > 1e-6 else float("nan"),
                grad_cosine_gt=gc_sob,
                grad_cosine_sur=gc_std,
                passed=gc_std >= self.PASS_THRESHOLD_GRAD_COS,
            )
            rec_sob = BenchmarkRecord(
                benchmark=f"{self.name}_grad_comparison",
                surrogate=sob_mlp.name,
                function=fname,
                r2_gt=r2_sob,
                r2_sur=r2_sob,
                retention=1.0,
                grad_cosine_gt=gc_sob,
                grad_cosine_sur=gc_sob,
                passed=gc_sob >= self.PASS_THRESHOLD_GRAD_COS,
            )
            records.extend([rec_std, rec_sob])
            logger.info(
                "  [grad_cmp] %s: std_MLP  R^2=%.3f gc=%.3f | "
                "Sobolev R^2=%.3f gc=%.3f  (pass gc>=%.2f)",
                fname, r2_std, gc_std, r2_sob, gc_sob,
                self.PASS_THRESHOLD_GRAD_COS,
            )

        return records

    def run(
        self,
        proxy: BiophysicalModelProxy,
        surrogates: List[_BaseModel],
        n_train: int,
        seed: int = 0,
    ) -> List[BenchmarkRecord]:
        rng = self._rng
        X_all = rng.uniform(-1.0, 1.0, (n_train + self.N_TEST, 2))
        X_tr, X_te = X_all[:n_train], X_all[n_train:]
        Z_gt_tr = proxy.respond(X_tr)
        Z_gt_te = proxy.respond(X_te)

        records: List[BenchmarkRecord] = []

        logger.info("  --- standard functions ---")
        records += self._run_pipeline(
            STANDARD_FUNCTIONS, proxy, surrogates,
            X_tr, X_te, Z_gt_tr, Z_gt_te,
            self.PASS_THRESHOLD_STANDARD,
        )

        logger.info("  --- adversarial functions (pipeline) ---")
        records += self._run_pipeline(
            ADVERSARIAL_FUNCTIONS, proxy, surrogates,
            X_tr, X_te, Z_gt_tr, Z_gt_te,
            self.PASS_THRESHOLD_ADVERSARIAL,
        )

        logger.info("  --- gradient accuracy comparison (direct training) ---")
        records += self._run_gradient_comparison(X_tr, X_te, n_train, seed)

        return records


class MultiFunctionSuiteBenchmark:
    """
    Tests whether a single decoder trained on pooled data from multiple diverse
    function types can simultaneously decode all of them from the surrogate
    representation.

    This benchmark assesses a prerequisite for universal approximation: the
    surrogate representation must be rich enough that a single linear mapping
    from the surrogate's 16-D output can recover any of K different scalar
    functions.  In the DRC context this corresponds to asking whether a single
    trained surrogate supports decoding an arbitrary downstream behavioral
    variable.

    A multi-output Ridge decoder is trained on all standard nonlinear functions
    simultaneously.  Evaluation is per-function: mean R^2 must exceed 0.50 and
    the worst-case (minimum) R^2 must exceed 0.30.  A large gap between mean
    and minimum R^2 indicates that the surrogate representation is selective
    toward certain function types, violating the universality requirement.
    """

    name = "MultiFunctionSuite"
    PASS_THRESHOLD_MEAN = 0.50
    PASS_THRESHOLD_MIN = 0.30
    N_TEST = 300

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def run(
        self,
        proxy: BiophysicalModelProxy,
        surrogates: List[_BaseModel],
        n_train: int,
    ) -> List[BenchmarkRecord]:
        rng = self._rng
        X_all = rng.uniform(-1.0, 1.0, (n_train + self.N_TEST, 2))
        X_tr, X_te = X_all[:n_train], X_all[n_train:]
        Z_gt_tr = proxy.respond(X_tr)
        Z_gt_te = proxy.respond(X_te)

        func_names = list(STANDARD_FUNCTIONS.keys())
        Y_tr = np.column_stack([f(X_tr) for f in STANDARD_FUNCTIONS.values()])
        Y_te = np.column_stack([f(X_te) for f in STANDARD_FUNCTIONS.values()])

        records: List[BenchmarkRecord] = []

        for sur in surrogates:
            sur.fit(X_tr, Z_gt_tr)
            Z_sur_tr = sur.predict(X_tr)
            Z_sur_te = sur.predict(X_te)

            _, r2_gt_arr = train_linear_decoder(Z_gt_tr, Y_tr, Z_gt_te, Y_te)
            _, r2_sur_arr = train_linear_decoder(Z_sur_tr, Y_tr, Z_sur_te, Y_te)

            # r2_score with multioutput='raw_values' is used implicitly here;
            # recompute per-function for individual records.
            dec_gt = Ridge(alpha=1.0).fit(Z_gt_tr, Y_tr)
            dec_sur = Ridge(alpha=1.0).fit(Z_sur_tr, Y_tr)
            per_gt = r2_score(Y_te, dec_gt.predict(Z_gt_te), multioutput="raw_values")
            per_sur = r2_score(Y_te, dec_sur.predict(Z_sur_te), multioutput="raw_values")

            for k, fname in enumerate(func_names):
                r2_gt_k = float(per_gt[k])
                r2_sur_k = float(per_sur[k])
                retention = r2_sur_k / r2_gt_k if abs(r2_gt_k) > 1e-6 else float("nan")
                rec = BenchmarkRecord(
                    benchmark=self.name,
                    surrogate=sur.name,
                    function=fname,
                    r2_gt=r2_gt_k,
                    r2_sur=r2_sur_k,
                    retention=retention,
                    passed=(
                        float(np.mean(per_sur)) >= self.PASS_THRESHOLD_MEAN
                        and float(np.min(per_sur)) >= self.PASS_THRESHOLD_MIN
                    ),
                )
                records.append(rec)

            mean_r2 = float(np.mean(per_sur))
            min_r2 = float(np.min(per_sur))
            passed = mean_r2 >= self.PASS_THRESHOLD_MEAN and min_r2 >= self.PASS_THRESHOLD_MIN
            logger.info(
                "  [%s] %s : mean R^2=%.3f  min R^2=%.3f  %s",
                self.name, sur.name, mean_r2, min_r2,
                "PASS" if passed else "FAIL",
            )

        return records



def _bar_chart(
    ax: plt.Axes,
    labels: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    threshold: Optional[float] = None,
    colors: Optional[List[str]] = None,
) -> None:
    x = np.arange(len(labels))
    clr = colors if colors else ["steelblue"] * len(labels)
    ax.bar(x, values, color=clr, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if threshold is not None:
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1.0,
                   label=f"threshold={threshold:.2f}")
        ax.legend(fontsize=8)
    ax.set_ylim(bottom=min(0.0, min(v for v in values if np.isfinite(v)) - 0.05))


def plot_benchmark_pipeline_results(
    records: List[BenchmarkRecord],
    benchmark_name: str,
    output_dir: str,
) -> None:
    """Bar charts of R^2 and performance retention for three-phase pipeline benchmarks."""
    sur_names = sorted({r.surrogate for r in records})
    func_names = sorted({r.function for r in records
                         if "grad_comparison" not in r.benchmark})

    if not func_names or not sur_names:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{benchmark_name}: pipeline evaluation", fontsize=12)

    # R^2 per (surrogate, function)
    ax = axes[0]
    n_sur = len(sur_names)
    n_func = len(func_names)
    width = 0.8 / n_sur
    for i, sname in enumerate(sur_names):
        vals = []
        for fname in func_names:
            matching = [r.r2_sur for r in records
                        if r.surrogate == sname and r.function == fname
                        and "grad_comparison" not in r.benchmark]
            vals.append(float(np.mean(matching)) if matching else float("nan"))
        xs = np.arange(n_func) + i * width - (n_sur - 1) * width / 2
        ax.bar(xs, vals, width=width * 0.9, label=sname)
    ax.set_xticks(np.arange(n_func))
    ax.set_xticklabels(func_names, rotation=30, ha="right")
    ax.set_title("$R^2$ by surrogate and function")
    ax.set_ylabel("$R^2$")
    ax.legend(fontsize=7)

    # performance retention per surrogate (mean across functions)
    ax = axes[1]
    ret_vals = []
    for sname in sur_names:
        rets = [r.retention for r in records
                if r.surrogate == sname and np.isfinite(r.retention)
                and "grad_comparison" not in r.benchmark]
        ret_vals.append(float(np.mean(rets)) if rets else float("nan"))
    _bar_chart(
        ax, sur_names, ret_vals,
        "Mean performance retention $R^2_{sur} / R^2_{GT}$",
        "Retention ratio",
        threshold=0.70,
    )

    fig.tight_layout()
    path = os.path.join(output_dir, f"{benchmark_name}_pipeline.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_gradient_comparison(
    records: List[BenchmarkRecord],
    output_dir: str,
) -> None:
    """Paired bar chart comparing function R^2 and gradient cosine similarity."""
    grad_recs = [r for r in records if "grad_comparison" in r.benchmark]
    if not grad_recs:
        return

    func_names = sorted({r.function for r in grad_recs})
    sur_names = ["Shallow MLP", "Gradient-Enriched MLP (Sobolev)"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Gradient-sensitive functions: standard vs. Sobolev surrogate", fontsize=12
    )

    for ax_idx, (metric, ylabel, threshold) in enumerate(
        [
            ("r2_sur", "$R^2$ (function values)", 0.40),
            ("grad_cosine_sur", r"Gradient cosine similarity $\langle\nabla f,\nabla\hat{f}\rangle$", 0.70),
        ]
    ):
        ax = axes[ax_idx]
        width = 0.35
        for i, sname in enumerate(sur_names):
            vals = []
            for fname in func_names:
                matching = [r for r in grad_recs
                            if r.surrogate == sname and r.function == fname]
                if matching:
                    v = getattr(matching[0], metric)
                    vals.append(float(v) if np.isfinite(float(v)) else 0.0)
                else:
                    vals.append(0.0)
            xs = np.arange(len(func_names)) + i * width - width / 2
            ax.bar(xs, vals, width=width * 0.9, label=sname)

        ax.set_xticks(np.arange(len(func_names)))
        ax.set_xticklabels(func_names, rotation=20, ha="right")
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1.0,
                   label=f"threshold={threshold:.2f}")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=7)

    fig.tight_layout()
    path = os.path.join(output_dir, "gradient_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_multi_function_heatmap(
    records: List[BenchmarkRecord],
    output_dir: str,
) -> None:
    """Heatmap of per-function R^2 for each surrogate in the multi-function suite."""
    mf_recs = [r for r in records if r.benchmark == "MultiFunctionSuite"]
    if not mf_recs:
        return

    sur_names = sorted({r.surrogate for r in mf_recs})
    func_names = sorted({r.function for r in mf_recs})

    mat = np.full((len(sur_names), len(func_names)), float("nan"))
    for r in mf_recs:
        i = sur_names.index(r.surrogate)
        j = func_names.index(r.function)
        mat[i, j] = r.r2_sur

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(func_names)))
    ax.set_yticks(range(len(sur_names)))
    ax.set_xticklabels(func_names, rotation=30, ha="right")
    ax.set_yticklabels(sur_names)
    fig.colorbar(im, ax=ax, label="$R^2$")
    for i in range(len(sur_names)):
        for j in range(len(func_names)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Multi-function suite: $R^2$ per (surrogate, function)")
    fig.tight_layout()
    path = os.path.join(output_dir, "multi_function_heatmap.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def build_proxy_surrogates(seed: int) -> List[_BaseModel]:
    return [
        LinearProxySurrogate(),
        MLPProxySurrogate(seed=seed),
    ]


def save_csv(records: List[BenchmarkRecord], output_dir: str) -> None:
    path = os.path.join(output_dir, "results_summary.csv")
    rows = [
        {
            "benchmark": r.benchmark,
            "surrogate": r.surrogate,
            "function": r.function,
            "r2_gt": r.r2_gt,
            "r2_sur": r.r2_sur,
            "retention": r.retention,
            "grad_cosine_gt": r.grad_cosine_gt,
            "grad_cosine_sur": r.grad_cosine_sur,
            "passed": r.passed,
        }
        for r in records
    ]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info("Results written to %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run surrogate stepping stone benchmarks."
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument(
        "--n-train", type=int, default=800, help="Training set size"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures/surrogate_benchmarks",
        help="Directory for output figures and CSV",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["linear", "basis", "nonlinear", "multi", "all"],
        default=["all"],
        help="Which benchmarks to run",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    os.makedirs(args.output_dir, exist_ok=True)

    run_all = "all" in args.benchmarks
    run = {
        "linear": run_all or "linear" in args.benchmarks,
        "basis": run_all or "basis" in args.benchmarks,
        "nonlinear": run_all or "nonlinear" in args.benchmarks,
        "multi": run_all or "multi" in args.benchmarks,
    }

    proxy = BiophysicalModelProxy(seed=args.seed)
    surrogates = build_proxy_surrogates(args.seed)
    all_records: List[BenchmarkRecord] = []

    if run["linear"]:
        logger.info("=== LinearApproximationBenchmark ===")
        bench = LinearApproximationBenchmark(seed=args.seed)
        recs = bench.run(proxy, surrogates, args.n_train)
        all_records += recs
        plot_benchmark_pipeline_results(recs, "LinearApproximation", args.output_dir)

    if run["basis"]:
        logger.info("=== BasisFunctionBenchmark ===")
        bench = BasisFunctionBenchmark(seed=args.seed)
        recs = bench.run(proxy, surrogates, args.n_train)
        all_records += recs
        plot_benchmark_pipeline_results(recs, "BasisFunction", args.output_dir)

    if run["nonlinear"]:
        logger.info("=== NonlinearApproximationBenchmark ===")
        bench = NonlinearApproximationBenchmark(seed=args.seed)
        recs = bench.run(proxy, surrogates, args.n_train, seed=args.seed)
        all_records += recs
        plot_benchmark_pipeline_results(
            [r for r in recs if "grad_comparison" not in r.benchmark],
            "NonlinearApproximation",
            args.output_dir,
        )
        plot_gradient_comparison(recs, args.output_dir)

    if run["multi"]:
        logger.info("=== MultiFunctionSuiteBenchmark ===")
        bench = MultiFunctionSuiteBenchmark(seed=args.seed)
        recs = bench.run(proxy, surrogates, args.n_train)
        all_records += recs
        plot_multi_function_heatmap(recs, args.output_dir)

    save_csv(all_records, args.output_dir)

    # Print a compact summary table
    df = pd.DataFrame(
        [
            {
                "benchmark": r.benchmark,
                "surrogate": r.surrogate,
                "function": r.function,
                "R^2_sur": f"{r.r2_sur:.3f}",
                "retention": f"{r.retention:.2f}" if np.isfinite(r.retention) else "n/a",
                "grad_cos": (
                    f"{r.grad_cosine_sur:.3f}"
                    if np.isfinite(r.grad_cosine_sur)
                    else ""
                ),
                "pass": "Y" if r.passed else "N",
            }
            for r in all_records
        ]
    )
    logger.info("\n%s", df.to_string(index=False))


if __name__ == "__main__":
    main()
