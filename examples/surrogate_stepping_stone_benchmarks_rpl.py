"""
surrogate_stepping_stone_benchmarks_rpl.py
==========================================
RPL (Residual Policy Learning) evaluation version of the stepping stone
benchmarks for surrogate models of biophysical neural responses.

This script implements the four benchmarks from
surrogate_stepping_stone_benchmarks.py but replaces the static
performance-retention metric (PRR) R^2_sur / R^2_GT with a dynamic
Residual Policy Learning evaluation adapted from livn/examples/lambda_example.py
 (LambdaAnalysis, lagrangian_dual_ascent).

Residual Policy Learning (RPL)
-----------
Standard R^2 retention-ratio evaluation answers a one-shot question: "how much
of the ground-truth performance does the surrogate recover?"  RPL provides
a richer, trajectory-based answer: "given the surrogate's predictions as a
fixed starting point, how much residual correction does a small corrector
model need, and how quickly does it converge?"

The framework treats the surrogate's decoded prediction as a base policy
(pi_base) and a small residual MLP as an optimizer policy (pi_theta).
A Lagrange multiplier lambda_t enforces the constraint that the optimizer
does not deviate too far from the base policy:

    L = L_task + lambda_t * KL(pi_theta || pi_base)

    lambda_{t+1} = (1 - alpha) * lambda_t
                   + alpha * max(0, P_t - P_thresh) / KL_ema

where P_t is the R^2 of the combined prediction and KL is the mean squared
residual correction (a Gaussian-policy KL proxy, identical to lambda_example.py).

Lambda rises when performance exceeds the threshold and corrections are small.
The area under the lambda curve over training episodes is the "offloading_integral"
and it is the primary surrogate quality metric:

    offloading_integral high  =>  base policy already satisfies the
                                  constraint; little extra work needed
    offloading_integral low   =>  large or many residual corrections needed
                                  to reach the threshold; much extra work

Relationship to retention ratio
--------------------------------
Both metrics should rank surrogates consistently, but RPL adds:
  1. A trajectory showing when the constraint is first satisfied
     (episodes_to_positive_lambda).
  2. Sensitivity to the cost of residual corrections, not just their outcome.
  3. A threshold that is adaptive: P_thresh = threshold_fraction * R^2_GT,
     so the standard automatically scales to each function's achievable ceiling.

Benchmark code
--------------
Benchmark functions, surrogate classes, and BiophysicalModelProxy are imported
from surrogate_stepping_stone_benchmarks.py.

Benchmark workflow
----------------------------------------------------

    Stimulus x (2-D)
         |
         +---> [Biophys. proxy *]  -- train surrogate once -->   [Surrogate]
         |           |                                                |
         |     z_gt (16-D)                                       z_sur (16-D)
         |           |                                                |
         +---> [Lin. decoder GT] <------ per benchmark ------>  [Lin. decoder S]
                     |                                                |
                 y_base_gt                                        y_base_sur
                                                                      |
                                                            [ResidualPolicy episodes]
                                                                      |
                                                           lambda trajectory + integral

    * Biophys. proxy: frozen random MLP, weights fixed at init with Glorot rule.

Usage
-----
    python surrogate_stepping_stone_benchmarks_rpl.py \\
        --benchmarks all --seed 42 --n-train 600 \\
        --n-episodes 30 --output-dir figures/surrogate_benchmarks_rpl

"""

import argparse
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Import shared components from the existing benchmark script
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from surrogate_stepping_stone_benchmarks import (  # noqa: E402
    ADVERSARIAL_FUNCTIONS,
    STANDARD_FUNCTIONS,
    BiophysicalModelProxy,
    LinearProxySurrogate,
    MLPProxySurrogate,
    fourier_signal,
    legendre_signal,
    rbf_signal,
    train_linear_decoder,
)

logger = logging.getLogger("surrogate_stepping_stone_benchmarks_rpl")

plt.style.use("ggplot")
plt.rc("font", size=10)
plt.rc("axes", titlesize=11, labelsize=10)
plt.rc("legend", fontsize=9)

# Suppress ConvergenceWarning for the residual MLP: each episode intentionally
# runs for a limited number of gradient steps (warm-start), so non-convergence
# per episode is expected and not an error.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Lagrangian Dual Ascent (from lambda_example.py)
def lagrangian_dual_ascent(
    state: dict, metrics: dict, alpha: float = 0.1
) -> float:
    """
    Lagrange multiplier update rule from lambda_example.py, generalised to
    accept a scalar reward P_t (here: R^2 on the eval split).

        lambda_{t+1} = (1 - alpha) * lambda_t
                       + alpha * max(0, P_t - P_thresh) / KL_ema

    Lambda rises when performance exceeds the threshold AND corrections
    (KL) are small.  It falls when either condition is violated.
    """
    old_lambda = state.get("lambda_t", 0.0)
    kl_ema = metrics.get("kl_ema", 0.0)
    eval_reward = metrics.get("eval_reward", 0.0)
    threshold = metrics.get("threshold", 0.0)
    if kl_ema > 1e-8:
        lambda_eq = max(0.0, eval_reward - threshold) / kl_ema
    else:
        lambda_eq = 0.0
    return (1.0 - alpha) * old_lambda + alpha * lambda_eq


class RPLLambdaTracker:
    """
    Tracks the Lagrange multiplier lambda_t over training episodes, adapted
    directly from LambdaAnalysis in lambda_example.py for regression.

    In the RL setting of lambda_example.py, KL is the squared norm of the
    action that deviates from the base policy.  Here, the analogous quantity
    is the mean squared residual correction: KL = mean(r_theta(x)^2) over
    the evaluation set.  Using the mean rather than the sum makes the KL
    independent of evaluation set size.

    The key summary metric is the "offloading integral": the area under the
    lambda curve over all episodes.  A large integral indicates that the
    base surrogate solved the task quickly and with small corrections,
    so lambda rose early and stayed high.  A small integral indicates
    that the surrogate required substantial residual correction throughout
    training.
    """

    def __init__(self, threshold: float = 0.0, kl_ema_beta: float = 0.05) -> None:
        self.state: dict = {"lambda_t": 0.0, "kl_ema": 0.0}
        self.history: List[dict] = []
        self.threshold = threshold
        self.kl_ema_beta = kl_ema_beta

    def on_episode_end(self, episode: int, logs: dict) -> None:
        ep_kl = float(logs.get("kl_penalty", 0.0))
        if ep_kl > 0:
            beta = self.kl_ema_beta
            old = self.state.get("kl_ema", 0.0)
            self.state["kl_ema"] = ep_kl if old == 0 else (1 - beta) * old + beta * ep_kl
        self.state["episode"] = episode
        self.history.append(
            {
                "episode": episode,
                "lambda_t": self.state["lambda_t"],
                "kl_penalty": ep_kl,
                "task_reward": float(logs.get("reward", 0.0)),
            }
        )

    def on_eval(self, episode: int, eval_reward: float, alpha: float = 0.1) -> None:
        metrics = {
            "eval_reward": eval_reward,
            "threshold": self.threshold,
            "kl_ema": self.state.get("kl_ema", 0.0),
        }
        self.state["lambda_t"] = lagrangian_dual_ascent(self.state, metrics, alpha=alpha)

    def summary(self) -> dict:
        if not self.history:
            return {
                "lambda_max": 0.0,
                "biological_learning_rate": 0.0,
                "offloading_integral": 0.0,
                "n_lambda_updates": 0,
                "final_lambda_t": 0.0,
            }
        episodes = np.array([r["episode"] for r in self.history], dtype=float)
        values = np.array([r["lambda_t"] for r in self.history])
        lam_max = float(np.max(values))
        tau_bio = 0.0
        if len(values) >= 2:
            w = min(5, len(values))
            smoothed = np.convolve(values, np.ones(w) / w, mode="valid")
            smooth_ep = episodes[w - 1 :]
            if len(smooth_ep) >= 2:
                tau_bio = float(np.polyfit(smooth_ep, smoothed, 1)[0])
        integral = float(np.trapezoid(values, episodes))
        n_updates = sum(
            1
            for i in range(1, len(self.history))
            if self.history[i]["lambda_t"] != self.history[i - 1]["lambda_t"]
        )
        return {
            "lambda_max": lam_max,
            "biological_learning_rate": tau_bio,
            "offloading_integral": integral,
            "n_lambda_updates": n_updates,
            "final_lambda_t": float(values[-1]),
        }


class ResidualPolicy:
    """
    Small MLP that models the correction needed on top of the surrogate's
    base prediction: r_theta(x) = y_true(x) - y_base(x).

    The combined prediction is pi_theta(x) = y_base(x) + r_theta(x).

    Warm-start training: each call to fit_episode() runs exactly
    iterations_per_episode gradient steps, continuing from the previous
    episode's fitted weights.  This is the regression analogue of running
    one RL episode, i.e. the residual model accumulates capacity over episodes,
    and the lambda tracker observes whether this capacity is being used.

    KL proxy: the mean squared correction mean(r_theta(x)^2) over the
    evaluation set plays the role of the Gaussian-policy KL divergence from
    lambda_example.py.  Small KL means the residual barely deviates from
    the base prediction; large KL means large corrections are needed.
    """

    def __init__(
        self,
        hidden: Tuple[int, ...] = (32,),
        iterations_per_episode: int = 50,
        seed: int = 0,
    ) -> None:
        self._model = MLPRegressor(
            hidden_layer_sizes=hidden,
            max_iter=iterations_per_episode,
            warm_start=True,
            random_state=seed,
            early_stopping=False,
        )
        self._scaler = StandardScaler()
        self._fitted = False

    def fit_episode(self, X: np.ndarray, residuals: np.ndarray) -> None:
        """Run one warm-start episode of gradient training on the residuals."""
        if not self._fitted:
            Xs = self._scaler.fit_transform(X)
            self._fitted = True
        else:
            Xs = self._scaler.transform(X)
        self._model.fit(Xs, residuals)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.zeros(len(X))
        return self._model.predict(self._scaler.transform(X))

    def kl_cost(self, X: np.ndarray) -> float:
        """Mean squared correction over X (per-sample KL proxy)."""
        return float(np.mean(self.predict(X) ** 2))


@dataclass
class RPLRecord:
    """One row in the RPL summary table."""

    benchmark: str
    surrogate: str
    function: str
    r2_gt: float                        # GT decoder upper bound
    r2_base: float                      # base surrogate without residual
    r2_final: float                     # surrogate + converged residual
    offloading_integral: float          # main RPL metric (higher = less extra work)
    final_lambda: float
    lambda_max: float
    episodes_to_positive_lambda: int    # lower = surrogate needed fewer corrections
    lambda_history: List[float] = field(default_factory=list)
    r2_history: List[float] = field(default_factory=list)


class RPLEvaluator:
    """
    Runs the full RPL episode loop for one (surrogate, benchmark function) pair.

    The evaluation is structured as follows:
      1. The surrogate's linear decoder predictions y_base form the base policy.
      2. A ResidualPolicy MLP is trained progressively over n_episodes warm-start
         episodes, each adding iterations_per_episode gradient steps.
      3. At each episode, the combined prediction y_base + r_theta is evaluated
         on a held-out split; P_t = R^2(y_true, y_base + r_theta).
      4. The Lagrange multiplier lambda_t is updated via dual ascent.

    The performance threshold is set adaptively:
        P_thresh = threshold_fraction * R^2_GT
    
    This ensures the standard scales with each function's achievable ceiling.
    A function where both GT and surrogate achieve R^2 ~ 0 (e.g. high-frequency
    oscillation, which the proxy cannot encode) will have a near-zero threshold,
    so lambda rises immediately and reports a high offloading_integral, thus
    indicating that no extra residual work is needed because the task is
    fundamentally unlearnable through the proxy.
    """

    def __init__(
        self,
        residual_hidden: Tuple[int, ...] = (32,),
        n_episodes: int = 30,
        iterations_per_episode: int = 50,
        threshold_fraction: float = 0.85,
        lambda_alpha: float = 0.1,
        kl_ema_beta: float = 0.05,
        seed: int = 0,
    ) -> None:
        self.residual_hidden = residual_hidden
        self.n_episodes = n_episodes
        self.iterations_per_episode = iterations_per_episode
        self.threshold_fraction = threshold_fraction
        self.lambda_alpha = lambda_alpha
        self.kl_ema_beta = kl_ema_beta
        self.seed = seed

    def run(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_ev: np.ndarray,
        y_ev: np.ndarray,
        y_base_tr: np.ndarray,
        y_base_ev: np.ndarray,
        r2_gt: float,
        surrogate_name: str,
        benchmark_name: str,
        function_name: str,
    ) -> RPLRecord:
        p_thresh = self.threshold_fraction * max(r2_gt, 0.0)

        residual = ResidualPolicy(
            hidden=self.residual_hidden,
            iterations_per_episode=self.iterations_per_episode,
            seed=self.seed,
        )
        tracker = RPLLambdaTracker(threshold=p_thresh, kl_ema_beta=self.kl_ema_beta)

        residuals_tr = y_tr - y_base_tr
        r2_base = float(r2_score(y_ev, y_base_ev))

        lambda_history: List[float] = []
        r2_history: List[float] = []

        for episode in range(self.n_episodes):
            residual.fit_episode(X_tr, residuals_tr)
            y_combined = y_base_ev + residual.predict(X_ev)
            P_t = float(r2_score(y_ev, y_combined))
            kl = residual.kl_cost(X_ev)

            logs = {"kl_penalty": kl, "reward": P_t}
            tracker.on_episode_end(episode, logs)
            tracker.on_eval(episode, P_t, alpha=self.lambda_alpha)

            lambda_history.append(tracker.state["lambda_t"])
            r2_history.append(P_t)

        summary = tracker.summary()
        r2_final = r2_history[-1] if r2_history else r2_base
        lambda_arr = np.array(lambda_history)
        ep_to_pos = int(np.argmax(lambda_arr > 0)) if np.any(lambda_arr > 0) else self.n_episodes

        logger.debug(
            "  [%s] %s | %s : R^2_base=%.3f  R^2_final=%.3f  "
            "integral=%.2f  ep_to_lam>0=%d",
            benchmark_name, surrogate_name, function_name,
            r2_base, r2_final,
            summary["offloading_integral"], ep_to_pos,
        )

        return RPLRecord(
            benchmark=benchmark_name,
            surrogate=surrogate_name,
            function=function_name,
            r2_gt=r2_gt,
            r2_base=r2_base,
            r2_final=r2_final,
            offloading_integral=summary["offloading_integral"],
            final_lambda=summary["final_lambda_t"],
            lambda_max=summary["lambda_max"],
            episodes_to_positive_lambda=ep_to_pos,
            lambda_history=lambda_history,
            r2_history=r2_history,
        )


# Benchmark Classes (RPL variants)


def _build_surrogate_predictions(
    proxy: BiophysicalModelProxy,
    surrogate,
    X_train: np.ndarray,
    X_eval: np.ndarray,
    y_train: np.ndarray,
    y_eval: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Shared helper: train linear decoders on GT and surrogate representations,
    return (y_base_train, y_base_eval, r2_gt).

    y_base is the surrogate decoder's prediction on the training and eval
    splits; r2_gt is the ground-truth decoder's R^2 (the ceiling for RPL).
    """
    Z_gt_train = proxy.respond(X_train)
    Z_gt_eval = proxy.respond(X_eval)
    Z_sur_train = surrogate.predict(X_train)
    Z_sur_eval = surrogate.predict(X_eval)

    dec_gt = Ridge(alpha=1.0).fit(Z_gt_train, y_train)
    dec_sur = Ridge(alpha=1.0).fit(Z_sur_train, y_train)

    y_base_train = dec_sur.predict(Z_sur_train)
    y_base_eval = dec_sur.predict(Z_sur_eval)
    r2_gt = float(r2_score(y_eval, dec_gt.predict(Z_gt_eval)))

    return y_base_train, y_base_eval, r2_gt


class LinearApproximationBenchmarkRPL:
    """
    RPL evaluation of the linear function approximation benchmark.

    Tests whether the surrogate preserves basic linear structure.
    A well-functioning pipeline should show lambda rising within
    a few episodes (surrogate already meets the threshold with small corrections)
    and a high offloading_integral. If lambda stays near zero throughout,
    this would indicate a fundamental problem in the benchmark pipeline.
    """

    name = "LinearApproximationRPL"
    N_TEST = 300

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def run(
        self,
        proxy: BiophysicalModelProxy,
        surrogates: list,
        n_train: int,
        evaluator: RPLEvaluator,
    ) -> List[RPLRecord]:
        rng = self._rng
        X_all = rng.uniform(-1.0, 1.0, (n_train + self.N_TEST, 2))
        X_tr, X_ev = X_all[:n_train], X_all[n_train:]

        # Fit proxy surrogates on proxy responses
        Z_gt_tr = proxy.respond(X_tr)
        for sur in surrogates:
            sur.fit(X_tr, Z_gt_tr)

        # Generate three random linear functions
        funcs = {}
        for tag in ["proj_a", "proj_b", "proj_c"]:
            A = rng.standard_normal(2)
            A = A / np.linalg.norm(A)
            b = float(rng.uniform(-1.0, 1.0))
            _A, _b = A.copy(), b
            funcs[tag] = lambda X, _A=_A, _b=_b: (X @ _A + _b).ravel()

        records: List[RPLRecord] = []
        for sur in surrogates:
            for fname, func in funcs.items():
                y_tr = func(X_tr)
                y_ev = func(X_ev)
                y_base_tr, y_base_ev, r2_gt = _build_surrogate_predictions(
                    proxy, sur, X_tr, X_ev, y_tr, y_ev
                )
                rec = evaluator.run(
                    X_tr, y_tr, X_ev, y_ev, y_base_tr, y_base_ev, r2_gt,
                    sur.name, self.name, fname,
                )
                records.append(rec)
        return records


class BasisFunctionBenchmarkRPL:
    """
    RPL evaluation of the basis function detection benchmark.

    Tests whether the surrogate representation preserves the structure of
    signals expressed as superpositions of known basis functions (Fourier,
    Legendre, Gaussian RBF).  A surrogate that retains structured information
    should show higher offloading_integral than one that collapses basis-
    function structure into undifferentiated noise.  Lambda convergence speed
    (episodes_to_positive_lambda) indicates how quickly the residual can
    compensate for any lost structure.
    """

    name = "BasisFunctionRPL"
    N_TEST = 300
    N_SIGNALS = 5

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def _make_basis_functions(self) -> Dict[str, List[Callable]]:
        rng = self._rng
        basis: Dict[str, List[Callable]] = {}

        freqs = np.array([1.0, 2.0, 3.0, 5.0, 7.0])
        fourier_sigs = []
        for _ in range(self.N_SIGNALS):
            c = rng.standard_normal(len(freqs))
            c = c / np.linalg.norm(c)
            _c, _f = c.copy(), freqs.copy()
            fourier_sigs.append(lambda X, _c=_c, _f=_f: fourier_signal(X, _c, _f))
        basis["fourier"] = fourier_sigs

        legendre_sigs = []
        for _ in range(self.N_SIGNALS):
            c = rng.standard_normal(5)
            c = c / np.linalg.norm(c)
            _c = c.copy()
            legendre_sigs.append(lambda X, _c=_c: legendre_signal(X, _c))
        basis["legendre"] = legendre_sigs

        centers = rng.uniform(-0.8, 0.8, (6, 2))
        rbf_sigs = []
        for _ in range(self.N_SIGNALS):
            c = rng.standard_normal(6)
            _c, _ctr = c.copy(), centers.copy()
            rbf_sigs.append(lambda X, _c=_c, _ctr=_ctr: rbf_signal(X, _c, _ctr))
        basis["rbf"] = rbf_sigs

        return basis

    def run(
        self,
        proxy: BiophysicalModelProxy,
        surrogates: list,
        n_train: int,
        evaluator: RPLEvaluator,
    ) -> List[RPLRecord]:
        rng = self._rng
        X_all = rng.uniform(-1.0, 1.0, (n_train + self.N_TEST, 2))
        X_tr, X_ev = X_all[:n_train], X_all[n_train:]
        Z_gt_tr = proxy.respond(X_tr)
        for sur in surrogates:
            sur.fit(X_tr, Z_gt_tr)

        basis_funcs = self._make_basis_functions()
        records: List[RPLRecord] = []

        for sur in surrogates:
            for basis_name, sig_list in basis_funcs.items():
                # Aggregate over N_SIGNALS signals of the same basis type
                integral_list, ep_list, r2b_list, r2f_list, r2gt_list = [], [], [], [], []
                for func in sig_list:
                    y_tr, y_ev = func(X_tr), func(X_ev)
                    y_base_tr, y_base_ev, r2_gt = _build_surrogate_predictions(
                        proxy, sur, X_tr, X_ev, y_tr, y_ev
                    )
                    rec = evaluator.run(
                        X_tr, y_tr, X_ev, y_ev, y_base_tr, y_base_ev, r2_gt,
                        sur.name, self.name, f"{basis_name}_signal",
                    )
                    integral_list.append(rec.offloading_integral)
                    ep_list.append(rec.episodes_to_positive_lambda)
                    r2b_list.append(rec.r2_base)
                    r2f_list.append(rec.r2_final)
                    r2gt_list.append(rec.r2_gt)

                # Summarise as mean over signals of the same basis type
                records.append(RPLRecord(
                    benchmark=self.name,
                    surrogate=sur.name,
                    function=basis_name,
                    r2_gt=float(np.mean(r2gt_list)),
                    r2_base=float(np.mean(r2b_list)),
                    r2_final=float(np.mean(r2f_list)),
                    offloading_integral=float(np.mean(integral_list)),
                    final_lambda=float("nan"),
                    lambda_max=float("nan"),
                    episodes_to_positive_lambda=int(np.mean(ep_list)),
                ))
                logger.info(
                    "  [%s] %s | %s mean: integral=%.2f  ep_to_lam>0=%.1f",
                    self.name, sur.name, basis_name,
                    float(np.mean(integral_list)), float(np.mean(ep_list)),
                )
        return records


class NonlinearApproximationBenchmarkRPL:
    """
    RPL evaluation of the nonlinear function approximation benchmark.

    Covers both the standard nonlinear suite (quadratic, Gaussian, etc.) and
    the adversarial suite (Rosenbrock, anisotropic ridge, saddle, etc.).  The
    RPL metric reveals not only whether the surrogate reaches the threshold but
    also the trajectory: a surrogate with high R^2_base may cause lambda to
    rise early (low episodes_to_positive_lambda) while a surrogate whose base
    prediction is far below the threshold will require many episodes of residual
    correction before lambda first becomes positive.

    For the high-frequency oscillation function the GT ceiling is near zero
    (R^2_GT ~ 0.02) because the proxy cannot encode this frequency.  The
    threshold P_thresh = 0.85 * 0.02 ~ 0.017 is therefore nearly zero, and
    any surrogate will exceed it quickly.  The offloading_integral will be
    high for both surrogates, thus indicating that no *extra* work is
    needed relative to what the proxy can achieve, not that the surrogate is
    "good" in an absolute sense.  R^2_base should be inspected alongside the
    RPL metrics to distinguish this degenerate case.
    """

    name = "NonlinearApproximationRPL"
    N_TEST = 300

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def run(
        self,
        proxy: BiophysicalModelProxy,
        surrogates: list,
        n_train: int,
        evaluator: RPLEvaluator,
    ) -> List[RPLRecord]:
        rng = self._rng
        X_all = rng.uniform(-1.0, 1.0, (n_train + self.N_TEST, 2))
        X_tr, X_ev = X_all[:n_train], X_all[n_train:]
        Z_gt_tr = proxy.respond(X_tr)
        for sur in surrogates:
            sur.fit(X_tr, Z_gt_tr)

        records: List[RPLRecord] = []
        all_funcs = {**STANDARD_FUNCTIONS, **ADVERSARIAL_FUNCTIONS}

        for sur in surrogates:
            for fname, func in all_funcs.items():
                y_tr, y_ev = func(X_tr), func(X_ev)
                y_base_tr, y_base_ev, r2_gt = _build_surrogate_predictions(
                    proxy, sur, X_tr, X_ev, y_tr, y_ev
                )
                rec = evaluator.run(
                    X_tr, y_tr, X_ev, y_ev, y_base_tr, y_base_ev, r2_gt,
                    sur.name, self.name, fname,
                )
                records.append(rec)
                logger.info(
                    "  [%s] %s | %s : R^2_base=%.3f  R^2_final=%.3f  "
                    "integral=%.2f  episodes_to_lambda>0=%d",
                    self.name, sur.name, fname, rec.r2_base, rec.r2_final,
                    rec.offloading_integral, rec.episodes_to_positive_lambda,
                )
        return records


class MultiFunctionSuiteBenchmarkRPL:
    """
    RPL evaluation with a shared multi-output residual model.

    This benchmark differs from NonlinearApproximationBenchmarkRPL in that
    instead of training K independent residual models (one per function),
    it trains a single multi-output MLP residual that must simultaneously
    correct the surrogate's predictions for all K standard functions.
    The residual model has the same total hidden-layer capacity as the
    per-function models, forcing it to allocate representation across
    all functions at once.

    The RPL interpretation carries over: if the surrogate representation is
    broadly informative about the inputs, a shared residual can correct all
    functions efficiently (low KL cost per function, high offloading integral).
    If the surrogate's representation only supports a subset of function types,
    the shared residual will be pulled toward those functions, leaving others
    under-corrected.  A high variance in per-function offloading_integral
    therefore diagnoses selective rather than universal representation
    (indicator of limited approximation capacity).

    The lambda tracker uses the minimum per-function performance as the
    reward signal.  This "worst-case" criterion ensures lambda only rises when
    all functions are being handled well, which is the correct test for
    universal approximation capacity.
    """

    name = "MultiFunctionSuiteRPL"
    N_TEST = 300

    def __init__(self, seed: int = 0) -> None:
        # Use a different RNG stream than NonlinearApproximation to generate
        # independent data, making this benchmark's data distinct.
        self._rng = np.random.default_rng(seed + 1000)

    def run(
        self,
        proxy: BiophysicalModelProxy,
        surrogates: list,
        n_train: int,
        evaluator: RPLEvaluator,
    ) -> List[RPLRecord]:
        rng = self._rng
        X_all = rng.uniform(-1.0, 1.0, (n_train + self.N_TEST, 2))
        X_tr, X_ev = X_all[:n_train], X_all[n_train:]
        Z_gt_tr = proxy.respond(X_tr)
        for sur in surrogates:
            sur.fit(X_tr, Z_gt_tr)

        func_names = list(STANDARD_FUNCTIONS.keys())
        funcs = list(STANDARD_FUNCTIONS.values())
        K = len(funcs)

        Y_tr = np.column_stack([f(X_tr) for f in funcs])  # (n_train, K)
        Y_ev = np.column_stack([f(X_ev) for f in funcs])  # (n_test, K)

        records: List[RPLRecord] = []

        for sur in surrogates:
            Z_sur_tr = sur.predict(X_tr)
            Z_sur_ev = sur.predict(X_ev)
            Z_gt_tr_ = proxy.respond(X_tr)
            Z_gt_ev = proxy.respond(X_ev)

            # Build per-function base predictions and GT R^2 values
            Y_base_tr = np.zeros_like(Y_tr)
            Y_base_ev = np.zeros_like(Y_ev)
            r2_gt_per_func = np.zeros(K)
            for k in range(K):
                dec_gt = Ridge(alpha=1.0).fit(Z_gt_tr_, Y_tr[:, k])
                dec_sur = Ridge(alpha=1.0).fit(Z_sur_tr, Y_tr[:, k])
                Y_base_tr[:, k] = dec_sur.predict(Z_sur_tr)
                Y_base_ev[:, k] = dec_sur.predict(Z_sur_ev)
                r2_gt_per_func[k] = float(
                    r2_score(Y_ev[:, k], dec_gt.predict(Z_gt_ev))
                )

            # Per-function lambda thresholds
            p_thresholds = evaluator.threshold_fraction * np.maximum(r2_gt_per_func, 0.0)

            # Shared multi-output residual model
            residual = MLPRegressor(
                hidden_layer_sizes=evaluator.residual_hidden,
                max_iter=evaluator.iterations_per_episode,
                warm_start=True,
                random_state=evaluator.seed,
                early_stopping=False,
            )
            residual_scaler = StandardScaler()
            residual_fitted = False

            # Use worst-case (minimum) performance as the lambda reward signal;
            # average KL across function outputs
            tracker = RPLLambdaTracker(
                threshold=float(np.min(p_thresholds)),
                kl_ema_beta=evaluator.kl_ema_beta,
            )

            Y_residuals_tr = Y_tr - Y_base_tr

            lambda_history: List[float] = []
            r2_min_history: List[float] = []

            for episode in range(evaluator.n_episodes):
                # Warm-start train shared residual on all function residuals
                Xs = (
                    residual_scaler.fit_transform(X_tr)
                    if not residual_fitted
                    else residual_scaler.transform(X_tr)
                )
                residual_fitted = True
                residual.fit(Xs, Y_residuals_tr)

                # Per-function evaluation
                Xs_ev = residual_scaler.transform(X_ev)
                corrections_ev = residual.predict(Xs_ev)  # (n_test, K)
                Y_combined_ev = Y_base_ev + corrections_ev

                per_func_r2 = np.array([
                    float(r2_score(Y_ev[:, k], Y_combined_ev[:, k]))
                    for k in range(K)
                ])
                P_t = float(np.min(per_func_r2))  # worst-case performance
                kl = float(np.mean(corrections_ev ** 2))  # mean squared correction

                logs = {"kl_penalty": kl, "reward": P_t}
                tracker.on_episode_end(episode, logs)
                tracker.on_eval(episode, P_t, alpha=evaluator.lambda_alpha)
                lambda_history.append(tracker.state["lambda_t"])
                r2_min_history.append(P_t)

            summary = tracker.summary()
            lambda_arr = np.array(lambda_history)
            ep_to_pos = (
                int(np.argmax(lambda_arr > 0))
                if np.any(lambda_arr > 0)
                else evaluator.n_episodes
            )

            # Emit one RPLRecord per function with shared residual final R^2
            Xs_ev = residual_scaler.transform(X_ev)
            corrections_final = residual.predict(Xs_ev)
            Y_final_ev = Y_base_ev + corrections_final

            for k, fname in enumerate(func_names):
                r2_final_k = float(r2_score(Y_ev[:, k], Y_final_ev[:, k]))
                r2_base_k = float(r2_score(Y_ev[:, k], Y_base_ev[:, k]))
                records.append(RPLRecord(
                    benchmark=self.name,
                    surrogate=sur.name,
                    function=fname,
                    r2_gt=float(r2_gt_per_func[k]),
                    r2_base=r2_base_k,
                    r2_final=r2_final_k,
                    offloading_integral=summary["offloading_integral"],
                    final_lambda=summary["final_lambda_t"],
                    lambda_max=summary["lambda_max"],
                    episodes_to_positive_lambda=ep_to_pos,
                    lambda_history=lambda_history,
                    r2_history=r2_min_history,
                ))

            mean_r2_base = float(np.mean([r2_score(Y_ev[:, k], Y_base_ev[:, k]) for k in range(K)]))
            mean_r2_final = float(np.mean([r2_score(Y_ev[:, k], Y_final_ev[:, k]) for k in range(K)]))
            logger.info(
                "  [%s] %s : mean R^2_base=%.3f  mean R^2_final=%.3f  "
                "integral=%.2f  episodes_to_lambda>0=%d",
                self.name, sur.name, mean_r2_base, mean_r2_final,
                summary["offloading_integral"], ep_to_pos,
            )
        return records



def plot_lambda_trajectories(
    records: List[RPLRecord],
    benchmark_name: str,
    output_dir: str,
    n_episodes: int,
) -> None:
    """Lambda_t over episodes for each (surrogate, function)."""
    bench_recs = [r for r in records if r.benchmark == benchmark_name and r.lambda_history]
    if not bench_recs:
        return

    func_names = sorted({r.function for r in bench_recs})
    sur_names = sorted({r.surrogate for r in bench_recs})
    n_funcs = len(func_names)
    n_cols = min(4, n_funcs)
    n_rows = (n_funcs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)
    fig.suptitle(f"{benchmark_name}: $\\lambda_t$ trajectories", fontsize=12)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax_idx, fname in enumerate(func_names):
        row, col = divmod(ax_idx, n_cols)
        ax = axes[row][col]
        for s_idx, sname in enumerate(sur_names):
            matching = [r for r in bench_recs if r.function == fname and r.surrogate == sname]
            if matching:
                lam = matching[0].lambda_history
                ax.plot(range(len(lam)), lam, label=sname, color=colors[s_idx % len(colors)])
        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
        ax.set_title(fname, fontsize=9)
        ax.set_xlabel("Episode")
        ax.set_ylabel("$\\lambda_t$")
        ax.legend(fontsize=9)

    # Hide unused axes
    for ax_idx in range(n_funcs, n_rows * n_cols):
        row, col = divmod(ax_idx, n_cols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    path = os.path.join(output_dir, f"{benchmark_name}_lambda.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_offloading_integral(
    records: List[RPLRecord],
    benchmark_name: str,
    output_dir: str,
) -> None:
    """Grouped bar chart of offloading_integral per (surrogate, function)."""
    bench_recs = [r for r in records if r.benchmark == benchmark_name]
    if not bench_recs:
        return

    func_names = sorted({r.function for r in bench_recs})
    sur_names = sorted({r.surrogate for r in bench_recs})
    n_sur = len(sur_names)
    width = 0.8 / n_sur

    fig, ax = plt.subplots(figsize=(max(8, len(func_names) * 1.5), 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, sname in enumerate(sur_names):
        vals = []
        for fname in func_names:
            m = [r.offloading_integral for r in bench_recs
                 if r.surrogate == sname and r.function == fname]
            vals.append(float(np.mean(m)) if m else 0.0)
        xs = np.arange(len(func_names)) + i * width - (n_sur - 1) * width / 2
        ax.bar(xs, vals, width=width * 0.9, label=sname, color=colors[i % len(colors)])

    ax.set_xticks(np.arange(len(func_names)))
    ax.set_xticklabels(func_names, rotation=30, ha="right")
    ax.set_ylabel("Offloading integral $\\int \\lambda_t\\,dt$")
    ax.set_title(f"{benchmark_name}: surrogate quality (higher = less extra work)")
    ax.legend(fontsize=9)
    fig.tight_layout()

    path = os.path.join(output_dir, f"{benchmark_name}_offloading.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def _make_vis_grid(n: int = 40) -> np.ndarray:
    """Return X_grid (n*n, 2) for a regular grid over [-1, 1]^2."""
    x = np.linspace(-1.0, 1.0, n)
    X0, X1 = np.meshgrid(x, x)
    return np.column_stack([X0.ravel(), X1.ravel()])


def plot_residual_field_maps(
    proxy: BiophysicalModelProxy,
    surrogates: list,
    functions: Dict[str, Callable[[np.ndarray], np.ndarray]],
    n_train: int,
    seed: int,
    output_dir: str,
    filename_prefix: str,
    suptitle_prefix: str,
    evaluator: "RPLEvaluator",
    grid_n: int = 40,
) -> None:
    """
    For each surrogate, save a figure with one row per function and four columns:

      True f(x)  |  Base prediction y_base  |  Initial residual |y - y_base|  |  Final prediction

    'Base prediction' is the surrogate+decoder output before any residual correction.
    'Initial residual' is the absolute error |y_true - y_base| on the grid, revealing
    the spatial structure of what the residual MLP must learn.
    'Final prediction' is the combined output after training the residual MLP for
    evaluator.n_episodes episodes.

    A good surrogate has a small, structureless initial residual field: the residual
    only needs to fine-tune.  A poor surrogate has a large, structured field: the
    residual must re-learn the entire function, and the offloading_integral stays low.

    Surrogates are re-fitted on a fresh training set drawn with `seed`.
    """
    rng = np.random.default_rng(seed)
    X_tr = rng.uniform(-1.0, 1.0, (n_train, 2))
    Z_gt_tr = proxy.respond(X_tr)

    for sur in surrogates:
        sur.fit(X_tr, Z_gt_tr)

    X_grid = _make_vis_grid(grid_n)

    col_headers = [
        "True $f(x)$",
        "Base prediction $\\hat{y}_{\\mathrm{base}}$",
        "Initial residual $|y - \\hat{y}_{\\mathrm{base}}|$",
        "Final prediction $\\hat{y}_{\\mathrm{base}} + r_{\\theta}$",
    ]

    for sur in surrogates:
        Z_sur_tr = sur.predict(X_tr)
        Z_sur_grid = sur.predict(X_grid)

        n_func = len(functions)
        fig, axes = plt.subplots(
            n_func, 4,
            figsize=(4 * 2.8, n_func * 2.5),
            squeeze=False,
        )

        for row, (fname, func) in enumerate(functions.items()):
            y_tr = func(X_tr)
            y_true_grid = func(X_grid).reshape(grid_n, grid_n)

            dec_sur = Ridge(alpha=1.0).fit(Z_sur_tr, y_tr)
            y_base_tr = dec_sur.predict(Z_sur_tr)
            y_base_grid = dec_sur.predict(Z_sur_grid).reshape(grid_n, grid_n)

            residual_init_grid = np.abs(y_true_grid - y_base_grid)

            residual = ResidualPolicy(
                hidden=evaluator.residual_hidden,
                iterations_per_episode=evaluator.iterations_per_episode,
                seed=evaluator.seed,
            )
            residuals_tr = y_tr - y_base_tr
            for _ in range(evaluator.n_episodes):
                residual.fit_episode(X_tr, residuals_tr)
            y_final_grid = y_base_grid + residual.predict(X_grid).reshape(grid_n, grid_n)

            vmin = float(y_true_grid.min())
            vmax = float(y_true_grid.max())
            panels = [y_true_grid, y_base_grid, residual_init_grid, y_final_grid]

            for col, data in enumerate(panels):
                ax = axes[row, col]
                if col == 2:
                    im = ax.imshow(
                        data, origin="lower", extent=[-1, 1, -1, 1],
                        vmin=0.0, vmax=float(residual_init_grid.max()),
                        cmap="Reds", aspect="auto",
                    )
                else:
                    im = ax.imshow(
                        data, origin="lower", extent=[-1, 1, -1, 1],
                        vmin=vmin, vmax=vmax, cmap="RdBu_r", aspect="auto",
                    )
                if row == 0:
                    ax.set_title(col_headers[col], fontsize=9)
                if col == 0:
                    ax.set_ylabel(fname, fontsize=9)
                ax.set_xticks([-1, 0, 1])
                ax.set_yticks([-1, 0, 1])
                ax.tick_params(labelsize=7)
                if col == 3:
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        sur_label = (
            sur.name.replace(" ", "_")
            .replace("(", "").replace(")", "")
            .replace("/", "_")
        )
        fig.suptitle(f"{suptitle_prefix} — {sur.name}", fontsize=11)
        fig.tight_layout()
        path = os.path.join(output_dir, f"{filename_prefix}_{sur_label}.png")
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        logger.info("Saved %s", path)


def save_csv(records: List[RPLRecord], output_dir: str) -> None:
    rows = [
        {
            "benchmark": r.benchmark,
            "surrogate": r.surrogate,
            "function": r.function,
            "r2_gt": r.r2_gt,
            "r2_base": r.r2_base,
            "r2_final": r.r2_final,
            "offloading_integral": r.offloading_integral,
            "final_lambda": r.final_lambda,
            "lambda_max": r.lambda_max,
            "episodes_to_positive_lambda": r.episodes_to_positive_lambda,
        }
        for r in records
    ]
    path = os.path.join(output_dir, "rpl_results_summary.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info("Results written to %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RPL evaluation of surrogate stepping stone benchmarks."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train", type=int, default=600)
    parser.add_argument(
        "--output-dir", type=str, default="figures/surrogate_benchmarks_rpl"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["linear", "basis", "nonlinear", "multi", "all"],
        default=["all"],
    )
    parser.add_argument("--n-episodes", type=int, default=30,
                        help="RPL training episodes per (surrogate, function) pair")
    parser.add_argument("--iterations-per-episode", type=int, default=50,
                        help="Warm-start gradient steps per episode")
    parser.add_argument("--rpl-threshold-fraction", type=float, default=0.85,
                        help="P_thresh = this * R^2_GT")
    parser.add_argument("--residual-hidden", type=int, default=32,
                        help="Hidden layer width of the residual MLP")
    parser.add_argument("--lambda-alpha", type=float, default=0.1,
                        help="Dual ascent step size")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"]
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
    surrogates = [LinearProxySurrogate(), MLPProxySurrogate(seed=args.seed)]

    evaluator = RPLEvaluator(
        residual_hidden=(args.residual_hidden,),
        n_episodes=args.n_episodes,
        iterations_per_episode=args.iterations_per_episode,
        threshold_fraction=args.rpl_threshold_fraction,
        lambda_alpha=args.lambda_alpha,
        seed=args.seed,
    )

    all_records: List[RPLRecord] = []

    if run["linear"]:
        logger.info("=== LinearApproximationBenchmarkRPL ===")
        recs = LinearApproximationBenchmarkRPL(seed=args.seed).run(
            proxy, surrogates, args.n_train, evaluator
        )
        all_records += recs
        plot_lambda_trajectories(recs, "LinearApproximationRPL", args.output_dir, args.n_episodes)
        plot_offloading_integral(recs, "LinearApproximationRPL", args.output_dir)

    if run["basis"]:
        logger.info("=== BasisFunctionBenchmarkRPL ===")
        recs = BasisFunctionBenchmarkRPL(seed=args.seed).run(
            proxy, surrogates, args.n_train, evaluator
        )
        all_records += recs
        plot_offloading_integral(recs, "BasisFunctionRPL", args.output_dir)

    if run["nonlinear"]:
        logger.info("=== NonlinearApproximationBenchmarkRPL ===")
        recs = NonlinearApproximationBenchmarkRPL(seed=args.seed).run(
            proxy, surrogates, args.n_train, evaluator
        )
        all_records += recs
        plot_lambda_trajectories(recs, "NonlinearApproximationRPL", args.output_dir, args.n_episodes)
        plot_offloading_integral(recs, "NonlinearApproximationRPL", args.output_dir)
        plot_residual_field_maps(
            proxy, surrogates, STANDARD_FUNCTIONS,
            args.n_train, args.seed, args.output_dir,
            "NonlinearApproximationRPL_standard_residuals",
            "Nonlinear benchmark (standard): residual field maps",
            evaluator,
        )
        plot_residual_field_maps(
            proxy, surrogates, ADVERSARIAL_FUNCTIONS,
            args.n_train, args.seed, args.output_dir,
            "NonlinearApproximationRPL_adversarial_residuals",
            "Nonlinear benchmark (adversarial): residual field maps",
            evaluator,
        )

    if run["multi"]:
        logger.info("=== MultiFunctionSuiteBenchmarkRPL ===")
        recs = MultiFunctionSuiteBenchmarkRPL(seed=args.seed).run(
            proxy, surrogates, args.n_train, evaluator
        )
        all_records += recs
        plot_lambda_trajectories(recs, "MultiFunctionSuiteRPL", args.output_dir, args.n_episodes)
        plot_offloading_integral(recs, "MultiFunctionSuiteRPL", args.output_dir)
        plot_residual_field_maps(
            proxy, surrogates, STANDARD_FUNCTIONS,
            args.n_train, args.seed, args.output_dir,
            "MultiFunctionSuiteRPL_residuals",
            "Multi-function suite: residual field maps",
            evaluator,
        )

    save_csv(all_records, args.output_dir)

    df = pd.DataFrame(
        [
            {
                "benchmark": r.benchmark,
                "surrogate": r.surrogate,
                "function": r.function,
                "R^2_base": f"{r.r2_base:.3f}",
                "R^2_final": f"{r.r2_final:.3f}",
                "integral": f"{r.offloading_integral:.2f}",
                "episodes_to_lambda>0": r.episodes_to_positive_lambda,
            }
            for r in all_records
        ]
    )
    logger.info("\n%s", df.to_string(index=False))


if __name__ == "__main__":
    main()
