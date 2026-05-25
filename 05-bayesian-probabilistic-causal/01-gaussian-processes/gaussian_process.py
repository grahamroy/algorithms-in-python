"""
gaussian_process.py --- companion code for "Gaussian Processes"
(Bayesian, Probabilistic & Causal Methods, Part 1).

Three demos:
  1. GP regression from scratch on a 1-D noisy regression problem
     with an RBF kernel.
  2. Comparison with scikit-learn's GaussianProcessRegressor.
  3. MLP baseline (no uncertainty) for contrast.

Dependencies: numpy, scikit-learn. Runs in well under a second.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neural_network import MLPRegressor


SEPARATOR = "=" * 72
RNG_SEED = 7


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


def true_f(x):
    return np.sin(x) * x / 2


def make_dataset(n_train=30, n_test=100, noise=0.20, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    X_train = rng.uniform(-5, 5, size=(n_train, 1))
    y_train = true_f(X_train.ravel()) + rng.normal(0, noise, size=n_train)
    X_test = np.linspace(-5.5, 5.5, n_test).reshape(-1, 1)
    y_test = true_f(X_test.ravel())
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# From-scratch GP with RBF kernel
# ---------------------------------------------------------------------------

class GPRegressor:
    """Gaussian process regression with an RBF kernel and
    fixed hyperparameters. Predictions return (mean, variance)."""

    def __init__(self, length_scale=1.0, sigma_f=1.0,
                 noise=0.20):
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.noise = noise

    def _rbf(self, A, B):
        # squared euclidean
        sq = (A ** 2).sum(axis=1)[:, None] + \
             (B ** 2).sum(axis=1)[None, :] - 2 * A @ B.T
        return (self.sigma_f ** 2) * np.exp(
            -sq / (2 * self.length_scale ** 2)
        )

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        K = self._rbf(self.X, self.X)
        self._K_inv = np.linalg.inv(K + (self.noise ** 2) *
                                    np.eye(len(self.X)))
        return self

    def predict(self, X_test, return_std=True):
        X_test = np.asarray(X_test, dtype=float)
        K_star = self._rbf(X_test, self.X)
        K_starstar = self._rbf(X_test, X_test)
        mu = K_star @ self._K_inv @ self.y
        cov = K_starstar - K_star @ self._K_inv @ K_star.T
        if return_std:
            return mu, np.sqrt(np.maximum(np.diag(cov), 0.0))
        return mu


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_from_scratch(X_train, y_train, X_test, y_test):
    banner("DEMO 1 --- GP regression from scratch on 30 noisy samples")

    gp = GPRegressor(length_scale=1.0, sigma_f=1.0, noise=0.20)
    gp.fit(X_train, y_train)
    mu, std = gp.predict(X_test, return_std=True)
    mse = float(((mu - y_test) ** 2).mean())
    width = float((2 * 1.96 * std).mean())
    coverage = float(((mu - 1.96 * std <= y_test) &
                      (y_test <= mu + 1.96 * std)).mean())
    print(f"  Kernel              : RBF (length-scale = 1.0, sigma_f = 1.0)")
    print(f"  Noise (sigma_n)     : 0.20")
    print(f"  Training set        : {len(X_train)} points")
    print(f"  Test set            : {len(X_test)} query points")
    print(f"  Test set MSE        : {mse:.4f}")
    print(f"  Average 95% interval width : {width:.2f}")
    print(f"  Coverage (true f within 95% CI) : {coverage:.2f}")
    return gp, mu


def demo_sklearn(X_train, y_train, X_test, y_test, our_mu):
    banner("DEMO 2 --- sklearn GaussianProcessRegressor")

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    sk = GaussianProcessRegressor(kernel=kernel,
                                  alpha=0.20 ** 2,
                                  n_restarts_optimizer=5,
                                  random_state=RNG_SEED).fit(
        X_train, y_train
    )
    mu = sk.predict(X_test)
    mse = float(((mu - y_test) ** 2).mean())
    fitted = sk.kernel_
    print(f"  Kernel              : {kernel}")
    print(f"  Optimised length-scale after fitting : "
          f"{sk.kernel_.k2.length_scale:.2f}")
    print(f"  Test set MSE        : {mse:.4f}")
    print(f"  Agreement with from-scratch (mean abs diff) : "
          f"{float(np.abs(mu - our_mu).mean()):.1e}")


def demo_mlp(X_train, y_train, X_test, y_test):
    banner("DEMO 3 --- MLP regressor for comparison (no uncertainty)")

    mlp = MLPRegressor(hidden_layer_sizes=(32, 32),
                       max_iter=5000, random_state=RNG_SEED).fit(
        X_train, y_train
    )
    mu = mlp.predict(X_test)
    mse = float(((mu - y_test) ** 2).mean())
    print(f"  Architecture        : 1 → 32 → 32 → 1, ReLU")
    print(f"  Test set MSE        : {mse:.4f}")
    print(f"  Uncertainty estimate : none")


def main() -> None:
    X_train, y_train, X_test, y_test = make_dataset()
    gp, mu = demo_from_scratch(X_train, y_train, X_test, y_test)
    demo_sklearn(X_train, y_train, X_test, y_test, mu)
    demo_mlp(X_train, y_train, X_test, y_test)
    print()


if __name__ == "__main__":
    main()
