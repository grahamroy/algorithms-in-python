"""
autoencoder.py --- companion code for "Autoencoders"
(Advanced Unsupervised Learning, Part 3).

Three demos:
  1. Small autoencoder from scratch in numpy (manual backprop,
     ReLU activations, Adam optimiser) trained on the digits
     dataset.
  2. PCA at the same bottleneck size for comparison.
  3. KNN classification accuracy on the latent codes vs raw
     pixels.

Dependencies: numpy, scikit-learn. Runs in well under a minute.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


SEPARATOR = "=" * 72
RNG_SEED = 7


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Tiny autoencoder in numpy
# ---------------------------------------------------------------------------

def relu(x):
    return np.maximum(x, 0.0)


def relu_grad(x):
    return (x > 0).astype(x.dtype)


class Autoencoder:
    """Symmetric fully-connected autoencoder with ReLU hidden
    activations and a linear output layer. Optimised with Adam."""

    def __init__(self, layer_sizes, lr=1e-3, batch_size=64,
                 n_epochs=300, random_state=RNG_SEED):
        # layer_sizes example: [64, 32, 8, 32, 64]
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.random_state = random_state

    def _init_params(self):
        rng = np.random.default_rng(self.random_state)
        self.W = []
        self.b = []
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            # He initialisation for ReLU
            std = np.sqrt(2.0 / fan_in)
            self.W.append(rng.normal(0, std, size=(fan_in, fan_out)))
            self.b.append(np.zeros(fan_out))
        # Adam state
        self.m_W = [np.zeros_like(W) for W in self.W]
        self.v_W = [np.zeros_like(W) for W in self.W]
        self.m_b = [np.zeros_like(b) for b in self.b]
        self.v_b = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def _forward(self, X):
        """Returns (activations, pre_activations). Bottleneck is
        the last hidden activation before the symmetric expansion;
        the linear output is the reconstruction."""
        a = X
        activations = [a]
        zs = []
        n_layers = len(self.W)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            zs.append(z)
            if i < n_layers - 1:
                a = relu(z)
            else:
                a = z  # linear output
            activations.append(a)
        return activations, zs

    def _backward(self, activations, zs, X):
        """Backprop the squared-error loss across all layers."""
        n = X.shape[0]
        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)
        # dL/dz for the output (linear) layer: 2 * (x_hat - x) / n
        x_hat = activations[-1]
        delta = (2.0 / n) * (x_hat - X)
        for i in reversed(range(len(self.W))):
            a_prev = activations[i]
            grads_W[i] = a_prev.T @ delta
            grads_b[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * relu_grad(zs[i - 1])
        return grads_W, grads_b

    def _adam_step(self, grads_W, grads_b,
                   beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        bc1 = 1 - beta1 ** self.t
        bc2 = 1 - beta2 ** self.t
        for i in range(len(self.W)):
            self.m_W[i] = beta1 * self.m_W[i] + (1 - beta1) * grads_W[i]
            self.v_W[i] = beta2 * self.v_W[i] + (1 - beta2) * (grads_W[i] ** 2)
            mhat = self.m_W[i] / bc1
            vhat = self.v_W[i] / bc2
            self.W[i] -= self.lr * mhat / (np.sqrt(vhat) + eps)
            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * grads_b[i]
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * (grads_b[i] ** 2)
            mhat = self.m_b[i] / bc1
            vhat = self.v_b[i] / bc2
            self.b[i] -= self.lr * mhat / (np.sqrt(vhat) + eps)

    def fit(self, X, X_val=None, verbose=False):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        self._init_params()
        rng = np.random.default_rng(self.random_state + 1)
        for epoch in range(self.n_epochs):
            perm = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                batch = X[idx]
                acts, zs = self._forward(batch)
                gW, gb = self._backward(acts, zs, batch)
                self._adam_step(gW, gb)
            if verbose and (epoch + 1) % 50 == 0:
                train_mse = self.reconstruction_mse(X)
                msg = f"  epoch {epoch + 1:>3}: train MSE = {train_mse:.4f}"
                if X_val is not None:
                    msg += f", val MSE = {self.reconstruction_mse(X_val):.4f}"
                print(msg)
        return self

    def reconstruct(self, X):
        acts, _ = self._forward(np.asarray(X, dtype=float))
        return acts[-1]

    def encode(self, X):
        """Return the bottleneck activation."""
        acts, _ = self._forward(np.asarray(X, dtype=float))
        # Bottleneck is the middle of the activations list
        mid = len(acts) // 2
        return acts[mid]

    def reconstruction_mse(self, X):
        x_hat = self.reconstruct(X)
        return float(((X - x_hat) ** 2).mean())


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_autoencoder(X_train, X_test):
    banner("DEMO 1 --- Autoencoder from scratch on the digits dataset")

    arch = [64, 32, 8, 32, 64]
    print(f"  Architecture       : "
          f"{' → '.join(str(s) for s in arch)}  (ReLU + linear out)")
    print(f"  Optimiser          : Adam, lr=1e-3, batch_size=64")
    print(f"  Epochs             : 300")

    ae = Autoencoder(layer_sizes=arch, lr=1e-3,
                     batch_size=64, n_epochs=300,
                     random_state=RNG_SEED)
    ae.fit(X_train, X_val=X_test, verbose=False)
    train_mse = ae.reconstruction_mse(X_train)
    test_mse = ae.reconstruction_mse(X_test)
    print(f"  Final train loss   : {train_mse:.4f}")
    print(f"  Final test loss    : {test_mse:.4f}")
    return ae


def demo_pca(X_train, X_test):
    banner("DEMO 2 --- PCA at the same bottleneck size for comparison")

    pca = PCA(n_components=8).fit(X_train)
    train_back = pca.inverse_transform(pca.transform(X_train))
    test_back = pca.inverse_transform(pca.transform(X_test))
    train_mse = float(((X_train - train_back) ** 2).mean())
    test_mse = float(((X_test - test_back) ** 2).mean())
    print(f"  n_components       : 8")
    print(f"  Train MSE          : {train_mse:.4f}")
    print(f"  Test MSE           : {test_mse:.4f}")
    return pca


def demo_knn(ae, pca, X_train, X_test, y_train, y_test):
    banner("DEMO 3 --- KNN-in-latent-space classification accuracy")

    # Autoencoder latent
    z_train = ae.encode(X_train)
    z_test = ae.encode(X_test)
    knn = KNeighborsClassifier(n_neighbors=5).fit(z_train, y_train)
    ae_acc = (knn.predict(z_test) == y_test).mean()

    # PCA latent
    z_train_p = pca.transform(X_train)
    z_test_p = pca.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=5).fit(z_train_p, y_train)
    pca_acc = (knn.predict(z_test_p) == y_test).mean()

    # Raw pixels
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    raw_acc = (knn.predict(X_test) == y_test).mean()

    print(f"  Autoencoder latent (8d)  KNN accuracy : {ae_acc:.3f}")
    print(f"  PCA latent (8d)          KNN accuracy : {pca_acc:.3f}")
    print(f"  Raw 64-d pixels          KNN accuracy : {raw_acc:.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    digits = load_digits()
    X = digits.data
    y = digits.target
    # Standardise so reconstruction MSE is in comparable units
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RNG_SEED, stratify=y
    )
    ae = demo_autoencoder(X_train, X_test)
    pca = demo_pca(X_train, X_test)
    demo_knn(ae, pca, X_train, X_test, y_train, y_test)
    print()


if __name__ == "__main__":
    main()
