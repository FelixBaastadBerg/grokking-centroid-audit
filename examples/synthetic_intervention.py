"""
Minimal, self-contained illustration of the centroid intervention (Equation 1 of the paper) on a
synthetic clustered cloud. Needs only numpy, scipy and scikit-learn; runs in a few seconds.

    x_i(alpha) = mu + alpha (mu_{y_i} - mu) + (x_i - mu_{y_i})

Every residual x_i - mu_{y_i} is left untouched, so all within-class distances are preserved;
only the class-centroid configuration is rescaled. Each cloud is then trace-normalised (centred
and divided by its root-mean-square distance to the grand mean).

Expected behaviour: PH-dimension (MST scaling) falls as the classes separate, while TwoNN stays
close to the true within-class dimension once each point's first two neighbours lie in its class.

Run from the repository root:  python examples/synthetic_intervention.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import geom  # noqa: E402

K, M, D_TRUE, AMBIENT = 23, 23, 5, 200      # classes, points per class, within-class dim, width


def make_cloud(rng):
    basis = np.linalg.qr(rng.normal(size=(AMBIENT, AMBIENT)))[0][:D_TRUE]
    centroids = 3.0 * rng.normal(size=(K, AMBIENT))
    X = np.concatenate([centroids[k] + rng.normal(size=(M, D_TRUE)) @ basis for k in range(K)])
    y = np.repeat(np.arange(K), M)
    return X, y


def intervene(X, y, alpha):
    mu = X.mean(0)
    mu_y = np.stack([X[y == c].mean(0) for c in range(K)])[y]
    return mu + alpha * (mu_y - mu) + (X - mu_y)


def trace_norm(Z):
    Zc = Z - Z.mean(0)
    return Zc / np.sqrt((Zc ** 2).sum(1).mean())


def main():
    rng = np.random.default_rng(0)
    X, y = make_cloud(rng)
    print(f"{K} classes x {M} points, true within-class dimension {D_TRUE}, ambient {AMBIENT}")
    print(f"{'alpha':>6}  {'PH-dim':>7}  {'TwoNN':>6}")
    for alpha in (0.0, 0.5, 1.0, 2.0, 8.0):
        Z = trace_norm(intervene(X, y, alpha))
        print(f"{alpha:>6g}  {geom.ph_dim(Z)[0]:>7.2f}  {geom.twonn(Z):>6.2f}")


if __name__ == "__main__":
    main()
