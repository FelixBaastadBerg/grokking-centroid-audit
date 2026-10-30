import torch
import copy
import pandas as pd
import pdb
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics import pairwise_distances
import io
import lzma
import zlib
import contextlib


from typing import Iterable

import torch
import numpy as np
from typing import Iterable, List, Tuple

def _flatten_like_params(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors if t is not None], dim=0)

def _unflatten_to_params(vec: torch.Tensor, params: List[torch.nn.Parameter]) -> List[torch.Tensor]:
    out = []
    i = 0
    for p in params:
        n = p.numel()
        out.append(vec[i:i+n].view_as(p))
        i += n
    return out

def hvp_from_grads(
    grads: List[torch.Tensor],
    params: List[torch.nn.Parameter],
    v_flat: torch.Tensor,
    retain_graph: bool = True,
) -> torch.Tensor:
    """
    Compute H v where H is Hessian of loss wrt params.
    Assumes `grads` are grad(loss, params) with create_graph=True.
    """
    v_parts = _unflatten_to_params(v_flat, params)
    gv = sum((g * v).sum() for g, v in zip(grads, v_parts))
    Hv_parts = torch.autograd.grad(gv, params, retain_graph=retain_graph, allow_unused=False)
    return _flatten_like_params(Hv_parts)

def hessian_trace_hutchinson_from_grads(
    grads: List[torch.Tensor],
    params: List[torch.nn.Parameter],
    num_samples: int = 16,
    distribution: str = "rademacher",
    seed: int = 0,
) -> float:
    device = grads[0].device
    g_flat = _flatten_like_params(grads)
    if g_flat.numel() == 0:
        return float("nan")

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    vals = []
    for _ in range(num_samples):
        if distribution == "rademacher":
            v = torch.empty_like(g_flat).bernoulli_(0.5, generator=gen).mul_(2).sub_(1)
        elif distribution == "gaussian":
            v = torch.randn_like(g_flat, generator=gen)
        else:
            raise ValueError("distribution must be 'rademacher' or 'gaussian'")

        Hv = hvp_from_grads(grads, params, v, retain_graph=True)
        vals.append(torch.dot(v, Hv))

    return float(torch.stack(vals).mean().detach().cpu().item())

def hessian_lambda_max_power_iteration_from_grads(
    grads: List[torch.Tensor],
    params: List[torch.nn.Parameter],
    num_iters: int = 30,
    tol: float = 1e-4,
    seed: int = 0,
) -> float:
    """
    Estimates the largest algebraic eigenvalue lambda_max(H).
    """
    device = grads[0].device
    n = sum(p.numel() for p in params)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    v = torch.randn(n, device=device, generator=gen)
    v = v / (v.norm() + 1e-12)

    prev = None
    for _ in range(num_iters):
        Hv = hvp_from_grads(grads, params, v, retain_graph=True)
        # Rayleigh quotient
        lam = torch.dot(v, Hv)
        # normalize
        v_next = Hv / (Hv.norm() + 1e-12)

        if prev is not None and torch.isfinite(lam) and abs(lam - prev) <= tol * (abs(prev) + 1e-12):
            v = v_next
            prev = lam
            break

        v = v_next
        prev = lam

    return float(prev.detach().cpu().item()) if prev is not None else float("nan")

def hessian_spectral_norm_power_iteration_from_grads(
    grads: List[torch.Tensor],
    params: List[torch.nn.Parameter],
    num_iters: int = 30,
    tol: float = 1e-4,
    seed: int = 0,
) -> float:
    """
    Estimates ||H||_2 = max |lambda_i| (spectral norm) for symmetric H.
    Uses power iteration on H^2 implicitly by iterating with H and tracking ||Hv||.
    """
    device = grads[0].device
    n = sum(p.numel() for p in params)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    v = torch.randn(n, device=device, generator=gen)
    v = v / (v.norm() + 1e-12)

    prev = None
    for _ in range(num_iters):
        Hv = hvp_from_grads(grads, params, v, retain_graph=True)
        sigma = Hv.norm()  # approximates largest singular value over iterations
        v_next = Hv / (sigma + 1e-12)

        if prev is not None and torch.isfinite(sigma) and abs(sigma - prev) <= tol * (abs(prev) + 1e-12):
            v = v_next
            prev = sigma
            break

        v = v_next
        prev = sigma

    return float(prev.detach().cpu().item()) if prev is not None else float("nan")

def hessian_frobenius_norm_hutchinson_from_grads(
    grads: List[torch.Tensor],
    params: List[torch.nn.Parameter],
    num_samples: int = 16,
    distribution: str = "rademacher",
    seed: int = 0,
) -> float:
    """
    Estimates ||H||_F via:
      tr(H^2) = E[ ||H v||^2 ]  for v with iid Rademacher/Gaussian entries.
    Then ||H||_F ≈ sqrt( mean ||H v||^2 ).
    """
    device = grads[0].device
    g_flat = _flatten_like_params(grads)
    if g_flat.numel() == 0:
        return float("nan")

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    vals = []
    for _ in range(num_samples):
        if distribution == "rademacher":
            v = torch.empty_like(g_flat).bernoulli_(0.5, generator=gen).mul_(2).sub_(1)
        elif distribution == "gaussian":
            v = torch.randn_like(g_flat, generator=gen)
        else:
            raise ValueError("distribution must be 'rademacher' or 'gaussian'")

        Hv = hvp_from_grads(grads, params, v, retain_graph=True)
        vals.append((Hv * Hv).sum())  # ||Hv||^2

    trH2 = torch.stack(vals).mean()
    frob = torch.sqrt(torch.clamp(trH2, min=0.0))
    return float(frob.detach().cpu().item())


def _flatten_grads(grads: Iterable[torch.Tensor]) -> torch.Tensor:
    """Flatten a list/iterable of gradient tensors into one 1D vector."""
    return torch.cat([g.reshape(-1) for g in grads if g is not None], dim=0)


def hessian_trace_hutchinson(
    loss: torch.Tensor,
    params: list[torch.nn.Parameter],
    num_samples: int = 16,
    distribution: str = "rademacher",  # "rademacher" or "gaussian"
    seed: int = 0,
) -> float:
    """
    Approximate tr(H) where H is Hessian of `loss` w.r.t. `params` using Hutchinson:
        tr(H) = E[v^T H v]
    estimated by averaging K samples.

    Uses autograd to compute HVPs without forming H.
    """
    device = loss.device
    # First derivative (create_graph=True so we can take derivatives of grads)
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True, allow_unused=False)

    # Flatten grads once
    g_flat = _flatten_grads(grads)
    if g_flat.numel() == 0:
        return float("nan")

    # RNG
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    estimates = []
    for _ in range(num_samples):
        if distribution == "rademacher":
            v = torch.empty_like(g_flat).bernoulli_(0.5, generator=gen).mul_(2).sub_(1)  # ±1
        elif distribution == "gaussian":
            v = torch.randn_like(g_flat, generator=gen)
        else:
            raise ValueError("distribution must be 'rademacher' or 'gaussian'")

        # Compute v^T g
        vg = torch.dot(g_flat, v)

        # Compute grad(v^T g) = H v (flattened)
        Hv_parts = torch.autograd.grad(vg, params, retain_graph=True, allow_unused=False)
        Hv = _flatten_grads(Hv_parts)

        # Hutchinson sample: v^T H v
        estimates.append(torch.dot(v, Hv))

    trH = torch.stack(estimates).mean()
    return float(trH.detach().cpu().item())


def magnitude_from_point_cloud(
    X: np.ndarray,
    metric: str = "euclidean",
    scale: float = 1.0,
    ridge: float = 1e-8,
    max_points: int | None = 512,
    seed: int = 0,
    norm_mode: str = "row_l2",   # ✅ normalization integrated
) -> tuple[float, float]:
    """
    Returns (Mag, PMag) using similarity matrix M_ij = exp(-scale * rho(x_i, x_j)),
    where rho is the chosen `metric` on the *normalized* point cloud.

    Mag = sum(beta)
    PMag = sum(max(beta, 0))
    beta solves: (M + ridge*I) beta = 1
    """
    if scale < 0:
        raise ValueError("scale must be >= 0")
    if ridge < 0:
        raise ValueError("ridge must be >= 0")

    X = np.asarray(X, dtype=np.float64)

    # ensure (n, d)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    # drop non-finite rows
    X = X[np.isfinite(X).all(axis=1)]
    n = X.shape[0]
    if n < 2:
        return float("nan"), float("nan")

    # subsample (magnitude is O(n^3))
    if max_points is not None and n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        X = X[idx]
        n = X.shape[0]

    # ✅ normalize BEFORE distances
    X = normalize_point_cloud(X, mode=norm_mode)

    D = pairwise_distances(X, metric=metric)
    D = np.asarray(D, dtype=np.float64)
    D[~np.isfinite(D)] = 0.0

    M = np.exp(-scale * D)

    A = M + ridge * np.eye(n, dtype=np.float64)
    b = np.ones(n, dtype=np.float64)

    try:
        beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        beta, *_ = np.linalg.lstsq(A, b, rcond=None)

    mag = float(beta.sum())
    pmag = float(np.maximum(beta, 0.0).sum())
    return mag, pmag

def e_alpha_from_mst_edges(edge_lengths: np.ndarray, alpha: float) -> float:
    e = np.asarray(edge_lengths, dtype=np.float64)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return float("nan")
    return float(np.sum(e ** alpha))


def normalize_point_cloud(
    X: np.ndarray,
    mode: str = "row_l2",   # "row_l2" | "zscore" | "none"
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Normalize point cloud X (n_points, d).

    row_l2: each point x_i scaled to unit L2 norm -> scale-invariant distances
    zscore: each feature dimension standardized -> also removes feature scaling/shift
    none:   no normalization
    """
    X = np.asarray(X, dtype=np.float64)

    if mode == "none":
        return X

    if mode == "row_l2":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / (norms + eps)

    if mode == "zscore":
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True)
        return (X - mu) / (sd + eps)

    raise ValueError(f"Unknown normalization mode: {mode}")


def _state_dict_to_bytes(state_dict, quantize_bits: int | None = None) -> bytes:
    """
    Serialize state_dict tensors into bytes.
    Optionally quantize each tensor to int8/int16/etc to make compression length more stable.
    """
    buf = io.BytesIO()

    # Deterministic order
    keys = sorted(state_dict.keys())

    for k in keys:
        t = state_dict[k].detach().to("cpu")
        arr = t.numpy()

        # Write key and shape metadata (small but helps determinism)
        buf.write(k.encode("utf-8"))
        buf.write(b"\0")
        buf.write(str(arr.shape).encode("utf-8"))
        buf.write(b"\0")
        buf.write(str(arr.dtype).encode("utf-8"))
        buf.write(b"\0")

        if quantize_bits is None:
            # raw bytes
            buf.write(arr.tobytes(order="C"))
        else:
            # symmetric per-tensor quantization
            # quantize_bits typical: 8 or 16
            max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
            if max_abs == 0.0:
                scale = 1.0
            else:
                # map to signed integer range
                qmax = (2 ** (quantize_bits - 1)) - 1
                scale = max_abs / qmax

            # store scale so dequantization is defined
            buf.write(np.float32(scale).tobytes())

            if arr.size:
                q = np.round(arr / scale).astype(np.int64)
                qmax = (2 ** (quantize_bits - 1)) - 1
                q = np.clip(q, -qmax - 1, qmax)

                if quantize_bits <= 8:
                    q = q.astype(np.int8)
                elif quantize_bits <= 16:
                    q = q.astype(np.int16)
                else:
                    q = q.astype(np.int32)

                buf.write(q.tobytes(order="C"))

    return buf.getvalue()


def compression_length_bits(model, compressor: str = "lzma", quantize_bits: int | None = None) -> float:
    """
    Returns compressed length in bits of model.state_dict() bytes.

    compressor: "lzma" (stronger, slower) or "zlib" (faster)
    quantize_bits: None for raw float bytes, or e.g. 8 for int8 quantized compression
    """
    state_dict = model.state_dict()
    raw = _state_dict_to_bytes(state_dict, quantize_bits=quantize_bits)

    if compressor == "lzma":
        comp = lzma.compress(raw, preset=6)
    elif compressor == "zlib":
        comp = zlib.compress(raw, level=9)
    else:
        raise ValueError(f"Unknown compressor: {compressor}")

    return float(len(comp) * 8)  # bits



def ph0_summaries_from_mst_edges(edge_lengths: np.ndarray) -> dict:
    """
    PH0 for a point cloud (with Euclidean metric) has death times equal to MST edge lengths.
    This function computes useful summaries of those death times.

    Returns dict with:
      - ph0_total_persistence: sum of edge lengths
      - ph0_persistence_entropy: entropy of normalized edge lengths
      - ph0_num_long_edges_p90: number of edges above 90th percentile
      - ph0_max_edge: maximum edge length
    """
    e = np.asarray(edge_lengths, dtype=np.float64)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {
            "ph0_total_persistence": float("nan"),
            "ph0_persistence_entropy": float("nan"),
            "ph0_num_long_edges_p90": float("nan"),
            "ph0_max_edge": float("nan"),
        }

    total = float(e.sum())
    max_edge = float(e.max())

    # persistence entropy (normalized edge lengths)
    if total <= 0:
        entropy = float("nan")
    else:
        p = e / total
        p = np.clip(p, 1e-12, 1.0)
        entropy = float(-(p * np.log(p)).sum())

    p90 = np.quantile(e, 0.90)
    num_long = float((e > p90).sum())

    return {
        "ph0_total_persistence": total,
        "ph0_persistence_entropy": entropy,
        "ph0_num_long_edges_p90": num_long,
        "ph0_max_edge": max_edge,
    }

def mst_edges(
    X: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Compute MST edge lengths for a point cloud X.
    Returns a 1D array of MST edge lengths.
    """
    D = pairwise_distances(X, metric=metric)
    T = minimum_spanning_tree(D)  # sparse
    # T.data are the nonzero entries = MST edge lengths
    return np.asarray(T.data, dtype=np.float64)


def correlation_dimension(
    X: np.ndarray,
    max_points: int = 512,
    n_radii: int = 24,
    q_low: float = 0.02,
    q_high: float = 0.50,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Estimate correlation dimension (CorrDim) via:
        C(r) = P(||x_i - x_j|| < r)  ~ r^d  ->  d = slope of log C(r) vs log r

    Returns: (corr_dim, corr_r2)
    """

    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    X = X[np.isfinite(X).all(axis=1)]
    N = X.shape[0]
    if N < 64:
        return float("nan"), float("nan")

    if N > max_points:
        idx = rng.choice(N, size=max_points, replace=False)
        X = X[idx]
        N = X.shape[0]

    # Pairwise distances (dense). We'll use upper triangle.
    D = pairwise_distances(X, metric="euclidean")
    iu = np.triu_indices(N, k=1)
    d = D[iu]
    d = d[np.isfinite(d)]
    if d.size < 1000:
        return float("nan"), float("nan")

    # Choose a scale range based on distance quantiles (avoids setting absolute radii)
    lo = np.quantile(d, q_low)
    hi = np.quantile(d, q_high)
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo or lo <= 0:
        return float("nan"), float("nan")

    radii = np.exp(np.linspace(np.log(lo), np.log(hi), n_radii))

    # Correlation integral C(r)
    # C(r) = fraction of pairs below r
    d_sorted = np.sort(d)
    counts = np.searchsorted(d_sorted, radii, side="right")
    C = counts / d_sorted.size
    C = np.clip(C, 1e-12, 1.0)

    # Fit only where not saturated (avoid C ~ 0 or ~1)
    mask = (C > 1e-6) & (C < 0.98)
    if mask.sum() < 6:
        return float("nan"), float("nan")

    x = np.log(radii[mask])
    y = np.log(C[mask])

    slope, intercept = np.polyfit(x, y, deg=1)

    # R^2
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float("nan") if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)

    return float(slope), r2


def mst_intrinsic_dimension(
    X: np.ndarray,
    n_list=(64, 128, 256, 512),
    n_repeats: int = 3,
    metric: str = "euclidean",
    seed: int = 0,
) -> float:
    """
    Estimate intrinsic/fractal dimension via MST length scaling:
        L(n) ~ n^{(d-1)/d}  =>  d = 1 / (1 - slope)
    where slope is from log L(n) vs log n.
    """
    rng = np.random.default_rng(seed)

    # Clean
    X = np.asarray(X)
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    N = X.shape[0]
    if N < min(n_list):
        return float("nan")

    logs_n = []
    logs_L = []

    for n in n_list:
        if n > N:
            continue
        for _ in range(n_repeats):
            idx = rng.choice(N, size=n, replace=False)
            Y = X[idx]

            # Pairwise distances (dense n×n) then MST
            D = pairwise_distances(Y, metric=metric)
            T = minimum_spanning_tree(D)  # sparse matrix
            L = float(T.sum())            # total MST length

            if L > 0 and np.isfinite(L):
                logs_n.append(np.log(n))
                logs_L.append(np.log(L))

    if len(logs_n) < 2:
        return float("nan")

    # Fit slope alpha in log-log
    alpha, intercept = np.polyfit(np.array(logs_n), np.array(logs_L), deg=1)

    # Convert slope to dimension
    if alpha >= 1.0:
        return float("inf")
    if alpha <= 0.0:
        return float("nan")

    d = 1.0 / (1.0 - alpha)
    return float(d)


def mst_dimension_with_fit(
    X: np.ndarray,
    n_list=(64, 128, 256, 512),
    n_repeats: int = 3,
    metric: str = "euclidean",
    seed: int = 0,
) -> tuple[float, float, float]:
    """
    Returns (mst_dim, mst_alpha, mst_r2)
    where alpha is the log-log slope: log L(n) ~ a + alpha log n.
    mst_dim = 1 / (1 - alpha)
    """
    rng = np.random.default_rng(seed)

    X = np.asarray(X)
    X = X[np.isfinite(X).all(axis=1)]
    N = X.shape[0]
    if N < min(n_list):
        return float("nan"), float("nan"), float("nan")

    logs_n = []
    logs_L = []

    for n in n_list:
        if n > N:
            continue
        for _ in range(n_repeats):
            idx = rng.choice(N, size=n, replace=False)
            Y = X[idx]

            D = pairwise_distances(Y, metric=metric)
            T = minimum_spanning_tree(D)
            L = float(T.sum())
            if L > 0 and np.isfinite(L):
                logs_n.append(np.log(n))
                logs_L.append(np.log(L))

    if len(logs_n) < 6:
        return float("nan"), float("nan"), float("nan")

    x = np.array(logs_n)
    y = np.array(logs_L)

    alpha, intercept = np.polyfit(x, y, deg=1)

    # R^2
    yhat = alpha * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float("nan") if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)

    if alpha >= 1.0:
        return float("inf"), float(alpha), r2
    if alpha <= 0.0:
        return float("nan"), float(alpha), r2

    mst_dim = float(1.0 / (1.0 - alpha))
    return mst_dim, float(alpha), r2



def weight_spectral_entropy(W: np.ndarray, eps: float = 1e-12) -> float:
    """
    Spectral entropy from singular values of W.
    Returns H(p) where p_i = s_i / sum_j s_j.
    """
    # Compute singular values
    s = np.linalg.svd(W, full_matrices=False, compute_uv=False)

    s = np.abs(s)
    Z = s.sum()
    if Z <= eps:
        return 0.0

    p = s / Z
    p = np.clip(p, eps, 1.0)
    H = -(p * np.log(p)).sum()
    return float(H)

def weight_effective_rank(W: np.ndarray, eps: float = 1e-12) -> float:
    """
    Effective rank proxy via participation ratio of singular values squared.
    """
    s = np.linalg.svd(W, full_matrices=False, compute_uv=False)
    lam = (np.abs(s) ** 2)
    denom = (lam**2).sum()
    if denom <= eps:
        return 0.0
    return float((lam.sum() ** 2) / denom)


def twonn_intrinsic_dimension(X: np.ndarray) -> float:
    """
    TwoNN intrinsic dimension estimator.
    X: (n, d) numpy array.
    Returns: estimated intrinsic dimension (float), or nan if fails.
    """
    # Need enough points
    if X.shape[0] < 20:
        return float("nan")

    # NearestNeighbors: ask for 3 because the closest neighbor is the point itself
    nn = NearestNeighbors(n_neighbors=3, algorithm="auto", metric="euclidean")
    nn.fit(X)
    dists, _ = nn.kneighbors(X)  # shape (n, 3): [self, 1NN, 2NN]

    r1 = dists[:, 1]
    r2 = dists[:, 2]

    # Remove degenerate points (duplicates / near-duplicates)
    mask = (r1 > 1e-12) & np.isfinite(r1) & np.isfinite(r2) & (r2 > r1)
    if mask.sum() < 20:
        return float("nan")

    mu = r2[mask] / r1[mask]
    # TwoNN: d_hat = 1 / mean(log(mu))
    logs = np.log(mu)
    denom = logs.mean()
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return float(1.0 / denom)


def participation_ratio(X: np.ndarray) -> float:
    """
    Robust PCA participation ratio.
    Uses covariance eigvals; falls back to randomized SVD if needed.
    """
    # Center
    X = X - X.mean(axis=0, keepdims=True)

    # Compute covariance in float64 for stability
    X64 = X.astype(np.float64, copy=False)

    # If d is large, covariance is dxd and can be expensive.
    # But your d is small-ish; this is usually fine.
    C = (X64.T @ X64) / max(1, X64.shape[0] - 1)

    # Symmetrize to avoid tiny numerical asymmetry
    C = 0.5 * (C + C.T)

    try:
        eigvals = np.linalg.eigvalsh(C)
    except np.linalg.LinAlgError:
        return float("nan")

    eigvals = np.clip(eigvals, 0.0, None)
    s1 = eigvals.sum()
    s2 = (eigvals**2).sum()
    if s2 <= 0:
        return 0.0
    return float((s1 * s1) / s2)



def resolve_probe_layers(model) -> tuple[list[str], str]:
    """
    Fix T6: the probe set used to be hardcoded to ["layers.0","layers.1","layers.2"],
    which raises ValueError on any model that is not the 2-hidden-layer MLP -- i.e.
    a Transformer run died immediately.

    Returns (probe_layer_names, pre_logits_source_module_name), discovered from the
    model's own module names so the metric module stays architecture-agnostic.
    """
    names = [n for n, _ in model.named_modules() if n]

    mlp_layers = sorted(n for n in names if n.startswith("layers.") and n.count(".") == 1)
    if mlp_layers:
        return mlp_layers, mlp_layers[-1]

    # transformer: residual stream after each block, plus per-block MLP / attention
    probes: list[str] = []
    blocks = sorted(n for n in names if n.startswith("transformer.layers.") and n.count(".") == 2)
    for b in blocks:
        i = b.rsplit(".", 1)[1]
        probes.append(b)                                   # resid_post.<i>
        for sub, _tag in ((f"{b}.linear2", "mlp_hidden"), (f"{b}.self_attn", "attn_out")):
            if sub in names:
                probes.append(sub)
    if "linear" in names:
        return probes, "linear"
    return probes, (probes[-1] if probes else "")


# stable, systematic display names for transformer probes
def probe_display_name(raw: str) -> str:
    if raw.startswith("transformer.layers."):
        parts = raw.split(".")
        i = parts[2]
        if len(parts) == 3:
            return f"resid_post.{i}"
        if parts[3] == "linear2":
            return f"mlp_hidden.{i}"
        if parts[3] == "self_attn":
            return f"attn_out.{i}"
    return raw


class MetricsLogger:
    def __init__(self, num_epochs: int, log_frequency: int, seed: int = 0):

        # Added metrics
        self.dims_df = pd.DataFrame(
            columns=["epoch", "layer", "metric_name", "value"]
        )

        # Fix T5: estimator subsampling used the GLOBAL np.random, so estimator
        # output was not reproducible from --seed. Everything now draws from this.
        self.rng = np.random.default_rng(seed)
        self.probe_layers: list[str] | None = None
        self.pre_logits_source: str = ""


        self.num_epochs = num_epochs
        self.log_frequency = log_frequency
        self.num_early_training_epochs = 0
        self.early_training_log_frequency = 10

        if self.num_early_training_epochs == 0:
            self.num_loged_epochs = (num_epochs) // log_frequency
        else:
            self.num_loged_epochs = (num_epochs) // log_frequency 

        # Metrics to log
        self.metric_fns = {
            "loss": self.compute_loss,
            "accuracy": self.compute_accuracy,
            "weights_l2": self.compute_weights_l2,
            "zero_terms": self.compute_zero_terms,
            "softmax_collapse": self.compute_softmax_collapse,
        }

        self.metrics_df = pd.DataFrame(columns=["epoch", "input_type", "metric_name", "layer", "value"])

        self._train_output = None
        self._train_targets = None
        self._test_output = None
        self._test_targets = None

    def _make_global_activation(self, acts: dict, layer_names: list[str]) -> torch.Tensor:
        """
        Concatenate per-layer activations into one big representation per sample.
        Output shape: (n, D_total)
        """
        chunks = []

        for lname in layer_names:
            A = acts[lname]

            # (n, t, d) -> (n, d)
            if A.ndim == 3:
                A = A[:, -1, :]

            # flatten all but batch
            if A.ndim > 2:
                A = A.flatten(start_dim=1)

            chunks.append(A)

        # concat along feature dimension
        Z = torch.cat(chunks, dim=1)
        return Z


    # Added metrics
    def _get_layer_activations(self, model, x, layer_names):
        """
        Run model(x) once and capture outputs of specified modules.
        Returns dict layer_name -> activation tensor (n, d or n, t, d)
        """
        acts = {}
        hooks = []

        def make_hook(name):
            def hook(module, inp, out):
                # Some modules (e.g. nn.MultiheadAttention) return a tuple
                # (output, weights); the activation we want is the first element.
                if isinstance(out, (tuple, list)):
                    out = out[0]
                acts[name] = out

                # Fix T4: pre_logits was referenced by the analysis code but never
                # actually logged, because this was commented out. The input to the
                # final classifier layer IS the pre-logit representation.
                if name == self.pre_logits_source:
                    acts["pre_logits"] = inp[0]
            return hook


        # Register hooks
        name_to_module = dict(model.named_modules())
        for name in layer_names:
            if name not in name_to_module:
                raise ValueError(f"Layer '{name}' not found. Available: {list(name_to_module.keys())[:50]} ...")
            hooks.append(name_to_module[name].register_forward_hook(make_hook(name)))

        # Forward pass
        _ = model(x)

        # Remove hooks
        for h in hooks:
            h.remove()

        return acts
    
    def log_hessian_spectrum_metrics(
        self,
        epoch: int,
        model: torch.nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_function,
        args,
        trace_samples: int = 16,
        frob_samples: int = 16,
        power_iters: int = 30,
        seed: int = 0,
        which_norm: str = "spectral",  # "spectral" or "frobenius"
    ):
        model.eval()
        device = next(model.parameters()).device
        x = x.to(device)
        y = y.to(device)

        # ---- HARD GUARD: Hessian requires gradients enabled ----
        assert torch.is_grad_enabled(), (
            "Grad is disabled (torch.no_grad is active) during Hessian logging."
        )

        # Transformers use scaled_dot_product_attention, whose fused CPU kernels
        # have no second derivative ("_scaled_dot_product_flash_attention_for_cpu_
        # backward is not implemented"). Force the math backend so HVPs work.
        # NB: sdpa_kernel() returns a SINGLE-USE context manager, so build a fresh
        # one at every use site rather than reusing one object.
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            def sdpa_ctx():
                return sdpa_kernel(SDPBackend.MATH)
        except Exception:
            def sdpa_ctx():
                return contextlib.nullcontext()

        # ---- Force FULL PRECISION for Hessian computation (prevents underflow to 0) ----
        autocast_ctx = (
            torch.cuda.amp.autocast(enabled=False) if x.is_cuda else contextlib.nullcontext()
        )

        with autocast_ctx, sdpa_ctx():
            output = model(x)
            if getattr(args, "use_transformer", False):
                output = output[:, -1]

            # force float32 for stable curvature estimates
            output = output.float()

            # ---- SCALE FIX + SCALAR FIX ----
            # We want a scalar loss whose magnitude does NOT shrink with batch size,
            # so we use sum of per-example losses.
            loss_vec = None
            try:
                # preferred: your wrapper supports dtype= and reduction=
                loss_vec = loss_function(output, y, dtype=torch.float32, reduction="none")
            except TypeError:
                # fallback 1: wrapper supports reduction but not dtype
                try:
                    loss_vec = loss_function(output, y, reduction="none")
                except TypeError:
                    # fallback 2: wrapper returns scalar already (or doesn't accept reduction)
                    loss_vec = loss_function(output, y)

            if isinstance(loss_vec, torch.Tensor) and loss_vec.ndim > 0:
                loss = loss_vec.sum()
            else:
                # already scalar
                loss = loss_vec

            # guarantee float32 scalar
            loss = loss.float()

        # ---- grads / HVP graph ----
        params = [p for p in model.parameters() if p.requires_grad]
        try:
            with sdpa_ctx():
                grads = torch.autograd.grad(
                    loss, params, create_graph=True, retain_graph=True, allow_unused=False
                )
        except RuntimeError as exc:
            # Some architectures have no second derivative available on this device.
            # Record NaN so the schema stays intact instead of killing the run.
            print(f"[logger] Hessian metrics unavailable at epoch {epoch}: {exc}")
            for nm in (f"hess_trace_hutch_{trace_samples}",
                       f"hess_lambda_max_pi_{power_iters}",
                       f"hess_norm2_pi_{power_iters}" if which_norm == "spectral"
                       else f"hess_fro_hutch_{frob_samples}"):
                self.dims_df.loc[len(self.dims_df)] = {
                    "epoch": epoch, "layer": "all_params",
                    "metric_name": nm, "value": float("nan")}
            return

        # ---- DEBUG: check gradient is not vanishing / broken ----
        g_flat = _flatten_like_params(grads)
        g_norm = float(g_flat.norm().detach().cpu())
        g_max = float(g_flat.abs().max().detach().cpu())
        if epoch % 1000 == 0:
            lv = float(loss.detach().cpu())
            shape_str = (
                str(tuple(loss_vec.shape)) if isinstance(loss_vec, torch.Tensor) else "scalar"
            )
            print(
                f"HESS DEBUG2 epoch={epoch} "
                f"loss={lv:.6g} loss_vec_shape={shape_str} "
                f"grad_norm={g_norm:.6g} grad_max={g_max:.6g} "
                f"num_params={g_flat.numel()} device={device} grad_enabled={torch.is_grad_enabled()}"
            )

        # ---- trace(H) ----  (all HVPs under the math attention kernel)
        with sdpa_ctx():
            trH = hessian_trace_hutchinson_from_grads(
                grads, params, num_samples=trace_samples, seed=seed
            )

            # ---- lambda_max(H) ----
            lmax = hessian_lambda_max_power_iteration_from_grads(
                grads, params, num_iters=power_iters, seed=seed + 1
            )

        # ---- norm(H) ----
        if which_norm == "spectral":
            with sdpa_ctx():
                hnorm = hessian_spectral_norm_power_iteration_from_grads(
                    grads, params, num_iters=power_iters, seed=seed + 2
                )
            norm_name = f"hess_norm2_pi_{power_iters}"
        elif which_norm == "frobenius":
            with sdpa_ctx():
                hnorm = hessian_frobenius_norm_hutchinson_from_grads(
                    grads, params, num_samples=frob_samples, seed=seed + 2
                )
            norm_name = f"hess_fro_hutch_{frob_samples}"
        else:
            raise ValueError("which_norm must be 'spectral' or 'frobenius'")

        # ---- Log to dims_df ----
        self.dims_df.loc[len(self.dims_df)] = {
            "epoch": epoch,
            "layer": "all_params",
            "metric_name": f"hess_trace_hutch_{trace_samples}",
            "value": float(trH),
        }
        self.dims_df.loc[len(self.dims_df)] = {
            "epoch": epoch,
            "layer": "all_params",
            "metric_name": f"hess_lambda_max_pi_{power_iters}",
            "value": float(lmax),
        }
        self.dims_df.loc[len(self.dims_df)] = {
            "epoch": epoch,
            "layer": "all_params",
            "metric_name": norm_name,
            "value": float(hnorm),
        }
    
    def log_magnitude_metrics(
        self,
        epoch: int,
        activations: torch.Tensor,
        layer_name: str,
        max_points: int = 256,        # start smaller: O(n^3) solve
        norm_mode: str = "row_l2",
        metric: str = "euclidean",
        scale: float = 1.0,
        ridge: float = 1e-8,
        seed: int = 0,
    ):
        X = activations.detach().float().cpu().numpy()

        # Flatten like your other metrics
        if X.ndim == 3:
            X = X[:, -1, :]
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        # Clean
        X = X[np.isfinite(X).all(axis=1)]
        if X.shape[0] < 2:
            mag, pmag = float("nan"), float("nan")
        else:
            mag, pmag = magnitude_from_point_cloud(
                X,
                metric=metric,
                scale=scale,
                ridge=ridge,
                max_points=max_points,
                seed=seed,
                norm_mode=norm_mode,
            )

        self.dims_df.loc[len(self.dims_df)] = {
            "epoch": epoch,
            "layer": layer_name,
            "metric_name": f"mag_{metric}_s{scale:g}_{norm_mode}",
            "value": float(mag),
        }
        self.dims_df.loc[len(self.dims_df)] = {
            "epoch": epoch,
            "layer": layer_name,
            "metric_name": f"pmag_{metric}_s{scale:g}_{norm_mode}",
            "value": float(pmag),
        }
    
    def log_compression_length(self, epoch, model):
        # Raw float weights compressed
        bits_lzma = compression_length_bits(model, compressor="lzma", quantize_bits=None)
        self.metrics_df.loc[len(self.metrics_df)] = {
            "epoch": epoch,
            "input_type": "general",
            "metric_name": "compression_lzma_bits",
            "layer": np.nan,
            "value": bits_lzma,
        }

        # Quantized (int8) compressed — recommended
        bits_lzma_q8 = compression_length_bits(model, compressor="lzma", quantize_bits=8)
        self.metrics_df.loc[len(self.metrics_df)] = {
            "epoch": epoch,
            "input_type": "general",
            "metric_name": "compression_lzma_bits_q8",
            "layer": np.nan,
            "value": bits_lzma_q8,
        }

    
    def log_ph0_metrics(self, epoch, activations, layer_name, max_points=512, norm_mode="row_l2", alphas=(0.5, 1.0, 2.0)):
        X = activations.detach().float().cpu().numpy()

        if X.ndim == 3:
            X = X[:, -1, :]
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        X = X[np.isfinite(X).all(axis=1)]
        if X.shape[0] < 64:
            summaries = {
                "ph0_total_persistence": float("nan"),
                "ph0_persistence_entropy": float("nan"),
                "ph0_num_long_edges_p90": float("nan"),
                "ph0_max_edge": float("nan"),
            }
            e_alpha_vals = {alpha: float("nan") for alpha in alphas}
        else:
            if X.shape[0] > max_points:
                idx = self.rng.choice(X.shape[0], size=max_points, replace=False)
                X = X[idx]

            # if you added normalization earlier:
            X = normalize_point_cloud(X, mode=norm_mode)

            edges = mst_edges(X)
            summaries = ph0_summaries_from_mst_edges(edges)
            e_alpha_vals = {alpha: e_alpha_from_mst_edges(edges, alpha) for alpha in alphas}

        # log PH0 summaries
        for metric_name, val in summaries.items():
            self.dims_df.loc[len(self.dims_df)] = {
                "epoch": epoch,
                "layer": layer_name,
                "metric_name": f"{metric_name}_{norm_mode}",
                "value": float(val),
            }

        # log E_alpha
        for alpha, val in e_alpha_vals.items():
            self.dims_df.loc[len(self.dims_df)] = {
                "epoch": epoch,
                "layer": layer_name,
                "metric_name": f"E_alpha_{alpha:g}_{norm_mode}",
                "value": float(val),
            }


    

    def log_mst_dimension(self, epoch, activations, layer_name, max_points=1024, norm_mode="row_l2"):
        X = activations.detach().float().cpu().numpy()

        # Flatten like your other metrics
        if X.ndim == 3:
            X = X[:, -1, :]
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        X = X[np.isfinite(X).all(axis=1)]
        if X.shape[0] > max_points:
            idx = self.rng.choice(X.shape[0], size=max_points, replace=False)
            X = X[idx]

        n_list = tuple(n for n in (64, 128, 256, 512) if n <= X.shape[0])
        if len(n_list) < 2:
            mst_dim, mst_alpha, mst_r2 = float("nan"), float("nan"), float("nan")
        else:
            X = normalize_point_cloud(X, mode=norm_mode)
            mst_dim, mst_alpha, mst_r2 = mst_dimension_with_fit(
                X, n_list=n_list, n_repeats=3
            )

        # log mst_dim
        self.dims_df.loc[len(self.dims_df)] = {
            "epoch": epoch,
            "layer": layer_name,
            "metric_name": "mst_dim",
            "value": mst_dim,
        }
        # log fit slope
        self.dims_df.loc[len(self.dims_df)] = {
            "epoch": epoch,
            "layer": layer_name,
            "metric_name": "mst_alpha",
            "value": mst_alpha,
        }
        # log fit quality
        self.dims_df.loc[len(self.dims_df)] = {
            "epoch": epoch,
            "layer": layer_name,
            "metric_name": "mst_r2",
            "value": mst_r2,
        }

    def log_corr_dimension(self, epoch, activations, layer_name, max_points=512, norm_mode="row_l2"):
        X = activations.detach().float().cpu().numpy()

        # Flatten
        if X.ndim == 3:
            X = X[:, -1, :]
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        X = normalize_point_cloud(X, mode=norm_mode)


        corr_dim, corr_r2 = correlation_dimension(X, max_points=max_points)

        self.dims_df.loc[len(self.dims_df)] = {
            "epoch": epoch,
            "layer": layer_name,
            "metric_name": "corr_dim",
            "value": corr_dim,
        }
        self.dims_df.loc[len(self.dims_df)] = {
            "epoch": epoch,
            "layer": layer_name,
            "metric_name": "corr_r2",
            "value": corr_r2,
        }



    
    def log_weight_spectrum_metrics(self, epoch_position: int, model):
        """
        Logs weight spectral entropy (+ optionally effective rank) for each linear layer weight.
        Appends to self.dims_df using the same schema: epoch, layer, metric_name, value.
        """
        for name, param in model.named_parameters():
            if not name.endswith(".weight"):
                continue
            # Fix T7: this used to require name.startswith("layers."), which is true
            # only for the MLP. A Transformer's parameters are transformer.layers.*,
            # embedding.*, linear.* -- so it silently logged ZERO weight rows.
            # Accept any 2-D weight matrix instead.
            if param.ndim != 2:
                continue

            W = param.detach().float().cpu().numpy()
            if W.ndim != 2:
                continue

            H = weight_spectral_entropy(W)
            self.dims_df.loc[len(self.dims_df)] = {
                "epoch": epoch_position,
                "layer": name,  # e.g. layers.0.weight
                "metric_name": "weight_spectral_entropy",
                "value": H,
            }

            # Optional: also log effective rank
            R = weight_effective_rank(W)
            self.dims_df.loc[len(self.dims_df)] = {
                "epoch": epoch_position,
                "layer": name,
                "metric_name": "weight_effective_rank",
                "value": R,
            }

    

    def log_twonn_dimension(self, epoch, activations, layer_name, max_points=1024, norm_mode="row_l2"):
        X = activations.detach().float().cpu().numpy()

        # flatten if needed
        if X.ndim == 3:
            X = X[:, -1, :]
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        # subsample
        if X.shape[0] > max_points:
            X = X[:max_points]

        # drop non-finite rows
        mask = np.isfinite(X).all(axis=1)
        X = X[mask]

        X = normalize_point_cloud(X, mode=norm_mode)

        val = twonn_intrinsic_dimension(X)

        self.dims_df.loc[len(self.dims_df)] = {
            "epoch": epoch,
            "layer": layer_name,
            "metric_name": "twonn_id",
            "value": val,
        }




    def log_participation_ratio(
        self,
        epoch: int,
        activations: torch.Tensor,
        layer_name: str,
    ):
        """
        activations: torch.Tensor of shape (n_samples, d)
        """
        X = activations.detach().cpu().numpy()
        X = X - X.mean(axis=0, keepdims=True)

        pr = participation_ratio(X)

        self.dims_df.loc[len(self.dims_df)] = {
            "epoch": epoch,
            "layer": layer_name,
            "metric_name": "pca_participation_ratio",
            "value": pr,
        }



    def _get_epoch_position(self, epoch: int) -> int:
        epoch_position = epoch // self.log_frequency
        if epoch > self.num_early_training_epochs:
            epoch_position += self.num_early_training_epochs // self.early_training_log_frequency
        return epoch_position

    def _run_full_batch_forward(self, model, data, targets, args):
        model.eval()
        device = next(model.parameters()).device
        data = data.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            output = model(data)
            if args.use_transformer:
                output = output[:,-1]
        return output, targets

    def log_metrics(self, model, epoch, save_model_checkpoints, saved_models,
                    all_data, all_targets, all_test_data, all_test_targets,
                    args, loss_function):
        """
        Compute and log all metrics in a full-batch setting.
        """

        epoch_position = self._get_epoch_position(epoch)

        if epoch in save_model_checkpoints:
            saved_models[epoch] = copy.deepcopy(model.state_dict())


        self._train_output, self._train_targets = self._run_full_batch_forward(model, all_data, all_targets, args)
        self._test_output, self._test_targets = self._run_full_batch_forward(model, all_test_data, all_test_targets, args)

        # ---- Dimension logging (v1): PCA PR on internal activations (train only) ----
        with torch.no_grad():
            if self.probe_layers is None:
                self.probe_layers, self.pre_logits_source = resolve_probe_layers(model)
                print(f"[logger] probes={self.probe_layers} pre_logits<-{self.pre_logits_source}")
            layer_names = self.probe_layers

            x_sub = all_data[: min(1024, all_data.shape[0])]

            # The pre_logits source must be HOOKED even when it is not itself a probe
            # (for the Transformer it is the final `linear`, whose own output is the
            # logits, not a representation). Hook it, then drop its raw output.
            hook_names = list(layer_names)
            if self.pre_logits_source and self.pre_logits_source not in hook_names:
                hook_names.append(self.pre_logits_source)
            acts = self._get_layer_activations(model, x_sub, hook_names)
            if self.pre_logits_source not in layer_names:
                acts.pop(self.pre_logits_source, None)

            for lname_raw, A in list(acts.items()):
                lname = probe_display_name(lname_raw)
                # Flatten transformer-style outputs if needed
                # If shape is (n, t, d), take last token
                if A.ndim == 3:
                    A = A[:, -1, :]

                # If layer outputs are not already 2D, flatten all but batch
                if A.ndim > 2:
                    A = A.flatten(start_dim=1)

                self.log_participation_ratio(
                    epoch=epoch,
                    activations=A,
                    layer_name=lname,
                )

                self.log_twonn_dimension(epoch, A, layer_name=lname)
                self.log_mst_dimension(epoch, A, layer_name=lname)
                self.log_corr_dimension(epoch, A, layer_name=lname)
                self.log_ph0_metrics(epoch, A, layer_name=lname)           # already includes E_alpha
                self.log_magnitude_metrics(epoch, A, layer_name=lname)     # logs mag + pmag

            
            self.log_weight_spectrum_metrics(epoch, model)
            self.log_compression_length(epoch, model)

            # ---- after per-layer logging ----
            Z = self._make_global_activation(acts, layer_names)  # probes only

            global_name = "all_layers_concat"

            self.log_participation_ratio(epoch=epoch, activations=Z, layer_name=global_name)
            self.log_twonn_dimension(epoch, Z, layer_name=global_name)
            self.log_mst_dimension(epoch, Z, layer_name=global_name)
            self.log_corr_dimension(epoch, Z, layer_name=global_name)
            self.log_ph0_metrics(epoch, Z, layer_name=global_name)
            self.log_magnitude_metrics(epoch, Z, layer_name=global_name)




        x_h = all_data[: min(256, all_data.shape[0])]
        y_h = all_targets[: min(256, all_targets.shape[0])]

        self.log_hessian_spectrum_metrics(
                    epoch=epoch,
                    model=model,
                    x=x_h,
                    y=y_h,
                    loss_function=loss_function,
                    args=args,
                    trace_samples=16,
                    frob_samples=16,
                    power_iters=30,
                    which_norm="spectral",  # or "frobenius"
                    seed=0,
        ) 



        for metric_name, metric_fn in self.metric_fns.items():
            rows = metric_fn(model=model,
                             epoch=epoch,
                             epoch_position=epoch_position,
                             args=args,
                             loss_function=loss_function)
            if rows is not None and len(rows) > 0:
                self.metrics_df = pd.concat([self.metrics_df, pd.DataFrame(rows)], ignore_index=True)


        self._train_output = None
        self._train_targets = None
        self._test_output = None
        self._test_targets = None


        


    def compute_loss(self, model, epoch, epoch_position, args, loss_function):
        ce_dtype = getattr(torch, args.cross_entropy_dtype)
        train_loss_val = loss_function(self._train_output, self._train_targets, dtype=ce_dtype).item()
        test_loss_val = loss_function(self._test_output, self._test_targets, dtype=ce_dtype).item()

        return [
            {
                "epoch": epoch,
                "input_type": "train",
                "metric_name": "loss",
                "layer": None,
                "value": train_loss_val
            },
            {
                "epoch": epoch,
                "input_type": "test",
                "metric_name": "loss",
                "layer": None,
                "value": test_loss_val
            }
        ]

    def compute_accuracy(self, model, epoch, epoch_position, args, loss_function):
        train_preds = self._train_output.argmax(dim=1)
        test_preds = self._test_output.argmax(dim=1)

        accuracy = lambda correct: correct.sum().item() / correct.size(0)

        train_acc = accuracy(train_preds == self._train_targets)
        test_acc = accuracy(test_preds == self._test_targets)

        return [
            {
                "epoch": epoch,
                "input_type": "train",
                "metric_name": "accuracy",
                "layer": None,
                "value": train_acc
            },
            {
                "epoch": epoch,
                "input_type": "test",
                "metric_name": "accuracy",
                "layer": None,
                "value": test_acc
            }
        ]

    def compute_weights_l2(self, model, epoch, epoch_position, args, loss_function):
        results = []
        for name, param in model.named_parameters():
            val = param.square().sum().sqrt().item()
            results.append({
                "epoch": epoch,
                "input_type": "general",
                "metric_name": "weights_l2",
                "layer": name,
                "value": val
            })
        return results

    def compute_zero_terms(self, model, epoch, epoch_position, args, loss_function):
        ce_dtype = getattr(torch, args.cross_entropy_dtype)
        full_loss = loss_function(self._train_output, self._train_targets, reduction="none", dtype=ce_dtype)
        zero_val = ((full_loss == 0).sum().item() / (full_loss.shape[0]))
        return [{
            "epoch": epoch,
            "input_type": "train",
            "metric_name": "zero_terms",
            "layer": None,
            "value": zero_val
        }]

    def compute_softmax_collapse(self, model, epoch, epoch_position, args, loss_function):
        ce_dtype = getattr(torch, args.cross_entropy_dtype)
        output = self._train_output.to(ce_dtype)
        output_off = output - output.amax(dim=1, keepdim=True)
        exp_output = torch.exp(output_off)
        sum_exp = torch.sum(exp_output, dim=-1, keepdim=True)
        log_softmax = output_off.amax(dim=1, keepdim=True)- torch.log(sum_exp)
        softmax_collapse = (sum_exp==1).float().mean().item()

        return [{
            "epoch": epoch,
            "input_type": "train",
            "metric_name": "softmax_collapse",
            "layer": None,
            "value": softmax_collapse
        }]