"""
Geometric measure battery for the activation manifold.

Deliberately spans FOUR different notions of "dimension", because the interesting question is
not what any one of them says but whether they disagree:

  fractal / topological  ph_dim (MST alpha-weight scaling), correlation_dim
  linear                 pca_participation_ratio, effective_rank
  local-scaling          twonn, mle_id (Levina-Bickel)
  per-point              lid_distribution  -> mean/sd/skew/bimodality of local ID

plus curvature (Ollivier-Ricci on the kNN graph), magnitude (multiscale effective size),
anisotropy (spectral decay), and neural-collapse statistics.

Every estimator here has a known-answer test in tools/geom_validate.py, and none is used before
it passes. Scale handling matters: correlation_dim and ph_dim are scale-free by construction,
magnitude is NOT (it has units of 1/distance), so magnitude is always reported on a cloud
normalised to unit mean pairwise distance.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import skew
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------- helpers
def _D(X):
    return squareform(pdist(np.asarray(X, float)))


def unit_scale(X):
    """Scale so mean pairwise distance is 1. Needed for any scale-dependent measure."""
    X = np.asarray(X, float) - np.mean(X, 0)
    d = pdist(X)
    m = d.mean()
    return X / m if m > 0 else X


# ---------------------------------------------------------------- fractal / topological
def ph_dim(X, alpha=1.0, n_list=None, n_repeats=5, seed=0):
    """Birdal-style PH dimension: E_alpha = sum of MST edge weights^alpha, dim = alpha/(1-slope)."""
    X = np.asarray(X, float)
    n = len(X)
    if n_list is None:
        hi = min(n, 1000)
        lo = max(50, hi // 5)
        n_list = list(range(lo, hi + 1, max(1, (hi - lo) // 10)))
    n_list = [m for m in n_list if 5 <= m <= n]
    if len(n_list) < 4:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for m in n_list:
        vals = []
        for _ in range(n_repeats):
            idx = rng.choice(n, m, replace=False)
            mst = minimum_spanning_tree(_D(X[idx]))
            w = mst.data
            vals.append(np.sum(w ** alpha))
        xs.append(np.log(m)); ys.append(np.log(np.mean(vals)))
    xs, ys = np.array(xs), np.array(ys)
    A = np.vstack([xs, np.ones_like(xs)]).T
    slope, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
    pred = A @ np.linalg.lstsq(A, ys, rcond=None)[0]
    ss = 1 - np.sum((ys - pred) ** 2) / max(np.sum((ys - ys.mean()) ** 2), 1e-12)
    if slope >= 1 - 1e-6:
        return np.nan, float(ss)
    return float(alpha / (1 - slope)), float(ss)


def correlation_dim(X, q_lo=0.05, q_hi=0.35):
    """Grassberger-Procaccia: slope of log C(r) vs log r in a mid-range scaling window."""
    d = pdist(np.asarray(X, float))
    d = d[d > 0]
    if len(d) < 20:
        return np.nan, np.nan
    rs = np.exp(np.linspace(np.log(np.quantile(d, q_lo)), np.log(np.quantile(d, q_hi)), 14))
    C = np.array([(d < r).mean() for r in rs])
    ok = C > 0
    if ok.sum() < 5:
        return np.nan, np.nan
    x, y = np.log(rs[ok]), np.log(C[ok])
    A = np.vstack([x, np.ones_like(x)]).T
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    pred = A @ coef
    ss = 1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
    return float(coef[0]), float(ss)


# ---------------------------------------------------------------- linear
def _eigs(X):
    X = np.asarray(X, float)
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2
    return lam[lam > 0]


def pca_participation_ratio(X):
    lam = _eigs(X)
    return float(lam.sum() ** 2 / np.sum(lam ** 2)) if len(lam) else np.nan


def effective_rank(X):
    """exp(Shannon entropy of the normalised eigenvalue spectrum)."""
    lam = _eigs(X)
    if not len(lam):
        return np.nan
    p = lam / lam.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def spectral_decay(X):
    """Slope of log eigenvalue vs log index -- anisotropy / power-law tail."""
    lam = _eigs(X)
    k = min(len(lam), 50)
    if k < 5:
        return np.nan
    x, y = np.log(np.arange(1, k + 1)), np.log(lam[:k])
    return float(np.polyfit(x, y, 1)[0])


# ---------------------------------------------------------------- local-scaling
def twonn(X, discard=0.1):
    """Facco 2017. mu = r2/r1; d from the CDF fit, discarding the heavy tail."""
    X = np.asarray(X, float)
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    d, _ = nn.kneighbors(X)
    r1, r2 = d[:, 1], d[:, 2]
    ok = r1 > 0
    mu = r2[ok] / r1[ok]
    mu = np.sort(mu[mu > 1])
    if len(mu) < 10:
        return np.nan
    n = len(mu)
    keep = int(n * (1 - discard))
    F = np.arange(1, n + 1) / n
    x, y = np.log(mu[:keep]), -np.log(1 - F[:keep])
    return float(np.sum(x * y) / np.sum(x * x))


def gride(X, n1=1, n2=None):
    """
    GRIDE, the generalized-ratios ID estimator of Denti et al. (Sci. Rep. 2022). Same family as
    TwoNN but with a tunable *scale*: instead of r_2/r_1 it uses mu = r_{n2}/r_{n1} for n2 > n1,
    so n1 sets how far the estimator reaches into the cloud. n2 = 2*n1 by default, as in the paper.

    Derivation of the likelihood, so the implementation is checkable rather than copied. Under a
    locally-uniform Poisson model on a d-manifold the enclosed volumes v_j proportional to r_j^d
    are Poisson arrival times, so v_{n1}/v_{n2} ~ Beta(n1, n2-n1) and mu^{-d} is that Beta. Change
    of variables gives

        f(mu | d) = d * mu^{-(d*n1 + 1)} * (1 - mu^{-d})^{n2-n1-1} / B(n1, n2-n1),   mu >= 1

    which at (n1, n2) = (1, 2) collapses to the Pareto(d) density TwoNN is built on -- that identity
    is the first validation gate in tools/b1_gride.py. d is the maximum-likelihood value.

    Why this matters here: n1 is a knob on neighbourhood scale with the estimator family held
    fixed, so it isolates the scale-locality effect from every other difference between estimators.
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import betaln

    X = np.asarray(X, float)
    n2 = 2 * n1 if n2 is None else n2
    if n2 <= n1 or len(X) <= n2:
        return np.nan
    nn = NearestNeighbors(n_neighbors=n2 + 1).fit(X)
    dist, _ = nn.kneighbors(X)
    r1, r2 = dist[:, n1], dist[:, n2]
    ok = (r1 > 0) & (r2 > r1)
    mu = r2[ok] / r1[ok]
    if len(mu) < 10:
        return np.nan
    lmu = np.log(mu)
    k = n2 - n1
    const = betaln(n1, k)

    def nll(ld):
        d = np.exp(ld)
        t = -d * lmu                                  # log(mu^{-d})
        # log1p(-exp(t)) is the numerically safe form of log(1 - mu^{-d})
        term = np.log1p(-np.exp(np.minimum(t, -1e-12))) if k > 1 else 0.0
        return -float(np.sum(ld - (d * n1 + 1) * lmu + (k - 1) * term - const))

    r = minimize_scalar(nll, bounds=(np.log(1e-3), np.log(1e3)), method="bounded")
    return float(np.exp(r.x)) if r.success else np.nan


def lid_distribution(X, k=20):
    """Per-point Levina-Bickel MLE local ID. Returns the vector, not just its mean."""
    X = np.asarray(X, float)
    k = min(k, len(X) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    d, _ = nn.kneighbors(X)
    d = d[:, 1:]
    out = np.full(len(X), np.nan)
    for i in range(len(X)):
        r = d[i]
        r = r[r > 0]
        if len(r) < 3:
            continue
        Tk = r[-1]
        out[i] = (len(r) - 1) / np.sum(np.log(Tk / r[:-1]))
    return out[np.isfinite(out)]


def mle_id(X, k=20):
    v = lid_distribution(X, k)
    return float(np.mean(v)) if len(v) else np.nan


def bimodality(v):
    """Sarle's bimodality coefficient: >5/9 suggests a bimodal (or heavily skewed) shape."""
    v = np.asarray(v, float)
    if len(v) < 8:
        return np.nan
    from scipy.stats import kurtosis
    g, k_ = skew(v), kurtosis(v, fisher=True)
    n = len(v)
    denom = 3 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
    return float((g ** 2 + 1) / (k_ + denom))


# ---------------------------------------------------------------- curvature
def ollivier_ricci_graph(adj, alpha=0.0, max_edges=1500, seed=0):
    """
    Ollivier-Ricci on a weighted graph: kappa(x,y) = 1 - W1(m_x, m_y) / d_G(x,y), with m_x the
    one-step measure on x's neighbours and BOTH the transport cost and the denominator using the
    graph shortest-path metric d_G.

    Using ambient Euclidean distance instead of d_G is wrong and silently inverts the sign on
    tree-like data -- two points on different branches are Euclidean-close but graph-far, so a
    tree reads as positively curved. That failure was caught at the gate; hence d_G here.
    """
    import ot
    from scipy.sparse.csgraph import shortest_path, connected_components
    from scipy.sparse import csr_matrix
    A = csr_matrix(adj)
    n = A.shape[0]
    ncomp, lab = connected_components(A, directed=False)
    if ncomp > 1:                       # restrict to the giant component; d_G is infinite across
        keep = np.where(lab == np.argmax(np.bincount(lab)))[0]
        A = A[keep][:, keep]
        n = A.shape[0]
    DG = shortest_path(A, directed=False)
    if not np.isfinite(DG).all():
        DG[~np.isfinite(DG)] = np.nanmax(DG[np.isfinite(DG)]) * 2
    Ad = A.toarray()
    nbrs = [np.where(Ad[i] > 0)[0] for i in range(n)]
    edges = [(i, int(j)) for i in range(n) for j in nbrs[i] if i < j]
    if len(edges) > max_edges:
        rng = np.random.default_rng(seed)
        edges = [edges[t] for t in rng.choice(len(edges), max_edges, replace=False)]
    ks = []
    for i, j in edges:
        ni, nj = nbrs[i], nbrs[j]
        if len(ni) == 0 or len(nj) == 0 or DG[i, j] <= 0:
            continue
        mi = np.full(len(ni), (1 - alpha) / len(ni))
        mj = np.full(len(nj), (1 - alpha) / len(nj))
        if alpha > 0:
            ni = np.r_[i, ni]; nj = np.r_[j, nj]
            mi = np.r_[alpha, mi]; mj = np.r_[alpha, mj]
        w1 = ot.emd2(mi / mi.sum(), mj / mj.sum(), DG[np.ix_(ni, nj)])
        ks.append(1 - w1 / DG[i, j])
    ks = np.array(ks)
    return {"orc_mean": float(ks.mean()) if len(ks) else np.nan,
            "orc_sd": float(ks.std()) if len(ks) else np.nan,
            "orc_frac_neg": float((ks < 0).mean()) if len(ks) else np.nan}


def knn_graph(X, k=10, symmetric=True):
    """Weighted kNN graph with Euclidean edge weights (the graph metric is derived from it)."""
    from scipy.sparse import csr_matrix
    X = np.asarray(X, float)
    n = len(X)
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, idx = nn.kneighbors(X)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    w = dist[:, 1:].ravel()
    w[w <= 0] = 1e-12
    A = csr_matrix((w, (rows, cols)), shape=(n, n))
    if symmetric:
        A = A.maximum(A.T)
    return A


def ollivier_ricci(X, k=10, alpha=0.0, max_edges=1500, seed=0):
    """
    Ollivier-Ricci on the kNN graph of a point cloud.

    *** GATE-FAILED AS A CURVATURE ESTIMATOR -- DO NOT INTERPRET AS MANIFOLD CURVATURE. ***

    ollivier_ricci_graph passes cleanly on explicit graphs (K_30 +0.97, cycle 0.00, lattice
    0.00, tree -0.32, random 3-regular -0.57). But on point clouds sampled from a manifold the
    value is set by the neighbourhood scale and the intrinsic dimension, not by curvature:
      - a flat 2D grid reads MORE positive than a 2-sphere at every k >= 8
      - every cloud tends to +0.2..+0.3 as k grows (dense kNN graphs are clique-like)
      - a FLAT 8-ball reads -0.32 while a FLAT 2D grid reads +0.03 at matched radius
    So the sign tracks dimension and density. Since intrinsic dimension is precisely what moves
    during grokking, curvature cannot be disentangled from it here.

    Retained only as a descriptive graph statistic at matched n and k, and used in STEP 3 to
    TEST the gate's prediction that it is redundant with the dimension axis.
    """
    return ollivier_ricci_graph(knn_graph(X, k), alpha=alpha, max_edges=max_edges, seed=seed)


# ---------------------------------------------------------------- magnitude
def magnitude_function(X, ts=None, unit=True):
    """
    |tX| = sum of entries of Z^{-1}, Z_ij = exp(-t d_ij). Scale-dependent by definition, so the
    cloud is unit-scaled first and the profile is reported over a fixed t grid.
    """
    Y = unit_scale(X) if unit else np.asarray(X, float)
    D = _D(Y)
    if ts is None:
        ts = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    out = {}
    for t in ts:
        Z = np.exp(-t * D)
        try:
            out[f"mag_t{t:g}"] = float(np.linalg.solve(Z, np.ones(len(Z))).sum())
        except np.linalg.LinAlgError:
            out[f"mag_t{t:g}"] = np.nan
    return out


# ---------------------------------------------------------------- neural collapse
def neural_collapse(X, y):
    """
    NC1  trace(Sigma_W Sigma_B^+)/K   -- within/between variance ratio, ->0 at collapse
    NC2  sd of off-diagonal cosines between centred class means (simplex ETF -> 0)
    """
    X = np.asarray(X, float); y = np.asarray(y)
    cls = np.unique(y)
    K = len(cls)
    if K < 2:
        return {"nc1": np.nan, "nc2_etf": np.nan, "nc_wb": np.nan}
    mu = X.mean(0)
    M = np.stack([X[y == c].mean(0) for c in cls])
    Sb = np.cov((M - mu).T, bias=True)
    Sw = np.zeros_like(Sb)
    for i, c in enumerate(cls):
        Z = X[y == c] - M[i]
        if len(Z) > 1:
            Sw += Z.T @ Z
    Sw /= len(X)
    nc1 = float(np.trace(Sw @ np.linalg.pinv(Sb)) / K)
    Mc = M - mu
    Mc = Mc / (np.linalg.norm(Mc, axis=1, keepdims=True) + 1e-12)
    G = Mc @ Mc.T
    off = G[~np.eye(K, dtype=bool)]
    nc2 = float(np.std(off + 1.0 / (K - 1)))
    wb = float(np.trace(Sw) / max(np.trace(Sb), 1e-12))
    return {"nc1": nc1, "nc2_etf": nc2, "nc_wb": wb}


# ---------------------------------------------------------------- battery
def clusteredness(X, k=10):
    """
    How cluster-like is the cloud, independent of any label. Needed because GATE 7 showed
    ph_dim and MLE-ID are driven to ~1 and ~0 by CLUSTERING alone, whatever the true
    within-cluster dimension. Without this control, "dimension dropped" and "the cloud broke
    into clumps" are indistinguishable.

    gap_ratio  median(distance to k-th NN) / median(pairwise distance): small when tight
               clumps sit far apart.
    dup_frac   fraction of points with a near-coincident neighbour.
    """
    X = np.asarray(X, float)
    n = len(X)
    k = min(k, n - 1)
    d, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    pw = pdist(X)
    med = np.median(pw)
    return {"gap_ratio": float(np.median(d[:, -1]) / med) if med > 0 else np.nan,
            "nn1_ratio": float(np.median(d[:, 1]) / med) if med > 0 else np.nan,
            "dup_frac": float((d[:, 1] < 1e-3 * med).mean())}


def battery(X, y=None, k_orc=10, cheap=False):
    X = np.asarray(X, float)
    out = {}
    pd_, pd_r2 = ph_dim(X)
    cd, cd_r2 = correlation_dim(X)
    out.update({"ph_dim": pd_, "ph_dim_r2": pd_r2, "corr_dim": cd, "corr_dim_r2": cd_r2})
    out["pca_pr"] = pca_participation_ratio(X)
    out["eff_rank"] = effective_rank(X)
    out["spec_decay"] = spectral_decay(X)
    out["twonn"] = twonn(X)
    # MLE-ID at several k: GATE 7 showed a single large k silently fails once the cloud
    # clusters (k spans the whole clump), so the k-profile is reported, not one number.
    for k in (5, 10, 20):
        v = lid_distribution(X, k)
        out[f"mle_id_k{k}"] = float(np.mean(v)) if len(v) else np.nan
        if k == 10:
            out["lid_sd"] = float(np.std(v)) if len(v) else np.nan
            out["lid_skew"] = float(skew(v)) if len(v) > 8 else np.nan
            out["lid_bimod"] = bimodality(v)
    out["mle_id"] = out["mle_id_k20"]
    out.update(clusteredness(X))
    if not cheap:
        out.update(ollivier_ricci(X, k=k_orc))
        out.update(magnitude_function(X))
    if y is not None:
        out.update(neural_collapse(X, y))
    return out
