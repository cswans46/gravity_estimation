"""Spatial IRIO: Bayesian multi-equation spatial Durbin model (Torój 2016, 2021, 2024, 2025).

Model (Torój 2016 eqs. 7-11, 19-23; Torój 2024/2025 Bayesian version)
--------------------------------------------------------------------
For each supplying sector s = 1..S, regions r = 1..R and year t:

    va^s_t = W_{s,t} ( sum_v beta^s_{v,t} va^v_t + beta^s_{0,t} y^s_t ) + eps^s_t,
    eps^s_t ~ N(0, sigma_s^2 I_R)

with *known* national coefficients (eqs. 8-9)

    beta^s_v = (VA^s / X^s) * Z^{sv} / VA^v,      beta^s_0 = VA^s / X^s,

so that summing over regions reproduces the national identity exactly. The only
unknowns are in the R x R spatial weight ("lead") matrices, one per sector:

    W_s(r, p) = g(d_rp; shape_s, scale_s) * m^s_r ** gamma_s
                / sum_r' g(d_r'p; shape_s, scale_s) * m^s_r' ** gamma_s

    g(d) = 1 - F_Gamma(d; shape, scale)      (gamma survival function, Torój 2016 eq. 19)

columns sum to one: W_s(r, p) is the share of region p's demand for s supplied by r.
m^s_r is the supply mass (regional share of sector-s value added by default; eq. 20).

Stacked over sectors (eq. 22-23): A_t va_t = b_t + eps_t with
    A_t = I_{RS} - [beta^s_{v,t} W_{s,t}]_{s,v},   b_t = [beta^s_{0,t} W_{s,t} y^s_t]_s

Log-likelihood:  sum_t log|A_t| - sum_s (n/2) log sigma_s^2 - sum_s SSR_s / (2 sigma_s^2).
With the Jeffreys prior p(sigma_s^2) ∝ 1/sigma_s^2, sigma_s^2 integrates out:

    log p(phi | data) = sum_t log|A_t(phi)| - (n_eff/2) sum_s log SSR_s(phi) + log p(phi)

phi_s = (log shape_s, log scale_s[, logit gamma_s]) is sampled by Metropolis-within-
Gibbs (one block per sector, adaptive Gaussian random walk). Updating one sector
changes R rows of A_t, so log|A_t| is updated with the matrix determinant lemma and
A_t^{-1} with Woodbury (O(R^2 RS) per proposal instead of O((RS)^3)).

Flows (Torój 2016 eqs. 4, 6):   Z[r, s, p, v] = colZ[s, p, v] * W_s(r, p)
with colZ[s,p,v] = Z^{sv}_national * va^v_p / VA^v by default, or user-supplied
column totals (e.g. provincial use tables).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, special, stats


# ----------------------------------------------------------------------------- data
@dataclass
class SDMData:
    """Inputs for the spatial IRIO model.

    va    : (T, S, R) regional value added by sector
    y     : (T, S, R) regional final demand for commodity s (national net final
            demand regionalised, e.g. by population, Torój 2016)
    Znat  : (T, S, S) national intermediate flows Z^{sv} (domestic table)
    Xnat  : (T, S)    national gross output
    dist  : (R, R)    distances; diagonal = internal distance (0 allowed for 'sf')
    mass  : (T, S, R) supply mass for W (default: va). Note va is the dependent
            variable, so the default makes W mildly endogenous (as in Torój 2016).
    """
    va: np.ndarray
    y: np.ndarray
    Znat: np.ndarray
    Xnat: np.ndarray
    dist: np.ndarray
    mass: np.ndarray | None = None
    sectors: list | None = None
    regions: list | None = None

    def __post_init__(self):
        self.va = np.asarray(self.va, float)
        T, S, R = self.va.shape
        self.y = np.asarray(self.y, float).reshape(T, S, R)
        self.Znat = np.asarray(self.Znat, float).reshape(T, S, S)
        self.Xnat = np.asarray(self.Xnat, float).reshape(T, S)
        self.dist = np.asarray(self.dist, float)
        self.mass = self.va.copy() if self.mass is None else np.asarray(self.mass, float)
        self.T, self.S, self.R = T, S, R

    def betas(self):
        """beta[t, s, v] and beta0[t, s] from national ratios (eqs. 8-9)."""
        VA = self.va.sum(axis=2)  # (T,S)
        ratio = np.divide(VA, self.Xnat, out=np.zeros_like(VA), where=self.Xnat > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = ratio[:, :, None] * self.Znat / np.where(VA > 0, VA, np.inf)[:, None, :]
        return np.nan_to_num(beta), ratio

    def national_identity_gap(self):
        """Max relative gap of VA^s = sum_v beta VA^v + beta0 Y^s (should be ~0)."""
        beta, beta0 = self.betas()
        VA = self.va.sum(axis=2)
        Y = self.y.sum(axis=2)
        rhs = np.einsum("tsv,tv->ts", beta, VA) + beta0 * Y
        return float(np.max(np.abs(rhs - VA) / np.maximum(np.abs(VA), 1e-12)))


# -------------------------------------------------------------------------- weights
def decay(d, shape, scale, kind: str = "sf"):
    """Gamma distance-decay g(d). 'sf' = 1 - CDF (Torój 2016), 'pdf' = density."""
    if kind == "sf":
        return stats.gamma.sf(d, a=shape, scale=scale)
    if kind == "pdf":
        return stats.gamma.pdf(d, a=shape, scale=scale)
    raise ValueError(kind)


def log_decay(d, shape, scale, kind: str = "sf"):
    """log g(d); uses scipy.special directly (much faster than scipy.stats in the MCMC loop)."""
    x = np.asarray(d, float) / scale
    if kind == "sf":
        with np.errstate(divide="ignore"):
            return np.log(special.gammaincc(shape, x))
    if kind == "pdf":
        with np.errstate(divide="ignore"):
            return special.xlogy(shape - 1, x) - x - special.gammaln(shape) - np.log(scale)
    raise ValueError(kind)


def weight_matrix(dist, shape, scale, mass_share=None, gamma=1.0, kind: str = "sf"):
    """Column-normalised W(r, p). mass_share: (R,) shares (or None = no supply term)."""
    lg = log_decay(dist, shape, scale, kind)
    if mass_share is not None and gamma != 0:
        with np.errstate(divide="ignore"):
            lg = lg + gamma * np.log(np.asarray(mass_share))[:, None]
    lg = lg - np.max(np.where(np.isfinite(lg), lg, -np.inf), axis=0, keepdims=True)
    w = np.exp(lg)
    w = np.nan_to_num(w)
    cs = w.sum(axis=0, keepdims=True)
    return np.divide(w, cs, out=np.full_like(w, 1.0 / w.shape[0]), where=cs > 0)


# --------------------------------------------------------------------------- priors
@dataclass
class SDMPrior:
    """Independent priors per sector.

    log shape ~ N(mu_logshape[s], sd_logshape[s]);  log scale ~ N(mu_logscale[s], sd_logscale[s])
    gamma (if estimated) ~ Beta(a_gamma, b_gamma)
    """
    mu_logshape: np.ndarray
    sd_logshape: np.ndarray
    mu_logscale: np.ndarray
    sd_logscale: np.ndarray
    a_gamma: float = 1.0
    b_gamma: float = 1.0

    @classmethod
    def weak(cls, S, dist, sd_logshape=1.0, sd_logscale=1.5):
        off = dist[~np.eye(len(dist), dtype=bool)]
        med = float(np.median(off[off > 0]))
        return cls(np.zeros(S), np.full(S, sd_logshape), np.full(S, np.log(med)), np.full(S, sd_logscale))

    def logpdf(self, s, phi, estimate_gamma):
        lp = stats.norm.logpdf(phi[0], self.mu_logshape[s], self.sd_logshape[s])
        lp += stats.norm.logpdf(phi[1], self.mu_logscale[s], self.sd_logscale[s])
        if estimate_gamma:
            g = special.expit(phi[2])
            # Beta prior on gamma + Jacobian of the logit transform
            lp += stats.beta.logpdf(g, self.a_gamma, self.b_gamma) + np.log(g) + np.log1p(-g)
        return lp


def fit_decay_to_power_law(delta, dist, kind: str = "sf", d_range=None):
    """Gamma (shape, scale) whose decay best matches d^-delta over the observed distances.

    Used to build a prior from the ICIO distance elasticities (this project's analogue
    of Torój 2024's use of Finnish regional tables). Least squares on log decay,
    up to a free constant, over the off-diagonal distances (or d_range grid).
    """
    d = np.asarray(dist, float)
    pts = d[~np.eye(len(d), dtype=bool)] if d_range is None else np.asarray(d_range, float)
    pts = np.unique(pts[pts > 0])
    target = -delta * np.log(pts)

    def loss(p):
        lg = log_decay(pts, np.exp(p[0]), np.exp(p[1]), kind)
        r = (lg - lg.mean()) - (target - target.mean())
        return np.sum(r ** 2)

    best = None
    for s0 in (0.3, 1.0, 3.0):
        for c0 in (0.3, 1.0, 3.0):
            x0 = [np.log(s0), np.log(np.median(pts) * c0)]
            r = optimize.minimize(loss, x0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-10})
            if best is None or r.fun < best.fun:
                best = r
    return float(np.exp(best.x[0])), float(np.exp(best.x[1]))


def prior_from_gravity(delta_s, dist, sd_logshape=0.5, sd_logscale=0.5, kind="sf"):
    """SDMPrior centred on gamma decays fitted to ICIO elasticities delta_s (S,)."""
    fits = np.array([fit_decay_to_power_law(dl, dist, kind) for dl in np.asarray(delta_s, float)])
    S = len(fits)
    return SDMPrior(np.log(fits[:, 0]), np.full(S, sd_logshape), np.log(fits[:, 1]), np.full(S, sd_logscale))


def fit_decay_to_shares(shares, dist, mass_share=None, gamma=1.0, kind="sf", weights=None):
    """Least-squares fit of (shape, scale) so W matches observed origin shares.

    shares[r, p] = share of region p's (domestic) purchases of a commodity coming
    from r (columns sum to 1), e.g. from Statistics Canada interprovincial flows.
    Useful (i) for an informative prior and (ii) to validate the SDM's W.
    """
    sh = np.asarray(shares, float)
    wts = np.ones(sh.shape[1]) if weights is None else np.asarray(weights, float)

    def loss(p):
        W = weight_matrix(dist, np.exp(p[0]), np.exp(p[1]), mass_share, gamma, kind)
        return np.sum(wts[None, :] * (W - sh) ** 2)

    off = dist[~np.eye(len(dist), dtype=bool)]
    best = None
    for s0 in (0.5, 1.0, 2.0):
        for c0 in (0.2, 1.0, 5.0):
            r = optimize.minimize(loss, [np.log(s0), np.log(np.median(off) * c0)], method="Nelder-Mead")
            if best is None or r.fun < best.fun:
                best = r
    return float(np.exp(best.x[0])), float(np.exp(best.x[1])), float(best.fun)


# --------------------------------------------------------------------------- model
@dataclass
class SDMResult:
    draws: np.ndarray               # (chains, n_keep, S, P) on the transformed scale
    logpost: np.ndarray             # (chains, n_keep)
    accept: np.ndarray              # (chains, S)
    param_names: list
    sectors: list
    mode: np.ndarray | None = None
    model: "SpatialIRIO" = field(default=None, repr=False)

    def natural(self):
        """Draws on the natural scale: shape, scale[, gamma]."""
        out = self.draws.copy()
        out[..., 0] = np.exp(out[..., 0])
        out[..., 1] = np.exp(out[..., 1])
        if out.shape[-1] == 3:
            out[..., 2] = special.expit(out[..., 2])
        return out

    def rhat(self):
        """Split-R-hat (Gelman et al. 2013) per sector x parameter, natural scale."""
        x = self.natural()
        C, N = x.shape[:2]
        h = N // 2
        chains = np.concatenate([x[:, :h], x[:, h:2 * h]], axis=0)  # (2C, h, S, P)
        m = chains.shape[0]
        means = chains.mean(axis=1)
        B = h * means.var(axis=0, ddof=1)
        Wv = chains.var(axis=1, ddof=1).mean(axis=0)
        var_hat = (h - 1) / h * Wv + B / h
        return np.sqrt(var_hat / np.where(Wv > 0, Wv, np.nan)) if m > 1 else np.full(Wv.shape, np.nan)

    def summary(self):
        import pandas as pd
        x = self.natural().reshape(-1, *self.draws.shape[2:])
        rh = self.rhat()
        rows = []
        for s, name in enumerate(self.sectors):
            for p, pn in enumerate(self.param_names):
                v = x[:, s, p]
                rows.append(dict(sector=name, param=pn, mean=v.mean(), sd=v.std(),
                                 q05=np.quantile(v, 0.05), q50=np.median(v), q95=np.quantile(v, 0.95),
                                 rhat=rh[s, p]))
        return pd.DataFrame(rows)


class SpatialIRIO:
    """Bayesian multi-equation SDM of Torój with gamma distance decay."""

    def __init__(self, data: SDMData, prior: SDMPrior | None = None, kind: str = "sf",
                 supply_exponent="fixed1", n_eff: str = "R-1"):
        """
        supply_exponent : 'fixed1' (gamma_s = 1), 'fixed0' (no supply term), a float
                          (fixed value) or 'estimate' (gamma_s in (0,1), Beta prior).
        n_eff           : 'R-1' (residuals sum to zero over regions each year because the
                          national identity holds exactly) or 'R'.
        """
        self.d = data
        self.kind = kind
        if kind == "pdf" and np.any(np.diag(data.dist) <= 0):
            raise ValueError("kind='pdf' needs positive internal distances on the diagonal")
        self.prior = prior or SDMPrior.weak(data.S, data.dist)
        self.estimate_gamma = supply_exponent == "estimate"
        self.gamma_fixed = {"fixed1": 1.0, "fixed0": 0.0}.get(supply_exponent, supply_exponent)
        self.P = 3 if self.estimate_gamma else 2
        self.param_names = ["shape", "scale"] + (["gamma"] if self.estimate_gamma else [])
        T, S, R = data.T, data.S, data.R
        self.beta, self.beta0 = data.betas()
        self.q = np.einsum("tsv,tvr->tsr", self.beta, data.va) + self.beta0[:, :, None] * data.y
        self.mshare = data.mass / np.maximum(data.mass.sum(axis=2, keepdims=True), 1e-300)
        self.n_eff = T * (R - 1 if n_eff == "R-1" else R)
        self.sectors = data.sectors or [f"s{i}" for i in range(S)]

    # --- pieces of the posterior
    def _gamma(self, phi):
        return special.expit(phi[2]) if self.estimate_gamma else self.gamma_fixed

    def W(self, s, phi, t):
        return weight_matrix(self.d.dist, np.exp(phi[0]), np.exp(phi[1]),
                             self.mshare[t, s], self._gamma(phi), self.kind)

    def W_all(self, s, phi):
        return np.stack([self.W(s, phi, t) for t in range(self.d.T)])

    def ssr(self, s, Ws):
        e = self.d.va[:, s, :] - np.einsum("trp,tp->tr", Ws, self.q[:, s, :])
        return float(np.sum(e ** 2))

    def A(self, t, Wt):
        S, R = self.d.S, self.d.R
        M = self.beta[t][:, :, None, None] * Wt[:, None, :, :]   # (s,v,r,p)
        return np.eye(S * R) - M.transpose(0, 2, 1, 3).reshape(S * R, S * R)

    def log_post(self, phi):
        """Full log posterior (sigma integrated out), phi: (S, P)."""
        S = self.d.S
        Wt = [np.stack([self.W(s, phi[s], t) for s in range(S)]) for t in range(self.d.T)]
        ld = sum(np.linalg.slogdet(self.A(t, Wt[t]))[1] for t in range(self.d.T))
        ssr = sum(np.log(self.ssr(s, np.stack([Wt[t][s] for t in range(self.d.T)]))) for s in range(S))
        lp = sum(self.prior.logpdf(s, phi[s], self.estimate_gamma) for s in range(S))
        return ld - 0.5 * self.n_eff * ssr + lp

    # --- state with incremental determinant
    class _State:
        pass

    def _init_state(self, phi):
        st = self._State()
        S, T, R = self.d.S, self.d.T, self.d.R
        st.phi = phi.copy()
        st.W = np.stack([self.W_all(s, phi[s]) for s in range(S)])  # (S,T,R,R)
        st.Ainv, st.logdet = [], np.zeros(T)
        for t in range(T):
            A = self.A(t, st.W[:, t])
            st.Ainv.append(np.linalg.inv(A))
            st.logdet[t] = np.linalg.slogdet(A)[1]
        st.logssr = np.array([np.log(self.ssr(s, st.W[s])) for s in range(S)])
        st.lprior = np.array([self.prior.logpdf(s, phi[s], self.estimate_gamma) for s in range(S)])
        return st

    def _lp_state(self, st):
        return st.logdet.sum() - 0.5 * self.n_eff * st.logssr.sum() + st.lprior.sum()

    def _propose_sector(self, st, s, phi_new):
        """Return (delta_logpost, cache) for replacing sector s parameters."""
        R, T = self.d.R, self.d.T
        Wn = self.W_all(s, phi_new)
        lssr = np.log(self.ssr(s, Wn))
        lpr = self.prior.logpdf(s, phi_new, self.estimate_gamma)
        if not np.isfinite(lpr) or not np.isfinite(lssr):
            return -np.inf, None
        dld = np.zeros(T)
        cache = []
        sl = slice(s * R, (s + 1) * R)
        for t in range(T):
            dW = Wn[t] - st.W[s, t]
            Delta = -(self.beta[t, s][:, None, None] * dW[None, :, :]).transpose(1, 0, 2).reshape(R, -1)
            AinvE = st.Ainv[t][:, sl]                     # (RS, R)
            K = np.eye(R) + Delta @ AinvE
            sign, ldK = np.linalg.slogdet(K)
            if sign == 0:
                return -np.inf, None
            dld[t] = ldK
            cache.append((Delta, AinvE, K))
        dlp = dld.sum() - 0.5 * self.n_eff * (lssr - st.logssr[s]) + (lpr - st.lprior[s])
        return dlp, (Wn, lssr, lpr, dld, cache)

    def _accept(self, st, s, phi_new, payload):
        Wn, lssr, lpr, dld, cache = payload
        for t, (Delta, AinvE, K) in enumerate(cache):
            # Woodbury: (A + E Delta)^-1 = Ainv - Ainv E K^-1 Delta Ainv
            st.Ainv[t] = st.Ainv[t] - AinvE @ np.linalg.solve(K, Delta @ st.Ainv[t])
        st.logdet += dld
        st.W[s] = Wn
        st.logssr[s] = lssr
        st.lprior[s] = lpr
        st.phi[s] = phi_new

    def _refresh(self, st):
        for t in range(self.d.T):
            A = self.A(t, st.W[:, t])
            st.Ainv[t] = np.linalg.inv(A)
            st.logdet[t] = np.linalg.slogdet(A)[1]

    def initial_phi(self):
        phi = np.column_stack([self.prior.mu_logshape, self.prior.mu_logscale])
        if self.estimate_gamma:
            phi = np.column_stack([phi, np.zeros(self.d.S)])
        return phi

    # --- posterior mode (block coordinate ascent); with a flat prior this is the MLE
    def fit_mode(self, phi0=None, sweeps: int = 5, tol: float = 1e-6, verbose: bool = False):
        st = self._init_state(self.initial_phi() if phi0 is None else np.asarray(phi0, float))
        last = self._lp_state(st)
        for sw in range(sweeps):
            for s in range(self.d.S):
                base = st.phi[s].copy()

                def f(x):
                    dl, _ = self._propose_sector(st, s, x)
                    return -dl if np.isfinite(dl) else 1e300

                r = optimize.minimize(f, base, method="Nelder-Mead",
                                      options={"xatol": 1e-5, "fatol": 1e-8, "maxiter": 400})
                dl, payload = self._propose_sector(st, s, r.x)
                if payload is not None and dl > 0:
                    self._accept(st, s, r.x, payload)
            self._refresh(st)
            cur = self._lp_state(st)
            if verbose:
                print(f"sweep {sw}: log posterior {cur:.4f}")
            if abs(cur - last) < tol:
                break
            last = cur
        self.mode_ = st.phi.copy()
        return self.mode_

    # --- MCMC
    def sample(self, n_iter: int = 4000, burn: int = 2000, thin: int = 2, chains: int = 4,
               init=None, jitter: float = 0.3, seed: int = 0, adapt_start: int = 200,
               refresh_every: int = 50, verbose: bool = False) -> SDMResult:
        """Adaptive Metropolis-within-Gibbs; one Gaussian RW block per sector."""
        rng = np.random.default_rng(seed)
        S, P = self.d.S, self.P
        base = self.initial_phi() if init is None else np.asarray(init, float)
        n_keep = (n_iter - burn) // thin
        draws = np.zeros((chains, n_keep, S, P))
        lps = np.zeros((chains, n_keep))
        acc = np.zeros((chains, S))
        for c in range(chains):
            st = self._init_state(base + jitter * rng.standard_normal(base.shape) * (c > 0 or init is None))
            hist = [[] for _ in range(S)]
            prop_cov = [np.eye(P) * 0.1 ** 2 for _ in range(S)]
            k = 0
            for it in range(n_iter):
                for s in range(S):
                    if it >= adapt_start and it < burn and it % 50 == 0 and len(hist[s]) > 50:
                        H = np.asarray(hist[s][-1000:])
                        prop_cov[s] = (2.38 ** 2 / P) * np.cov(H.T) + 1e-6 * np.eye(P)
                    prop = st.phi[s] + rng.multivariate_normal(np.zeros(P), prop_cov[s])
                    dl, payload = self._propose_sector(st, s, prop)
                    if np.log(rng.uniform()) < dl:
                        self._accept(st, s, prop, payload)
                        if it >= burn:
                            acc[c, s] += 1
                    hist[s].append(st.phi[s].copy())
                if (it + 1) % refresh_every == 0:
                    self._refresh(st)
                if it >= burn and (it - burn) % thin == 0 and k < n_keep:
                    draws[c, k] = st.phi
                    lps[c, k] = self._lp_state(st)
                    k += 1
                if verbose and (it + 1) % 500 == 0:
                    print(f"chain {c} it {it + 1}: lp={self._lp_state(st):.3f}")
        acc /= max(n_iter - burn, 1)
        res = SDMResult(draws, lps, acc, self.param_names, self.sectors,
                        mode=getattr(self, "mode_", None), model=self)
        return res

    def sigma2_draws(self, phi, rng=None):
        """Draw sigma_s^2 | phi, data ~ Inv-Gamma(n_eff/2, SSR_s/2)."""
        rng = rng or np.random.default_rng()
        ssr = np.array([self.ssr(s, self.W_all(s, phi[s])) for s in range(self.d.S)])
        return ssr / 2 / rng.gamma(self.n_eff / 2, size=self.d.S)

    # --- flows
    def column_totals_toroj(self, t):
        """colZ[s,p,v] = Z^{sv} * va^v_p / VA^v (Torój 2016 eq. 4)."""
        va = self.d.va[t]
        share = va / np.maximum(va.sum(axis=1, keepdims=True), 1e-300)   # (v,p)
        return self.d.Znat[t][:, None, :] * share.T[None, :, :]          # (s,p,v)

    def flows(self, phi, t=-1, col_totals=None):
        """Z[r, s, p, v] for one parameter vector phi (S,P)."""
        t = t % self.d.T
        colZ = self.column_totals_toroj(t) if col_totals is None else np.asarray(col_totals, float)
        Ws = np.stack([self.W(s, phi[s], t) for s in range(self.d.S)])   # (s,r,p)
        return np.einsum("spv,srp->rspv", colZ, Ws)

    def posterior_flows(self, res: SDMResult, t=-1, col_totals=None, n_draws: int = 200,
                        quantiles=(0.05, 0.95), seed: int = 0, keep_draws: bool = False):
        """Posterior mean / sd (and optional quantiles over stored draws) of Z[r,s,p,v]."""
        rng = np.random.default_rng(seed)
        flat = res.draws.reshape(-1, *res.draws.shape[2:])
        pick = rng.choice(len(flat), size=min(n_draws, len(flat)), replace=False)
        mean = None
        M2 = None
        store = [] if keep_draws or quantiles else None
        for n, i in enumerate(pick, 1):
            Z = self.flows(flat[i], t, col_totals)
            if mean is None:
                mean, M2 = np.zeros_like(Z), np.zeros_like(Z)
            dlt = Z - mean
            mean += dlt / n
            M2 += dlt * (Z - mean)
            if store is not None:
                store.append(Z)
        out = {"mean": mean, "sd": np.sqrt(M2 / max(len(pick) - 1, 1))}
        if store is not None:
            arr = np.stack(store)
            if quantiles:
                out["quantiles"] = {q: np.quantile(arr, q, axis=0) for q in quantiles}
            if keep_draws:
                out["draws"] = arr
        return out


def origin_shares_from_flows(T_irb):
    """Observed origin shares sh[j, r, p] = T[r,j,p] / sum_r T[r,j,p] (columns sum to 1)."""
    T = np.asarray(T_irb, float)
    tot = T.sum(axis=0, keepdims=True)
    sh = np.divide(T, tot, out=np.zeros_like(T), where=tot > 0)
    return sh.transpose(1, 0, 2)


def balance_flows(Z4, row_totals, col_totals, **ras_kw):
    """RAS-balance Z[r,s,p,v] to row totals (r,s) and column totals (p,v).

    Torój (2025, Sect. 5) notes the initial spatial-IRIO Z "is subject to balancing";
    typical targets are regional intermediate sales (row) and intermediate purchases
    (column). Returns (Z_balanced, info).
    """
    from .ras import ras
    R, S = Z4.shape[:2]
    X, info = ras(Z4.reshape(R * S, R * S), np.ravel(row_totals), np.ravel(col_totals), **ras_kw)
    return X.reshape(R, S, R, S), info
