"""Spatial IRIO: Bayesian multi-equation spatial Durbin model (Torój 2016, 2021, 2024, 2025).

Model (Torój 2024, eqs. 1-6; Torój 2016 eqs. 7-11, 19-21)
---------------------------------------------------------
For each commodity/sector i, regions r, years t, with v^i_t (R x 1) regional value added:

  (3)  v^i_t = W_{i,t} ( sum_j beta_{i,j} v^j_t + beta_{i,f} f^i_t ) [+ beta_{i,f} f^{i,dir}_t] + eps^i_t

  beta_{i,j} = (v_i / x_i) a_{i,j} (x_j / v_j),  beta_{i,f} = v_i / x_i   (national ratios, known)

  f^{i,dir}: optional final demand that is supplied *by the region itself* (e.g. international
  exports by province of origin) -- an extension of this project, off unless provided.

Spatial weights, Torój (2024) eq. (5) combined with the 2016 supply term (eq. 20):

  w_i^{r,s} ∝ [1 - P_Gamma( d_rs^(1-gamma_i) * dtilde_rs^gamma_i ; shape_i, scale_i )] * (m^i_r)^psi_i

  * 1 - P_Gamma: one minus the gamma CDF (Torój 2024 Sect. 2; the "most up-to-date" form).
  * gamma_i in (0,1): weight on the second distance measure dtilde (driving time, rescaled to
    the mean of d). Used only if ``dist_time`` is given (prior U(0,1), Torój 2024 eq. 10).
  * psi_i in [0,1]: supply-side exponent on the region's share m^i_r of sector-i activity
    (Torój 2016 eq. 20; psi = 0 drops it, which is the 2024 specification). Estimated with a
    U(0,1) prior by default, following the 2016 paper's joint estimation.
  * columns normalised to 1 (share of region s's demand for i supplied by each r).

Stacked (eq. 4): A_t v_t = B_t f_t + eps_t,  A_t = I_SR - [beta_{i,j} W_i]_{i,j}
Errors: eps ~ N(0, Sigma ⊗ I_R), Sigma S x S with cross-sector correlation (2024 Sect. 2).

Likelihood (eq. 6, LeSage & Pace 2009):
  ln p = sum_t ln|A_t| - (n/2) ln|Sigma| - 1/2 tr(Sigma^-1 E'E) + const,
  E = (n x S) residual matrix, n = number of region-year rows.

Covariance treatment
  'iw'   (default; Torój 2024 eq. 9): Sigma ~ inv-Wishart(df = S, H), H_ii = 0.4 Var_r(v_i),
         prior correlations 0.3. Sigma is integrated out analytically:
             ln p(theta | data) = sum_t ln|A_t| - ((df + n)/2) ln|H + E'E| + ln p(theta)
         (Torój samples Sigma by random-walk MH; integrating it out targets the same
         marginal posterior of theta and mixes better.)
  'diag' : diagonal Sigma with Jeffreys priors: - (n/2) sum_i ln SSR_i.
  n : T*R ('R', as in Torój) or T*(R-1) ('R-1', sensitivity: residuals sum to zero over
      regions each year because eq. (3) aggregates to an exact national identity).

Priors on (shape, scale)
  'toroj'  (2024 Sect. 3): shape ~ U(a_min, a_max); scale | shape ~ TN(mu_i(shape),
           sigma_i(shape), Tmin(shape), Tmax(shape)) where mu_i(shape) is the scale at which
           the share of the decay profile within r_home km equals the observed home-region
           share of sector i's intermediate supply (Torój: Finnish tables; here: Statistics
           Canada's interprovincial flows), sigma from the scale matching mean + sd, and
           Tmin/Tmax are the scales beyond which the profile is numerically insensitive.
  'gravity': log-normal around gamma decays fitted to ICIO/IPT distance elasticities.
  'weak'   : log-normal, log shape ~ N(0,1), log scale ~ N(log median distance, 1.5).

Sampler: Metropolis-within-Gibbs, one block per sector (adaptive Gaussian RW on
transformed parameters; Torój's fixed proposal variances available). One sector's update
changes R rows of A_t: ln|A_t| via the matrix determinant lemma, A_t^-1 via Woodbury.

Flows (2024 eqs. 1-2): Z[r,i,s,j] = colZ[i,s,j] * w_i^{r,s}, colZ default = z_ij v^s_j / v_j
(or provincial use-table column totals), then the extended RAS of Sect. 2 (rows = regional
supply potential, columns, R x R block sums = national z_ij): ``balance_three_way``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, special, stats


# ============================================================================ data
@dataclass
class SDMData:
    """Inputs (T years, S sectors, R regions).

    va       : (T, S, R) regional value added
    y        : (T, S, R) regional final demand for commodity i located in region r
               (domestic-origin; from provincial SUTs, see ``final_demand``)
    Znat     : (T, S, S) national (domestic) intermediate flows z_ij
    Xnat     : (T, S) national gross output
    dist     : (R, R) distances (diagonal = 0 or internal distance)
    dist_time: (R, R) optional second measure (driving time), rescaled to dist's mean
    mass     : (T, S, R) supply mass for the 2016 supply term (default va; pass employment
               or lagged va to avoid the endogeneity of using the dependent variable)
    y_direct : (T, S, R) optional final demand supplied directly by region r (exports)
    """
    va: np.ndarray
    y: np.ndarray
    Znat: np.ndarray
    Xnat: np.ndarray
    dist: np.ndarray
    mass: np.ndarray | None = None
    sectors: list | None = None
    regions: list | None = None
    dist_time: np.ndarray | None = None
    y_direct: np.ndarray | None = None

    def __post_init__(self):
        self.va = np.asarray(self.va, float)
        T, S, R = self.va.shape
        self.y = np.asarray(self.y, float).reshape(T, S, R)
        self.Znat = np.asarray(self.Znat, float).reshape(T, S, S)
        self.Xnat = np.asarray(self.Xnat, float).reshape(T, S)
        self.dist = np.asarray(self.dist, float)
        self.mass = self.va.copy() if self.mass is None else np.asarray(self.mass, float)
        self.y_direct = np.zeros_like(self.y) if self.y_direct is None else np.asarray(self.y_direct, float)
        if self.dist_time is not None:
            dt = np.asarray(self.dist_time, float)
            off = ~np.eye(R, dtype=bool)
            self.dist_time = dt * (self.dist[off].mean() / dt[off].mean())
        self.T, self.S, self.R = T, S, R

    def betas(self):
        VA = self.va.sum(axis=2)
        ratio = np.divide(VA, self.Xnat, out=np.zeros_like(VA), where=self.Xnat > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = ratio[:, :, None] * self.Znat / np.where(VA > 0, VA, np.inf)[:, None, :]
        return np.nan_to_num(beta), ratio

    def national_identity_gap(self):
        """max_i |sum_j beta VA_j + beta_f F_i - VA_i| / VA_i (0 if y sums to national f)."""
        beta, beta0 = self.betas()
        VA = self.va.sum(axis=2)
        F = (self.y + self.y_direct).sum(axis=2)
        rhs = np.einsum("tsv,tv->ts", beta, VA) + beta0 * F
        return float(np.max(np.abs(rhs - VA) / np.maximum(np.abs(VA), 1e-12)))


# ======================================================================= weights
def log_decay(d, shape, scale, kind: str = "cdf"):
    """log of the decay profile. 'cdf' (= 'sf'): log(1 - P_Gamma(d)) (Torój 2016/2024);
    'pdf': log gamma density (sensitivity only)."""
    x = np.asarray(d, float) / scale
    if kind in ("cdf", "sf"):
        with np.errstate(divide="ignore"):
            return np.log(special.gammaincc(shape, x))
    if kind == "pdf":
        with np.errstate(divide="ignore"):
            return special.xlogy(shape - 1, x) - x - special.gammaln(shape) - np.log(scale)
    raise ValueError(kind)


def decay(d, shape, scale, kind: str = "cdf"):
    return np.exp(log_decay(d, shape, scale, kind))


def mixed_distance(dist, dist_time=None, gamma=0.0):
    if dist_time is None or gamma == 0:
        return dist
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((dist > 0) & (dist_time > 0), dist ** (1 - gamma) * dist_time ** gamma, 0.0)


def weight_matrix(dist, shape, scale, mass_share=None, psi=1.0, kind: str = "cdf",
                  dist_time=None, gamma=0.0):
    """Column-normalised W(r, p) (Torój 2024 eq. 5 x 2016 eq. 20)."""
    lg = log_decay(mixed_distance(dist, dist_time, gamma), shape, scale, kind)
    if mass_share is not None and psi != 0:
        with np.errstate(divide="ignore"):
            lg = lg + psi * np.log(np.asarray(mass_share))[:, None]
    lg = lg - np.max(np.where(np.isfinite(lg), lg, -np.inf), axis=0, keepdims=True)
    w = np.nan_to_num(np.exp(lg))
    cs = w.sum(axis=0, keepdims=True)
    return np.divide(w, cs, out=np.full_like(w, 1.0 / w.shape[0]), where=cs > 0)


# ======================================================================== priors
class LogNormalPrior:
    """log shape ~ N(mu_a, sd_a), log scale ~ N(mu_c, sd_c), independent per sector."""

    def __init__(self, mu_logshape, sd_logshape, mu_logscale, sd_logscale):
        self.mu_a, self.sd_a = np.asarray(mu_logshape, float), np.asarray(sd_logshape, float)
        self.mu_c, self.sd_c = np.asarray(mu_logscale, float), np.asarray(sd_logscale, float)

    @classmethod
    def weak(cls, S, dist, sd_logshape=1.0, sd_logscale=1.5):
        off = dist[~np.eye(len(dist), dtype=bool)]
        med = float(np.median(off[off > 0]))
        return cls(np.zeros(S), np.full(S, sd_logshape), np.full(S, np.log(med)), np.full(S, sd_logscale))

    def logpdf(self, s, la, lc):
        return (stats.norm.logpdf(la, self.mu_a[s], self.sd_a[s])
                + stats.norm.logpdf(lc, self.mu_c[s], self.sd_c[s]))

    def initial(self, s):
        return self.mu_a[s], self.mu_c[s]


SDMPrior = LogNormalPrior  # backwards-compatible name


def home_share_profile(shape, scale, r_home, d_max, step=1.0, kind="cdf"):
    """Share of the decay profile (on a 0..d_max grid) lying within r_home (Torój 2024 Sect. 3)."""
    grid = np.arange(0.0, d_max + step, step)
    g = decay(grid, shape, scale, kind)
    tot = g.sum()
    return float(g[grid < r_home].sum() / tot) if tot > 0 else np.nan


class TorojPrior:
    """Torój (2024) eqs. (7)-(8): shape ~ U(a_min, a_max); scale | shape ~ TN(mu(shape), sigma(shape),
    Tmin(shape), Tmax(shape)), calibrated to observed home-region supply shares.

    home_mean, home_sd : (S,) mean and s.d. across regions of the share of sector i's
                         (intermediate) supply delivered to the home region
    r_home             : radius (km) interpreted as within-region supply (Torój: 51.2 km, computed
                         for the 73 Polish NUTS-3 regions he estimates on -- the Finnish tables
                         supply only the home shares; ``canada_r_home`` gives the Canadian analogue)
    d_max              : maximum distance of the grid (Torój: 800 km for Poland)
    The functions of shape are evaluated on a log grid and interpolated.
    """

    def __init__(self, home_mean, home_sd, r_home, d_max, a_min=0.01, a_max=10.0, n_grid=60,
                 step=None, kind="cdf", eps=1e-4):
        self.hm, self.hs = np.asarray(home_mean, float), np.asarray(home_sd, float)
        self.r_home, self.d_max, self.kind = float(r_home), float(d_max), kind
        self.a_min, self.a_max = a_min, a_max
        self.step = step or max(1.0, d_max / 2000)
        self.lgrid = np.linspace(np.log(a_min), np.log(a_max), n_grid)
        flat = self._flat_share()
        S = len(self.hm)
        self.mu = np.zeros((S, n_grid)); self.sig = np.zeros((S, n_grid))
        self.tmin = np.zeros(n_grid); self.tmax = np.zeros(n_grid)
        for g, la in enumerate(self.lgrid):
            a = np.exp(la)
            self.tmin[g] = self._scale_for(a, 1 - eps)
            self.tmax[g] = self._scale_for(a, flat * (1 + eps) + eps)
            for s in range(S):
                c_mu = self._scale_for(a, np.clip(self.hm[s], flat + 2 * eps, 1 - 2 * eps))
                c_up = self._scale_for(a, np.clip(self.hm[s] + self.hs[s], flat + 2 * eps, 1 - 2 * eps))
                self.mu[s, g] = c_mu
                self.sig[s, g] = max(c_mu - c_up, 1e-3 * c_mu)

    def _flat_share(self):
        grid = np.arange(0.0, self.d_max + self.step, self.step)
        return float((grid < self.r_home).mean())

    def _scale_for(self, a, target):
        f = lambda lc: home_share_profile(a, np.exp(lc), self.r_home, self.d_max, self.step, self.kind) - target
        lo, hi = np.log(1e-6), np.log(1e7)
        flo, fhi = f(lo), f(hi)
        if not (np.isfinite(flo) and np.isfinite(fhi)) or flo * fhi > 0:
            return np.exp(lo if abs(flo) < abs(fhi) else hi)
        return float(np.exp(optimize.brentq(f, lo, hi, xtol=1e-6)))

    def _interp(self, arr, la):
        return np.interp(la, self.lgrid, arr)

    def logpdf(self, s, la, lc):
        if not (self.lgrid[0] <= la <= self.lgrid[-1]):
            return -np.inf
        c = np.exp(lc)
        lo, hi = self._interp(self.tmin, la), self._interp(self.tmax, la)
        if not lo <= c <= hi:
            return -np.inf
        mu, sd = self._interp(self.mu[s], la), self._interp(self.sig[s], la)
        lp_c = stats.truncnorm.logpdf(c, (lo - mu) / sd, (hi - mu) / sd, loc=mu, scale=sd)
        lp_a = -np.log(self.a_max - self.a_min)
        return lp_a + lp_c + la + lc  # Jacobians of the log transforms

    def initial(self, s):
        """Prior means (Torój starts the chains there): shape mid-range in logs, scale mu(shape)."""
        la = 0.5 * (self.lgrid[0] + self.lgrid[-1])
        return la, float(np.log(self._interp(self.mu[s], la)))


def home_share_stats(T_rib):
    """Mean and s.d. across origin provinces of the home share T[r,i,r] / sum_b T[r,i,b]."""
    T = np.asarray(T_rib, float)
    R = T.shape[0]
    own = np.stack([T[r, :, r] for r in range(R)])           # (R, S)
    tot = T.sum(axis=2)                                      # (R, S)
    sh = np.divide(own, tot, out=np.full_like(own, np.nan), where=tot > 0)
    return np.nanmean(sh, axis=0), np.nanstd(sh, axis=0)


def fit_decay_to_power_law(delta, dist, kind="cdf", d_range=None):
    """Gamma (shape, scale) whose 1-CDF profile best matches d^-delta over the distances."""
    d = np.asarray(dist, float)
    pts = d[~np.eye(len(d), dtype=bool)] if d_range is None else np.asarray(d_range, float)
    pts = np.unique(pts[pts > 0])
    target = -delta * np.log(pts)

    def loss(p):
        lg = log_decay(pts, np.exp(p[0]), np.exp(p[1]), kind)
        if not np.all(np.isfinite(lg)):
            return 1e12
        r = (lg - lg.mean()) - (target - target.mean())
        return np.sum(r ** 2)

    best = None
    for s0 in (0.05, 0.3, 1.0, 3.0):
        for c0 in (0.3, 1.0, 3.0):
            r = optimize.minimize(loss, [np.log(s0), np.log(np.median(pts) * c0)], method="Nelder-Mead",
                                  options={"xatol": 1e-6, "fatol": 1e-10})
            if best is None or r.fun < best.fun:
                best = r
    return float(np.exp(best.x[0])), float(np.exp(best.x[1]))


def prior_from_gravity(delta_s, dist, sd_logshape=0.5, sd_logscale=0.5, kind="cdf"):
    fits = np.array([fit_decay_to_power_law(dl, dist, kind) for dl in np.asarray(delta_s, float)])
    S = len(fits)
    return LogNormalPrior(np.log(fits[:, 0]), np.full(S, sd_logshape), np.log(fits[:, 1]), np.full(S, sd_logscale))


def fit_decay_to_shares(shares, dist, mass_share=None, psi=1.0, kind="cdf", weights=None):
    """Least-squares (shape, scale) so W matches observed origin shares (validation/prior)."""
    sh = np.asarray(shares, float)
    wts = np.ones(sh.shape[1]) if weights is None else np.asarray(weights, float)

    def loss(p):
        W = weight_matrix(dist, np.exp(p[0]), np.exp(p[1]), mass_share, psi, kind)
        return np.sum(wts[None, :] * (W - sh) ** 2)

    off = dist[~np.eye(len(dist), dtype=bool)]
    best = None
    for s0 in (0.05, 0.5, 1.0, 2.0):
        for c0 in (0.2, 1.0, 5.0):
            r = optimize.minimize(loss, [np.log(s0), np.log(np.median(off) * c0)], method="Nelder-Mead")
            if best is None or r.fun < best.fun:
                best = r
    return float(np.exp(best.x[0])), float(np.exp(best.x[1])), float(best.fun)


# ======================================================================= results
@dataclass
class SDMResult:
    draws: np.ndarray               # (chains, n_keep, S, P) transformed scale
    logpost: np.ndarray
    accept: np.ndarray
    param_names: list
    sectors: list
    mode: np.ndarray | None = None
    model: "SpatialIRIO" = field(default=None, repr=False)

    def natural(self):
        out = self.draws.copy()
        for p, n in enumerate(self.param_names):
            out[..., p] = np.exp(out[..., p]) if n in ("shape", "scale") else special.expit(out[..., p])
        return out

    def rhat(self):
        x = self.natural()
        C, N = x.shape[:2]
        h = N // 2
        ch = np.concatenate([x[:, :h], x[:, h:2 * h]], axis=0)
        if ch.shape[0] < 2:
            return np.full(x.shape[2:], np.nan)
        B = h * ch.mean(axis=1).var(axis=0, ddof=1)
        Wv = ch.var(axis=1, ddof=1).mean(axis=0)
        return np.sqrt(((h - 1) / h * Wv + B / h) / np.where(Wv > 0, Wv, np.nan))

    def shape_scale_corr(self):
        """Pearson corr of log shape and log scale per sector. Torój (2025): a value near -1
        signals that the decay profile is identified even if R-hat for shape/scale is not."""
        d = self.draws.reshape(-1, *self.draws.shape[2:])
        return np.array([np.corrcoef(d[:, s, 0], d[:, s, 1])[0, 1] for s in range(d.shape[1])])

    def summary(self):
        import pandas as pd
        x = self.natural().reshape(-1, *self.draws.shape[2:])
        rh, cc = self.rhat(), self.shape_scale_corr()
        rows = []
        for s, name in enumerate(self.sectors):
            for p, pn in enumerate(self.param_names):
                v = x[:, s, p]
                rows.append(dict(sector=name, param=pn, mean=v.mean(), sd=v.std(), q05=np.quantile(v, .05),
                                 q50=np.median(v), q95=np.quantile(v, .95), rhat=rh[s, p],
                                 corr_logshape_logscale=cc[s]))
        return pd.DataFrame(rows)


# ========================================================================= model
class SpatialIRIO:
    def __init__(self, data: SDMData, prior=None, kind: str = "cdf", supply_exponent="estimate",
                 covariance: str = "iw", n_eff: str = "R", iw_scale: float = 0.4, iw_corr: float = 0.3,
                 iw_df=None):
        """
        supply_exponent : 'estimate' (psi_i ~ U(0,1), Torój 2016), 'fixed0' (no supply term,
                          Torój 2024), 'fixed1', or a float.
        covariance      : 'iw' (full Sigma, inverse-Wishart, Torój 2024) or 'diag'.
        n_eff           : 'R' (n = T R, Torój) or 'R-1' (sensitivity).
        """
        self.d = data
        self.kind = "cdf" if kind == "sf" else kind
        if self.kind == "pdf" and np.any(np.diag(data.dist) <= 0):
            raise ValueError("kind='pdf' needs positive internal distances on the diagonal")
        T, S, R = data.T, data.S, data.R
        self.prior = prior or LogNormalPrior.weak(S, data.dist)
        self.est_psi = supply_exponent == "estimate"
        self.psi_fixed = {"fixed1": 1.0, "fixed0": 0.0}.get(supply_exponent, supply_exponent)
        self.est_gamma = data.dist_time is not None
        self.param_names = ["shape", "scale"] + (["gamma"] if self.est_gamma else []) + (["psi"] if self.est_psi else [])
        self.P = len(self.param_names)
        self._ig = 2 if self.est_gamma else None
        self._ipsi = (3 if self.est_gamma else 2) if self.est_psi else None
        self.beta, self.beta0 = data.betas()
        self.q = np.einsum("tsv,tvr->tsr", self.beta, data.va) + self.beta0[:, :, None] * data.y
        self.vdir = self.beta0[:, :, None] * data.y_direct
        self.mshare = data.mass / np.maximum(data.mass.sum(axis=2, keepdims=True), 1e-300)
        self.n = T * (R - 1 if n_eff == "R-1" else R)
        self.cov = covariance
        if covariance == "iw":
            v = data.va.transpose(0, 2, 1).reshape(T * R, S)
            sd = np.sqrt(iw_scale * v.var(axis=0))
            self.H = iw_corr * np.outer(sd, sd)
            np.fill_diagonal(self.H, sd ** 2)
            self.df = S if iw_df is None else iw_df
        self.sectors = data.sectors or [f"s{i}" for i in range(S)]

    # ---------------------------------------------------------------- pieces
    def _gamma(self, phi):
        return special.expit(phi[self._ig]) if self.est_gamma else 0.0

    def _psi(self, phi):
        return special.expit(phi[self._ipsi]) if self.est_psi else self.psi_fixed

    def W(self, s, phi, t):
        return weight_matrix(self.d.dist, np.exp(phi[0]), np.exp(phi[1]), self.mshare[t, s], self._psi(phi),
                             self.kind, self.d.dist_time, self._gamma(phi))

    def W_all(self, s, phi):
        return np.stack([self.W(s, phi, t) for t in range(self.d.T)])

    def resid(self, s, Ws):
        """Residual column for sector s, stacked over (t, r)."""
        e = self.d.va[:, s, :] - np.einsum("trp,tp->tr", Ws, self.q[:, s, :]) - self.vdir[:, s, :]
        return e.reshape(-1)

    def A(self, t, Wt):
        S, R = self.d.S, self.d.R
        M = self.beta[t][:, :, None, None] * Wt[:, None, :, :]
        return np.eye(S * R) - M.transpose(0, 2, 1, 3).reshape(S * R, S * R)

    def log_prior(self, s, phi):
        lp = self.prior.logpdf(s, phi[0], phi[1])
        for idx in (self._ig, self._ipsi):
            if idx is not None:  # U(0,1) on the natural scale + logit Jacobian
                g = special.expit(phi[idx])
                lp += np.log(g) + np.log1p(-g)
        return lp

    def _cov_term(self, G):
        if self.cov == "iw":
            sign, ld = np.linalg.slogdet(self.H + G)
            return -0.5 * (self.df + self.n) * ld
        return -0.5 * self.n * np.sum(np.log(np.diag(G)))

    def log_post(self, phi):
        S, T = self.d.S, self.d.T
        Wall = [self.W_all(s, phi[s]) for s in range(S)]
        ld = sum(np.linalg.slogdet(self.A(t, np.stack([Wall[s][t] for s in range(S)])))[1] for t in range(T))
        E = np.column_stack([self.resid(s, Wall[s]) for s in range(S)])
        return ld + self._cov_term(E.T @ E) + sum(self.log_prior(s, phi[s]) for s in range(S))

    # ---------------------------------------------------- incremental state
    class _State:
        pass

    def _init_state(self, phi):
        st = self._State()
        S, T = self.d.S, self.d.T
        st.phi = np.array(phi, float).copy()
        st.W = np.stack([self.W_all(s, st.phi[s]) for s in range(S)])
        st.Ainv, st.logdet = [], np.zeros(T)
        for t in range(T):
            A = self.A(t, st.W[:, t])
            st.Ainv.append(np.linalg.inv(A))
            st.logdet[t] = np.linalg.slogdet(A)[1]
        st.E = np.column_stack([self.resid(s, st.W[s]) for s in range(S)])
        st.G = st.E.T @ st.E
        st.covterm = self._cov_term(st.G)
        st.lprior = np.array([self.log_prior(s, st.phi[s]) for s in range(S)])
        return st

    def _lp_state(self, st):
        return st.logdet.sum() + st.covterm + st.lprior.sum()

    def _propose_sector(self, st, s, phi_new):
        R, T = self.d.R, self.d.T
        lpr = self.log_prior(s, phi_new)
        if not np.isfinite(lpr):
            return -np.inf, None
        Wn = self.W_all(s, phi_new)
        e = self.resid(s, Wn)
        G = st.G.copy()
        g = st.E.T @ e
        g[s] = e @ e
        G[s, :] = g
        G[:, s] = g
        cov = self._cov_term(G)
        if not np.isfinite(cov):
            return -np.inf, None
        dld = np.zeros(T)
        cache = []
        sl = slice(s * R, (s + 1) * R)
        for t in range(T):
            dW = Wn[t] - st.W[s, t]
            Delta = -(self.beta[t, s][:, None, None] * dW[None]).transpose(1, 0, 2).reshape(R, -1)
            AinvE = st.Ainv[t][:, sl]
            K = np.eye(R) + Delta @ AinvE
            sign, ldK = np.linalg.slogdet(K)
            if sign == 0:
                return -np.inf, None
            dld[t] = ldK
            cache.append((Delta, AinvE, K))
        dlp = dld.sum() + (cov - st.covterm) + (lpr - st.lprior[s])
        return dlp, (Wn, e, G, cov, lpr, dld, cache)

    def _accept(self, st, s, phi_new, payload):
        Wn, e, G, cov, lpr, dld, cache = payload
        for t, (Delta, AinvE, K) in enumerate(cache):
            st.Ainv[t] = st.Ainv[t] - AinvE @ np.linalg.solve(K, Delta @ st.Ainv[t])
        st.logdet += dld
        st.W[s] = Wn
        st.E[:, s] = e
        st.G = G
        st.covterm = cov
        st.lprior[s] = lpr
        st.phi[s] = phi_new

    def _refresh(self, st):
        for t in range(self.d.T):
            A = self.A(t, st.W[:, t])
            st.Ainv[t] = np.linalg.inv(A)
            st.logdet[t] = np.linalg.slogdet(A)[1]

    def initial_phi(self):
        rows = []
        for s in range(self.d.S):
            la, lc = self.prior.initial(s)
            rows.append([la, lc] + [0.0] * (self.P - 2))
        return np.array(rows, float)

    # ------------------------------------------------------------- mode
    def fit_mode(self, phi0=None, sweeps: int = 8, tol: float = 1e-6, verbose: bool = False):
        st = self._init_state(self.initial_phi() if phi0 is None else phi0)
        last = self._lp_state(st)
        for sw in range(sweeps):
            for s in range(self.d.S):
                def f(x):
                    dl, _ = self._propose_sector(st, s, x)
                    return -dl if np.isfinite(dl) else 1e300
                r = optimize.minimize(f, st.phi[s].copy(), method="Nelder-Mead",
                                      options={"xatol": 1e-5, "fatol": 1e-8, "maxiter": 200 * self.P})
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

    # ------------------------------------------------------------- MCMC
    TOROJ_PROPOSAL_VAR = {"shape": 5e-5, "scale": 25e-5, "gamma": 10e-5, "psi": 10e-5}

    def sample(self, n_iter: int = 4000, burn: int = 2000, thin: int = 2, chains: int = 4, init=None,
               jitter: float = 0.3, seed: int = 0, adapt: bool = True, proposal: str = "adaptive",
               adapt_start: int = 200, refresh_every: int = 50, verbose: bool = False) -> SDMResult:
        """proposal='adaptive' (Haario-type, adapted during burn-in only) or 'toroj' (fixed
        diagonal variances of Torój 2024 fn. 3 on the transformed parameters)."""
        rng = np.random.default_rng(seed)
        S, P = self.d.S, self.P
        base = self.initial_phi() if init is None else np.asarray(init, float)
        n_keep = (n_iter - burn) // thin
        draws = np.zeros((chains, n_keep, S, P))
        lps = np.zeros((chains, n_keep))
        acc = np.zeros((chains, S))
        v0 = np.diag([self.TOROJ_PROPOSAL_VAR[n] for n in self.param_names]) if proposal == "toroj" \
            else np.eye(P) * 0.1 ** 2
        for c in range(chains):
            start = base + (jitter * rng.standard_normal(base.shape) if c > 0 else 0)
            # keep starting points inside the prior support
            for s in range(S):
                if not np.isfinite(self.log_prior(s, start[s])):
                    start[s] = base[s]
            st = self._init_state(start)
            hist = [[] for _ in range(S)]
            cov = [v0.copy() for _ in range(S)]
            k = 0
            for it in range(n_iter):
                for s in range(S):
                    if adapt and proposal == "adaptive" and adapt_start <= it < burn and it % 50 == 0 \
                            and len(hist[s]) > 50:
                        H = np.asarray(hist[s][-1000:])
                        cov[s] = (2.38 ** 2 / P) * np.cov(H.T) + 1e-6 * np.eye(P)
                    prop = st.phi[s] + rng.multivariate_normal(np.zeros(P), cov[s])
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
        return SDMResult(draws, lps, acc, self.param_names, self.sectors,
                         mode=getattr(self, "mode_", None), model=self)

    # ------------------------------------------------------------- flows
    def column_totals_toroj(self, t):
        """colZ[i, s, j] = z_ij v^s_j / v_j (Torój 2024 eq. 1)."""
        va = self.d.va[t]
        share = va / np.maximum(va.sum(axis=1, keepdims=True), 1e-300)
        return self.d.Znat[t][:, None, :] * share.T[None, :, :]

    def flows(self, phi, t=-1, col_totals=None):
        """Initial Z[r, i, s, j] (before balancing) for one parameter vector."""
        t = t % self.d.T
        colZ = self.column_totals_toroj(t) if col_totals is None else np.asarray(col_totals, float)
        Ws = np.stack([self.W(s, phi[s], t) for s in range(self.d.S)])
        return np.einsum("spv,srp->rspv", colZ, Ws)

    def posterior_flows(self, res: SDMResult, t=-1, col_totals=None, n_draws: int = 200,
                        quantiles=(0.05, 0.95), seed: int = 0, keep_draws: bool = False, balance=None):
        """Posterior mean/sd (and quantiles) of Z. ``balance`` = dict(row=..., col=..., block=...)
        applies ``balance_three_way`` to every draw."""
        rng = np.random.default_rng(seed)
        flat = res.draws.reshape(-1, *res.draws.shape[2:])
        pick = rng.choice(len(flat), size=min(n_draws, len(flat)), replace=False)
        mean = M2 = None
        store = [] if (keep_draws or quantiles) else None
        for n, i in enumerate(pick, 1):
            Z = self.flows(flat[i], t, col_totals)
            if balance:
                Z, _ = balance_three_way(Z, **balance)
            if mean is None:
                mean, M2 = np.zeros_like(Z), np.zeros_like(Z)
            d = Z - mean
            mean += d / n
            M2 += d * (Z - mean)
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


# ===================================================================== balancing
def balance_three_way(Z4, row=None, col=None, block=None, tol: float = 1e-9, maxit: int = 5000):
    """Extended RAS of Torój (2021, 2024 Sect. 2) on Z[r, i, s, j]:
      row   (R, S): sum_{s,j} Z = regional supply potential of i in r
      col   (R, S): sum_{r}   Z[., i, s, j] summed over i -> purchases of j in s  [given as (s, j)]
      block (S, S): sum_{r,s} Z[., i, ., j] = national z_ij
    Any of the three can be None. Iterative proportional fitting over the given margins.
    The margins must be mutually consistent (sum_r row[r,i] = sum_j block[i,j] and
    sum_s col[s,j] = sum_i block[i,j]); if ``block`` is given, row and column targets are
    rescaled sector by sector to satisfy this and the factors are returned in info.
    """
    Z = np.array(Z4, float)
    info = {"converged": False, "iterations": 0}
    if block is not None:
        block = np.asarray(block, float)
        if row is not None:
            row = np.array(row, float)
            f = np.divide(block.sum(axis=1), row.sum(axis=0), out=np.ones(row.shape[1]), where=row.sum(axis=0) > 0)
            row *= f[None, :]
            info["row_rescale"] = f
        if col is not None:
            col = np.array(col, float)
            f = np.divide(block.sum(axis=0), col.sum(axis=0), out=np.ones(col.shape[1]), where=col.sum(axis=0) > 0)
            col *= f[None, :]
            info["col_rescale"] = f
    for it in range(1, maxit + 1):
        err = 0.0
        if row is not None:
            cur = Z.sum(axis=(2, 3))
            f = np.divide(row, cur, out=np.ones_like(cur), where=cur > 0)
            Z *= f[:, :, None, None]
            err = max(err, np.max(np.abs(cur - row)) / max(np.max(row), 1e-300))
        if col is not None:
            cur = Z.sum(axis=(0, 1))
            f = np.divide(col, cur, out=np.ones_like(cur), where=cur > 0)
            Z *= f[None, None, :, :]
            err = max(err, np.max(np.abs(cur - col)) / max(np.max(col), 1e-300))
        if block is not None:
            cur = Z.sum(axis=(0, 2))
            f = np.divide(block, cur, out=np.ones_like(cur), where=cur > 0)
            Z *= f[None, :, None, :]
            err = max(err, np.max(np.abs(cur - block)) / max(np.max(block), 1e-300))
        info["iterations"] = it
        if err < tol:
            info["converged"] = True
            break
    info["max_rel_err"] = err
    return Z, info


def balance_flows(Z4, row_totals, col_totals, **ras_kw):
    """Two-margin RAS on Z[r,s,p,v] (rows (r,s), columns (p,v))."""
    from .ras import ras
    R, S = Z4.shape[:2]
    X, info = ras(Z4.reshape(R * S, R * S), np.ravel(row_totals), np.ravel(col_totals), **ras_kw)
    return X.reshape(R, S, R, S), info


def origin_shares_from_flows(T_irb):
    T = np.asarray(T_irb, float)
    tot = T.sum(axis=0, keepdims=True)
    sh = np.divide(T, tot, out=np.zeros_like(T), where=tot > 0)
    return sh.transpose(1, 0, 2)


# ===================================================================== Canada helpers
# Provincial/territorial capitals (lat, lon) and land/sea adjacency, for the Torój r_home
# analogue (average distance from a capital to the border along the line to each
# neighbouring capital, approximated by half the capital-to-capital distance).
CANADA_CAPITALS = {
    "NL": (47.561, -52.713), "PE": (46.238, -63.131), "NS": (44.649, -63.575), "NB": (45.964, -66.643),
    "QC": (46.813, -71.208), "ON": (43.653, -79.383), "MB": (49.895, -97.138), "SK": (50.445, -104.619),
    "AB": (53.546, -113.494), "BC": (48.428, -123.366), "YT": (60.721, -135.057), "NT": (62.454, -114.372),
    "NU": (63.749, -68.522)}
CANADA_NEIGHBOURS = {
    "NL": ["QC"], "PE": ["NS", "NB"], "NS": ["NB", "PE"], "NB": ["QC", "NS", "PE"],
    "QC": ["ON", "NB", "NL"], "ON": ["QC", "MB"], "MB": ["ON", "SK", "NU"], "SK": ["MB", "AB", "NT"],
    "AB": ["SK", "BC", "NT"], "BC": ["AB", "YT", "NT"], "YT": ["BC", "NT"], "NT": ["YT", "BC", "AB", "SK", "NU"],
    "NU": ["NT", "MB"]}


def canada_r_home(provinces=None, polygons=None, return_pairs: bool = False, rule: str = "median"):
    """Torój's within-region radius for Canadian provinces/territories (km).

    Torój (2024, Sect. 3): the average of all distances from each regional capital to its own
    region's border, measured on the straight lines to the capitals of neighbouring regions.
    * polygons given (``geography.read_shapefile`` on Statistics Canada's provincial boundary
      file, keyed by province code): the exact definition (``geography.toroj_r_home``).
    * no polygons: approximation -- the border is assumed half-way between the two capitals,
      so each distance is 0.5 x the capital-to-capital great-circle distance. This is NOT
      exact: it over-states the radius for capitals far inside their province and under-states
      it for capitals near a border (e.g. Toronto, Victoria, Quebec City).
    Distances are pooled over all (region, neighbour) pairs, as in Torój.
    """
    from .distances import haversine
    provs = provinces or list(CANADA_CAPITALS)
    if polygons is not None:
        from .geography import toroj_r_home
        r, rows = toroj_r_home(polygons, CANADA_CAPITALS, CANADA_NEIGHBOURS, provs, rule=rule)
        return (r, rows) if return_pairs else r
    rows = []
    for p in provs:
        la, lo = CANADA_CAPITALS[p]
        for q in CANADA_NEIGHBOURS[p]:
            if q in provs:
                rows.append((p, q, 0.5 * float(haversine(la, lo, *CANADA_CAPITALS[q]))))
    r = float(np.mean([d for _, _, d in rows]))
    return (r, rows) if return_pairs else r
