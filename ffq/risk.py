"""
Risk model and portfolio construction.

Covariance
    Sigma = 0.5 x APT factor-model covariance (B Sigma_F B' + D)
          + 0.5 x Ledoit-Wolf shrunk sample covariance,
    from weekly USD returns over three years. The factor model gives stable
    estimates for 20+ stocks from 156 weeks; the sample part keeps industry
    co-movements (e.g. DBS-OCBC) that the factor model cannot see.

Construction (Markowitz, with the capital allocation line)
    1. Greedy selection by score with a MAX pairwise correlation gate and sector,
       semiconductor and gold name caps.
    2. Constrained maximum-Sharpe (tangency) portfolio, expected return
       mu_i = SML required return k_i + alpha_i (Treynor-Black / Grinold-Kahn alpha).
    3. Capital allocation line: invest y = min(1, target vol / book vol) in the
       tangency book, the rest in cash.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import config as C
from .factors import FactorEngine

try:
    from sklearn.covariance import LedoitWolf
except Exception:  # pragma: no cover
    LedoitWolf = None


def covariance(fe: FactorEngine, tickers: list[str], date) -> tuple[pd.DataFrame, dict]:
    """Annualised blended covariance matrix of weekly USD returns."""
    apt = fe.apt(tickers, date)
    info = {"apt": apt}
    if apt is None:
        return pd.DataFrame(), info
    tk = apt["tickers"]
    B, Sf, D = apt["B"], apt["cov_f"], apt["resvar"]
    S_apt = B @ Sf @ B.T + np.diag(D)
    wk = fe.weeks_upto(date)[-C.COV_WEEKS:]
    R = fe.wret.loc[wk, tk].dropna(how="any")
    S = S_apt
    if LedoitWolf is not None and len(R) >= 52:
        lw = LedoitWolf().fit(R.values)
        S = C.COV_BLEND * S_apt + (1 - C.COV_BLEND) * lw.covariance_
        info["lw_shrinkage"] = float(lw.shrinkage_)
        info["lw_corr"] = pd.DataFrame(_corr(lw.covariance_), index=tk, columns=tk)
    else:
        info["lw_corr"] = pd.DataFrame(_corr(S_apt), index=tk, columns=tk)
    S = (S + S.T) / 2.0 * 52.0
    return pd.DataFrame(S, index=tk, columns=tk), info


def _corr(S: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    return S / np.outer(d, d)


def select(ranked: list[str], corr: pd.DataFrame, n_max: int = C.N_MAX,
           log: list | None = None) -> list[str]:
    """Walk down the ranking; accept a stock unless it breaks a correlation or name cap."""
    picked, sectors, n_semi, n_gold = [], {}, 0, 0
    for tk in ranked:
        if len(picked) >= n_max:
            break
        sec = C.SECTOR.get(tk, "Other")
        why = None
        if tk not in corr.index:
            why = "no covariance estimate (short history)"
        elif tk in C.SEMI and n_semi >= C.SEMI_N_MAX:
            why = f"semiconductor cap ({C.SEMI_N_MAX} names)"
        elif tk in C.GOLD and n_gold >= C.GOLD_N_MAX:
            why = "gold miner cap (1 name)"
        elif sectors.get(sec, 0) >= C.SECTOR_N_MAX:
            why = f"{sec} sector cap ({C.SECTOR_N_MAX} names)"
        elif picked:
            cmax = corr.loc[tk, picked].max()
            if cmax > C.CORR_MAX_PAIR:
                other = corr.loc[tk, picked].idxmax()
                why = f"correlation {cmax:.2f} with {other} above {C.CORR_MAX_PAIR}"
        if why:
            if log is not None:
                log.append((tk, "skip", why))
            continue
        picked.append(tk)
        sectors[sec] = sectors.get(sec, 0) + 1
        n_semi += tk in C.SEMI
        n_gold += tk in C.GOLD
        if log is not None:
            log.append((tk, "select", ""))
    return picked


def _constraints(names, Sigma, beta_spy, relax):
    n = len(names)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    secs = pd.Series([C.SECTOR.get(t, "Other") for t in names])
    sec_cap = relax.get("sector_cap", C.SECTOR_W_MAX)
    for s in secs.unique():
        idx = np.where(secs.values == s)[0]
        if len(idx) * C.W_MAX > sec_cap + 1e-9:
            cons.append({"type": "ineq", "fun": lambda w, idx=idx: sec_cap - w[idx].sum()})
    semi_idx = np.array([i for i, t in enumerate(names) if t in C.SEMI])
    if len(semi_idx) * C.W_MAX > C.SEMI_W_MAX:
        cons.append({"type": "ineq", "fun": lambda w: C.SEMI_W_MAX - w[semi_idx].sum()})
    if relax.get("country", True):
        for ctry in ("SG", "US"):
            idx = np.array([i for i, t in enumerate(names) if t.endswith(".SI") == (ctry == "SG")])
            if len(idx):
                cons.append({"type": "ineq", "fun": lambda w, idx=idx: w[idx].sum() - C.COUNTRY_W_MIN})
    if relax.get("beta", True) and beta_spy is not None and np.all(np.isfinite(beta_spy)):
        cons.append({"type": "ineq", "fun": lambda w: C.BETA_SPY_MAX - w @ beta_spy})
    if relax.get("rc", True):
        def rc(w):
            v = w @ Sigma @ w
            return C.RC_MAX * v - w * (Sigma @ w)
        cons.append({"type": "ineq", "fun": rc})
    return cons


def _feasible_relaxations(names, beta_spy):
    """Drop constraints that cannot all hold for this set of names, and say which."""
    relax, notes = {}, []
    secs = pd.Series([C.SECTOR.get(t, "Other") for t in names])
    caps = sum(min(C.SECTOR_W_MAX, (secs == s).sum() * C.W_MAX) for s in secs.unique())
    if caps < 1.0:
        k = secs.nunique()
        relax["sector_cap"] = min(1.0, max(C.SECTOR_W_MAX, 1.0 / k + 0.05))
        notes.append(f"sector cap relaxed to {relax['sector_cap']:.0%} ({k} sectors only)")
    n_sg = sum(t.endswith(".SI") for t in names)
    n_us = len(names) - n_sg
    if min(n_sg, n_us) * C.W_MAX < C.COUNTRY_W_MIN:
        relax["country"] = False
        notes.append("country minimum dropped (too few names in one market)")
    if beta_spy is not None and np.all(np.isfinite(beta_spy)):
        lowest = np.sort(beta_spy)[: int(np.ceil(1 / C.W_MAX))]
        if lowest.mean() > C.BETA_SPY_MAX:
            relax["beta"] = False
            notes.append("beta cap dropped (no feasible low-beta mix)")
    return relax, notes


def max_sharpe(names: list[str], Sigma: pd.DataFrame, mu: pd.Series, rf: float,
               beta_spy: pd.Series | None = None) -> tuple[pd.Series, dict]:
    """Long-only constrained tangency portfolio. Falls back to minimum variance, then
    to capped inverse-variance weights, so it always returns a valid book."""
    names = list(names)
    n = len(names)
    diag = {"method": None, "notes": []}
    if n == 0:
        return pd.Series(dtype=float), diag
    if n * C.W_MAX < 1.0 - 1e-9:
        # too few justified names to be fully invested within the 15% cap: equal caps, rest cash
        w = pd.Series(C.W_MAX, index=names)
        diag["method"] = "equal-cap (too few names; remainder in cash)"
        return w, diag
    S = Sigma.loc[names, names].values
    ex = (mu.loc[names].values - rf)
    b = beta_spy.loc[names].values if beta_spy is not None else None
    relax, notes = _feasible_relaxations(names, b)
    diag["notes"] += notes
    bounds = [(0.0, C.W_MAX)] * n
    w0 = np.full(n, 1.0 / n)

    def neg_sharpe(w):
        v = w @ S @ w
        return -(w @ ex) / np.sqrt(max(v, 1e-12))

    def variance(w):
        return w @ S @ w

    for obj, label in ((neg_sharpe, "max-Sharpe"), (variance, "min-variance")):
        if label == "max-Sharpe" and not np.any(ex > 0):
            continue
        for attempt_relax in (relax, {**relax, "rc": False}, {**relax, "rc": False, "beta": False, "country": False}):
            cons = _constraints(names, S, b, attempt_relax)
            res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons,
                           options={"maxiter": 500, "ftol": 1e-10})
            if res.success and _valid(res.x, cons):
                w = np.clip(res.x, 0, None)
                w = w / w.sum()
                if attempt_relax is not relax:
                    dropped = [k for k in ("rc", "beta", "country") if attempt_relax.get(k) is False and relax.get(k, True)]
                    diag["notes"].append(f"relaxed: {', '.join(dropped)}")
                diag["method"] = label
                return pd.Series(w, index=names), diag
    # last resort: inverse variance, capped
    iv = 1.0 / np.clip(np.diag(S), 1e-8, None)
    w = _cap_normalise(iv / iv.sum(), C.W_MAX)
    diag["method"] = "inverse-variance fallback"
    return pd.Series(w, index=names), diag


def _valid(w, cons, tol=1e-6) -> bool:
    for c in cons:
        v = np.atleast_1d(c["fun"](w))
        if c["type"] == "eq" and np.any(np.abs(v) > 1e-5):
            return False
        if c["type"] == "ineq" and np.any(v < -tol):
            return False
    return True


def _cap_normalise(w: np.ndarray, cap: float) -> np.ndarray:
    w = w.copy()
    for _ in range(50):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = ~over & (w < cap)
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    return w


def build_weights(names, Sigma, mu, rf, beta_spy):
    """Tangency weights with a minimum holding size, then the CAL exposure scale."""
    w, diag = max_sharpe(names, Sigma, mu, rf, beta_spy)
    if len(w) and diag["method"] in ("max-Sharpe", "min-variance"):
        small = w[w < C.W_MIN - 1e-9].index.tolist()
        keep = [t for t in w.index if t not in small]
        if small and len(keep) * C.W_MAX >= 1.0:
            w2, diag2 = max_sharpe(keep, Sigma, mu, rf, beta_spy)
            if diag2["method"] in ("max-Sharpe", "min-variance"):
                diag2["notes"] = diag["notes"] + diag2["notes"] + [f"dropped below {C.W_MIN:.0%}: {', '.join(small)}"]
                w, diag = w2, diag2
    if len(w) == 0:
        return w, 0.0, diag
    S = Sigma.loc[w.index, w.index].values
    vol = float(np.sqrt(w.values @ S @ w.values))
    invested = float(w.sum())
    y = min(1.0, C.SIGMA_TARGET / vol) if vol > 0 else 1.0
    y = max(y, C.EXPOSURE_FLOOR)
    diag["book_vol"] = vol
    diag["cal_y"] = y
    return w * y, vol, diag


def risk_report(w: pd.Series, Sigma: pd.DataFrame, apt: dict | None, beta_spy: pd.Series,
                mu: pd.Series, rf: float, horizon_days: int = C.HOLD_DAYS) -> dict:
    """Portfolio risk statistics for the screen and the report."""
    if len(w) == 0:
        return {}
    names = list(w.index)
    S = Sigma.loc[names, names].values
    wv = w.values
    var = wv @ S @ wv
    vol = np.sqrt(var)
    rc = wv * (S @ wv) / var if var > 0 else np.zeros_like(wv)
    sd = np.sqrt(np.diag(S))
    corr = S / np.outer(sd, sd)
    iu = np.triu_indices(len(names), 1)
    pair_w = np.outer(wv, wv)[iu]
    avg_corr = float((corr[iu] * pair_w).sum() / pair_w.sum()) if pair_w.sum() > 0 else np.nan
    h = horizon_days / 252.0
    mu_p = float(wv @ mu.loc[names].values + (1 - wv.sum()) * 0.0)
    out = {
        "vol_ann": float(vol),
        "beta_spy": float(wv @ beta_spy.loc[names].values) if beta_spy is not None else np.nan,
        "exp_return_ann": mu_p,
        "sharpe_ex_ante": float((mu_p - rf * wv.sum()) / vol) if vol > 0 else np.nan,
        "diversification_ratio": float((wv @ sd) / vol) if vol > 0 else np.nan,
        "effective_n": float(1.0 / ((wv / wv.sum()) ** 2).sum()),
        "avg_pair_corr": avg_corr,
        "max_pair_corr": float(corr[iu].max()) if len(iu[0]) else np.nan,
        "var95_9wk": float(1.645 * vol * np.sqrt(h) - mu_p * h),
        "cvar95_9wk": float(2.063 * vol * np.sqrt(h) - mu_p * h),
        "risk_contrib": pd.Series(rc, index=names),
        "invested": float(wv.sum()),
    }
    if apt is not None:
        idx = [apt["tickers"].index(t) for t in names if t in apt["tickers"]]
        if len(idx) == len(names):
            B = apt["B"][idx]
            expo = wv @ B
            fvar = expo @ apt["cov_f"] @ expo * 52.0
            out["apt_exposure"] = pd.Series(expo, index=apt["factors"])
            out["factor_risk_share"] = float(fvar / var) if var > 0 else np.nan
    return out
