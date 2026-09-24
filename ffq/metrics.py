"""
Performance statistics: per-game distribution, time-series risk-adjusted measures
(Sharpe, Sortino, Treynor, Jensen's alpha, information ratio, M-squared, maximum
drawdown), and a paired moving-block bootstrap for differences between strategies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

ANN_GAMES = np.sqrt(252.0 / C.HOLD_DAYS)


def game_stats(g: pd.DataFrame, col: str = "ret") -> dict:
    r = g[col].dropna()
    if len(r) == 0:
        return {}
    rf = g.loc[r.index, "rf_game"] if "rf_game" in g else pd.Series(0.0, index=r.index)
    ex = r - rf
    down = r[r < 0]
    q = r.quantile([0.01, 0.05]).values
    out = {
        "n": len(r),
        "mean": r.mean(), "median": r.median(), "std": r.std(ddof=1),
        "win": (r > 0).mean(), "worst": r.min(), "best": r.max(),
        "p1": q[0], "p5": q[1],
        "cvar5": r[r <= q[1]].mean(),
        "p_lt_5": (r < -0.05).mean(), "p_lt_10": (r < -0.10).mean(),
        "sharpe0": r.mean() / r.std(ddof=1) * ANN_GAMES if r.std(ddof=1) > 0 else np.nan,
        "sharpe": ex.mean() / r.std(ddof=1) * ANN_GAMES if r.std(ddof=1) > 0 else np.nan,
        "sortino0": r.mean() / np.sqrt((np.minimum(r, 0) ** 2).mean()) * ANN_GAMES if len(down) else np.nan,
    }
    if "spy_tr" in g:
        b = g.loc[r.index, "spy_tr"]
        out["beat_spy"] = (r > b).mean()
        out["excess_vs_spy"] = (r - b).mean()
    return out


def nav_stats(nav: pd.Series, rf_annual: pd.Series, bench: pd.Series | None = None) -> dict:
    """Time-series statistics of a daily NAV (chained games)."""
    nav = nav.dropna()
    if len(nav) < 30:
        return {}
    r = nav.pct_change().dropna()
    rf_d = (rf_annual.reindex(r.index).ffill().fillna(0.0)) / 252.0
    ex = r - rf_d
    yrs = len(r) / 252.0
    cagr = nav.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = r.std(ddof=1) * np.sqrt(252)
    dd = nav / nav.cummax() - 1
    down = np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(252)
    out = {
        "cagr": cagr, "vol": vol,
        "sharpe": ex.mean() * 252 / vol if vol > 0 else np.nan,
        "sortino": ex.mean() * 252 / down if down > 0 else np.nan,
        "max_dd": dd.min(),
        "calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
    }
    if bench is not None:
        # weekly returns for beta: SG and US close at different times of day
        wn = nav.resample("W-FRI").last().pct_change().dropna()
        wb = bench.reindex(nav.index).ffill().resample("W-FRI").last().pct_change().reindex(wn.index)
        rfw = (rf_annual.resample("W-FRI").mean() / 52.0).reindex(wn.index).ffill().fillna(0.0)
        ok = wb.notna()
        if ok.sum() > 26:
            y, x = (wn - rfw)[ok], (wb - rfw)[ok]
            beta = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
            alpha_w = y.mean() - beta * x.mean()
            active = (wn - wb)[ok]
            te = active.std(ddof=1) * np.sqrt(52)
            bvol = wb[ok].std(ddof=1) * np.sqrt(52)
            out.update({
                "beta": beta,
                "jensen_alpha": alpha_w * 52,
                "treynor": y.mean() * 52 / beta if beta != 0 else np.nan,
                "info_ratio": active.mean() * 52 / te if te > 0 else np.nan,
                "tracking_error": te,
                "corr_bench": np.corrcoef(x, y)[0, 1],
                "m2": (out["sharpe"] * bvol + rfw.mean() * 52) if np.isfinite(out["sharpe"]) else np.nan,
            })
            bnav = bench.reindex(nav.index).ffill()
            bdd = bnav / bnav.cummax() - 1
            out["bench_max_dd"] = bdd.min()
    return out


def block_bootstrap_diff(a: pd.Series, b: pd.Series, block: int = 9, n: int = 5000,
                         seed: int = 7) -> dict:
    """Paired moving-block bootstrap of mean(a - b) and Sharpe(a) - Sharpe(b).

    Overlapping games are not independent; blocks of 9 starts (~45 trading days)
    keep that dependence, as in the complete record section 17.
    """
    df = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    if len(df) < 2 * block:
        return {}
    rng = np.random.default_rng(seed)
    A, B = df["a"].values, df["b"].values
    N = len(df)
    k = int(np.ceil(N / block))
    means, sharpes = np.empty(n), np.empty(n)
    for j in range(n):
        st = rng.integers(0, N - block + 1, size=k)
        idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:N]
        a_, b_ = A[idx], B[idx]
        means[j] = (a_ - b_).mean()
        sa, sb = a_.std(ddof=1), b_.std(ddof=1)
        sharpes[j] = (a_.mean() / sa - b_.mean() / sb) * ANN_GAMES if sa > 0 and sb > 0 else np.nan
    d = A - B
    lo, hi = np.percentile(means, [2.5, 97.5])
    slo, shi = np.nanpercentile(sharpes, [2.5, 97.5])
    return {"mean_diff": d.mean(), "mean_ci": (lo, hi), "mean_sig": not (lo <= 0 <= hi),
            "sharpe_diff": (A.mean() / A.std(ddof=1) - B.mean() / B.std(ddof=1)) * ANN_GAMES,
            "sharpe_ci": (slo, shi), "sharpe_sig": not (slo <= 0 <= shi), "n": N}
