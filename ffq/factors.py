"""
Factor models: Fama-French three-factor (FF3), CAPM / security market line, and the
APT macro-factor model.

All factors are built from Yahoo ETF prices, so the backtest needs only Yahoo data:

  FF3, US stocks:   MKT = SPY - rf    SMB = IWM - IWB    HML = IWD - IWF
  FF3, SG stocks:   MKT = EWS - rf    SMB = SCZ - EFA    HML = EFV - EFG
  APT (risk):       MKT_US, MKT_SG, SMB_US, HML_US, RATES (IEF - rf), OIL (USO),
                    GOLD_F (GLD), USD (UUP)

Returns are weekly (Friday to Friday) and in US dollars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data import Market, weekly_returns


def weekly_rf(m: Market) -> pd.Series:
    """Weekly risk-free return from the 13-week T-bill yield."""
    rf = m.rf_annual.resample("W-FRI").mean()
    return (1.0 + rf) ** (1.0 / 52.0) - 1.0


def factor_panel(m: Market) -> pd.DataFrame:
    """Weekly factor returns (columns) indexed by week-ending Friday."""
    r = weekly_returns(m.etf)
    rf = weekly_rf(m).reindex(r.index).ffill()

    def col(a, b=None):
        if a not in r.columns or (b is not None and b not in r.columns):
            return pd.Series(np.nan, index=r.index)
        return r[a] - (r[b] if b is not None else 0.0)

    f = pd.DataFrame(index=r.index)
    f["RF"] = rf
    f["MKT_US"] = col("SPY") - rf
    f["MKT_SG"] = col("EWS") - rf
    f["SMB_US"] = col("IWM", "IWB")
    f["HML_US"] = col("IWD", "IWF")
    f["SMB_INT"] = col("SCZ", "EFA")
    f["HML_INT"] = col("EFV", "EFG")
    f["RMW_US"] = col("QUAL", "SPY")     # quality minus market, a profitability proxy
    f["MOM_US"] = col("MTUM", "SPY")
    f["RATES"] = col("IEF") - rf
    f["OIL"] = col("USO")
    f["GOLD_F"] = col("GLD")
    f["USD"] = col("UUP")
    return f


FF3_SETS = {"US": ["MKT_US", "SMB_US", "HML_US"], "SG": ["MKT_SG", "SMB_INT", "HML_INT"]}


def _ols(y: np.ndarray, X: np.ndarray):
    """OLS with intercept. Returns (alpha, betas, residuals, r2)."""
    Xc = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    fit = Xc @ coef
    resid = y - fit
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else np.nan
    return coef[0], coef[1:], resid, r2


class FactorEngine:
    """Pre-computes weekly stock and factor returns once; answers queries by date."""

    def __init__(self, m: Market):
        self.m = m
        self.fac = factor_panel(m)
        self.wret = weekly_returns(m.px)
        self.wret = self.wret.reindex(self.fac.index)

    def weeks_upto(self, date) -> pd.DatetimeIndex:
        """Complete weeks (Friday labels) on or before `date`: no look-ahead."""
        return self.fac.index[self.fac.index <= pd.Timestamp(date)]

    # ---- CAPM / SML ----------------------------------------------------------
    def capm(self, tickers, date, weeks=C.BETA_WEEKS, min_weeks=C.BETA_MIN_WEEKS) -> pd.DataFrame:
        """Raw and Blume-adjusted beta to the local market and to the S&P 500 (SPY)."""
        wk = self.weeks_upto(date)[-weeks:]
        F = self.fac.loc[wk]
        rf = F["RF"]
        out = {}
        for tk in tickers:
            if tk not in self.wret.columns:
                continue
            y = self.wret.loc[wk, tk] - rf
            row = {}
            for lab, mk in (("local", "MKT_SG" if tk.endswith(".SI") else "MKT_US"), ("spy", "MKT_US")):
                x = F[mk]
                ok = y.notna() & x.notna()
                if ok.sum() < min_weeks:
                    row[f"beta_{lab}"] = np.nan
                    continue
                xv, yv = x[ok].values, y[ok].values
                b = np.cov(xv, yv, ddof=1)[0, 1] / np.var(xv, ddof=1)
                row[f"beta_{lab}"] = b
            row["beta_adj"] = (C.BLUME[0] * row["beta_local"] + C.BLUME[1]
                               if not np.isnan(row.get("beta_local", np.nan)) else np.nan)
            out[tk] = row
        return pd.DataFrame(out).T

    def cost_of_equity(self, beta_adj: pd.Series, date) -> pd.Series:
        """SML required return: k = rf + adjusted beta x market risk premium."""
        rf = float(self.m.rf_annual.asof(pd.Timestamp(date)))
        return rf + beta_adj * C.MRP

    # ---- Fama-French residual momentum ----------------------------------------
    def ff3(self, tickers, date) -> pd.DataFrame:
        """FF3 loadings, alpha and the 12-1 FF3 appraisal ratio for each stock.

        Betas are estimated on up to 156 weeks ending one month ago. The signal is
        the stock's factor-adjusted return (alpha + residual) over the formation
        window [t-252, t-21] trading days, divided by its residual volatility: the
        Treynor-Black appraisal ratio, a.k.a. residual momentum (Blitz, Huij and
        Martens 2011). Ranking on it removes the market, size and value bets that
        make plain momentum crash.
        """
        m = self.m
        i = m.loc(date)
        if i < C.FORM_DAYS:
            return pd.DataFrame()
        d_end, d_start = m.dates[i - C.SKIP_DAYS], m.dates[i - C.FORM_DAYS]
        wk_all = self.fac.index[self.fac.index <= d_end]
        wk_est = wk_all[-C.FF_WEEKS:]
        form = (wk_est > d_start)
        out = {}
        for tk in tickers:
            if tk not in self.wret.columns:
                continue
            cols = FF3_SETS["SG" if tk.endswith(".SI") else "US"]
            F = self.fac.loc[wk_est, cols]
            y = self.wret.loc[wk_est, tk] - self.fac.loc[wk_est, "RF"]
            ok = (y.notna() & F.notna().all(axis=1)).values
            if ok.sum() < C.FF_MIN_WEEKS or (ok & form).sum() < 30:
                continue
            a, b, _, r2 = _ols(y.values[ok], F.values[ok])
            resid_all = y.values - F.values @ b          # alpha + residual, every week
            fr = resid_all[ok & form]
            est_res = resid_all[ok] - a
            sig_e = est_res.std(ddof=1)
            if not np.isfinite(sig_e) or sig_e <= 0:
                continue
            out[tk] = {
                "ff_mkt": b[0], "ff_smb": b[1], "ff_hml": b[2], "ff_r2": r2,
                "ff_alpha_ann": a * 52.0,
                "resid_vol_ann": sig_e * np.sqrt(52.0),
                "ff_alpha_form_ann": fr.mean() * 52.0,
                "ff_mom": fr.mean() / fr.std(ddof=1) * np.sqrt(len(fr)) if fr.std(ddof=1) > 0 else np.nan,
            }
        return pd.DataFrame(out).T

    def raw_momentum(self, tickers, date) -> pd.Series:
        """Legacy 12-1 momentum on the combined calendar (USD)."""
        m = self.m
        i = m.loc(date)
        if i < C.FORM_DAYS:
            return pd.Series(dtype=float)
        p1, p0 = m.px.iloc[i - C.SKIP_DAYS], m.px.iloc[i - C.FORM_DAYS]
        r = (p1 / p0 - 1.0)
        return r.reindex([t for t in tickers if t in r.index]).dropna()

    # ---- APT loadings for the risk model -------------------------------------
    def apt(self, tickers, date, weeks=C.COV_WEEKS, min_weeks=C.COV_MIN_WEEKS):
        """Loadings B (stocks x factors), factor covariance and residual variances (weekly)."""
        wk = self.weeks_upto(date)[-weeks:]
        cols = [c for c in C.APT_FACTORS if self.fac.loc[wk, c].notna().sum() >= min_weeks]
        F = self.fac.loc[wk, cols]
        Fok = F.notna().all(axis=1)
        B, resvar, keep = [], [], []
        for tk in tickers:
            if tk not in self.wret.columns:
                continue
            y = self.wret.loc[wk, tk] - self.fac.loc[wk, "RF"]
            ok = (y.notna() & Fok).values
            if ok.sum() < min_weeks:
                continue
            a, b, res, _ = _ols(y.values[ok], F.values[ok])
            B.append(b)
            resvar.append(res.var(ddof=len(cols) + 1))
            keep.append(tk)
        if not keep:
            return None
        Fv = F[Fok]
        # exponentially weighted factor covariance: recent volatility matters more
        lam = 0.5 ** (1.0 / C.EWMA_HALFLIFE_WEEKS)
        w = lam ** np.arange(len(Fv))[::-1]
        w = w / w.sum()
        mu = (Fv.values * w[:, None]).sum(axis=0)
        dev = Fv.values - mu
        cov_f = (dev * w[:, None]).T @ dev / (1.0 - (w ** 2).sum())
        return {"tickers": keep, "factors": cols, "B": np.array(B), "cov_f": cov_f,
                "resvar": np.array(resvar)}
