"""
The FFQ strategy at one date: universe -> SML/FF3/fundamentals -> gate -> score ->
correlation-aware selection -> max-Sharpe weights -> capital allocation line.

`FFQStrategy.book(date)` uses only information available at the close of `date`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from . import fundamentals as FU
from . import risk as R
from .data import Market, us_liquid
from .factors import FactorEngine


@dataclass
class Book:
    date: pd.Timestamp
    weights: pd.Series                         # fraction of capital per stock (sum <= 1)
    table: pd.DataFrame = field(default_factory=pd.DataFrame)   # every stock examined
    log: list = field(default_factory=list)    # selection trail
    risk: dict = field(default_factory=dict)
    diag: dict = field(default_factory=dict)
    picked: list = field(default_factory=list)  # names chosen before weighting


def _z(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score, winsorised at +/-3."""
    s = pd.to_numeric(s, errors="coerce").astype(float)
    ok = s.notna()
    if ok.sum() < 3 or s[ok].std(ddof=0) == 0:
        return pd.Series(0.0, index=s.index).where(ok)
    z = (s - s[ok].mean()) / s[ok].std(ddof=0)
    return z.clip(-3, 3)


def _zmean(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Average of the available z-scores (a missing input does not count against a stock)."""
    zs = pd.concat([_z(df[c]) for c in cols if c in df.columns], axis=1)
    return zs.mean(axis=1, skipna=True)


class FFQStrategy:
    """Parameters that define the backtest variants:

    use_gate        apply the fundamental gate (False = price-only risk engine)
    use_fund_score  include quality and value in the ranking score
    optimizer       "maxsharpe" (with the capital allocation line) or "equal"
    universe        "liquid" (SG 46 + top-40 US by liquidity) or "legacy" (SG 46 + US 28 + gold)
    countries       ("SG", "US") or a subset, e.g. ("US",) for the long SEC-EDGAR test
    """

    def __init__(self, m: Market, fe: FactorEngine, periods: dict, use_gate=True,
                 use_fund_score=True, optimizer="maxsharpe", universe="liquid",
                 countries=("SG", "US"), name="FFQ"):
        self.m, self.fe, self.periods = m, fe, periods
        self.use_gate, self.use_fund_score = use_gate, use_fund_score
        self.optimizer, self.universe, self.name = optimizer, universe, name
        self.countries = tuple(countries)
        self._count = m.px.notna().cumsum()      # days of history per stock, by date

    # ---- universe ------------------------------------------------------------
    def universe_at(self, i: int) -> list[str]:
        m = self.m
        cnt = self._count.iloc[i]
        long_enough = set(cnt[cnt > C.FORM_DAYS].index)
        sg = [t for t in C.SG if t in long_enough] if "SG" in self.countries else []
        us = []
        if "US" in self.countries:
            if self.universe == "legacy":
                us = [t for t in C.US_LEGACY + C.GOLD if t in long_enough]
            else:
                pool = [t for t in C.US_POOL if t in m.px.columns]
                us = [t for t in us_liquid(m, i, pool, C.US_TOP_N) if t in long_enough]
        return [t for t in sg + us if pd.notna(m.px[t].iloc[i])]

    # ---- fundamentals at a date ----------------------------------------------
    def fundamentals_at(self, tickers, i: int) -> dict:
        m = self.m
        date = m.dates[i]
        lo = max(0, i - 251)
        out = {}
        for tk in tickers:
            per = self.periods.get(tk)
            if per is None or not per.annual:
                out[tk] = {"fy_end": None}
                continue
            q, f = m.quote_ccy.get(tk, "USD"), m.fin_ccy.get(tk, "USD")
            fxq = float(m.fx[q].iloc[i]) if m.fx.get(q) is not None else 1.0
            if m.fx.get(f) is not None:
                fxf = float(m.fx[f].iloc[i])
            elif f == q:
                fxf = fxq
            else:
                fxf = np.nan   # unknown reporting currency: valuation cannot be trusted
            price = float(m.raw_local[tk].iloc[i]) if tk in m.raw_local.columns else np.nan
            d12 = float(m.divs_local[tk].iloc[lo:i + 1].sum()) if tk in m.divs_local.columns else np.nan
            out[tk] = FU.metrics(tk, per, date, price, fxq, fxf, d12)
        return out

    def assess(self, tickers, i: int, with_fund: bool = True) -> pd.DataFrame:
        """Betas, SML cost of equity, FF3 statistics, fundamentals and gate for each ticker."""
        m = self.m
        date = m.dates[i]
        capm = self.fe.capm(tickers, date)
        ff = self.fe.ff3(tickers, date)
        mom = self.fe.raw_momentum(tickers, date)
        k_e = self.fe.cost_of_equity(capm["beta_adj"], date) if len(capm) else pd.Series(dtype=float)
        fund = self.fundamentals_at(tickers, i) if with_fund else {}
        rows = {}
        for tk in tickers:
            r = {"country": m.country(tk), "sector": C.SECTOR.get(tk, "Other"),
                 "mom_12_1": mom.get(tk, np.nan)}
            if tk in capm.index:
                r.update(capm.loc[tk].to_dict())
                r["k_e"] = float(k_e.get(tk, np.nan))
            else:
                r.update(beta_local=np.nan, beta_spy=np.nan, beta_adj=np.nan, k_e=np.nan)
            if tk in ff.index:
                r.update(ff.loc[tk].to_dict())
            if with_fund:
                fm = fund.get(tk, {})
                passed, fails, flags = FU.gate(tk, fm, r["k_e"])
                r.update({k: v for k, v in fm.items() if not isinstance(v, dict)})
                r.update(FU.quality_value_parts(fm, r["k_e"]))
                r["gate_pass"], r["gate_fails"], r["flags"] = passed, "; ".join(fails), "; ".join(flags)
            rows[tk] = r
        return pd.DataFrame(rows).T

    # ---- the book --------------------------------------------------------------
    def book(self, date) -> Book:
        m, fe = self.m, self.fe
        i = m.loc(date)
        date = m.dates[i]
        uni = self.universe_at(i)
        rf = float(m.rf_annual.iloc[i])
        table = self.assess(uni, i, with_fund=(self.use_gate or self.use_fund_score))
        if table.empty or "ff_mom" not in table.columns:
            return Book(date=date, weights=pd.Series(dtype=float), table=table,
                        diag={"note": "not enough history for the factor model"})

        elig = table["ff_mom"].notna()
        if self.use_gate:
            elig &= table["gate_pass"].fillna(False).astype(bool)
        table["eligible"] = elig
        E = table[elig].copy()
        for c in ("z_mom", "z_qual", "z_value", "score"):
            table[c] = np.nan
        # scores are z-scored within each country: SG and US valuations are not comparable
        parts = []
        for ctry, g in E.groupby("country"):
            g = g.copy()
            g["z_mom"] = _z(g["ff_mom"])
            if self.use_fund_score:
                g["z_qual"] = _zmean(g, ["q_spread", "q_fscore", "q_growth", "q_accruals"])
                g["z_value"] = _zmean(g, ["v_ep", "v_bp", "v_fcf"])
                g["score"] = (C.W_MOM * g["z_mom"] + C.W_QUAL * g["z_qual"].fillna(0)
                              + C.W_VALUE * g["z_value"].fillna(0))
            else:
                g["score"] = g["z_mom"]
            parts.append(g)
        diag = {"n_universe": len(uni), "n_eligible": int(elig.sum())}
        if not parts:
            diag["note"] = "no stock passed the gate"
            return Book(date=date, weights=pd.Series(dtype=float), table=table, diag=diag)
        E = pd.concat(parts)
        for c in ("z_mom", "z_qual", "z_value", "score"):
            if c in E.columns:
                table.loc[E.index, c] = E[c]
        cand = []
        for ctry, g in E.groupby("country"):
            cand += list(g.sort_values("score", ascending=False).index[: C.CANDIDATES_PER_COUNTRY])
        cand = sorted(cand, key=lambda t: -float(E.at[t, "score"]))

        Sigma, cinfo = R.covariance(fe, cand, date)
        log = []
        picked = R.select(cand, cinfo.get("lw_corr", pd.DataFrame()), C.N_MAX, log)
        diag.update(n_candidates=len(cand), lw_shrinkage=cinfo.get("lw_shrinkage"))
        if not picked:
            return Book(date=date, weights=pd.Series(dtype=float), table=table, log=log, diag=diag)

        # expected return = SML required return + alpha (IC x residual vol x standardised score)
        sc = E.loc[picked, "score"].astype(float)
        sd = float(E["score"].std(ddof=0)) or 1.0
        zc = (sc - float(E["score"].mean())) / sd
        resid = pd.to_numeric(table.loc[picked, "resid_vol_ann"], errors="coerce").fillna(0.25)
        alpha = C.IC * resid * zc
        ke = pd.to_numeric(table.loc[picked, "k_e"], errors="coerce").fillna(rf + C.MRP)
        mu = ke + alpha
        beta_spy = pd.to_numeric(table.loc[picked, "beta_spy"], errors="coerce").fillna(1.0)
        table["alpha_exante"], table["mu"] = np.nan, np.nan
        table.loc[picked, "alpha_exante"] = alpha
        table.loc[picked, "mu"] = mu

        if self.optimizer == "equal":
            n = len(picked)
            w = pd.Series(min(1.0 / n, C.W_MAX), index=picked)
            diag["method"] = "equal weight"
        else:
            w, vol, d2 = R.build_weights(picked, Sigma, mu, rf, beta_spy)
            diag.update(d2)
        w = w[w > 1e-6]
        risk = R.risk_report(w, Sigma, cinfo.get("apt"), beta_spy, mu, rf) if len(w) else {}
        table["weight"] = 0.0
        table.loc[w.index, "weight"] = w
        return Book(date=date, weights=w, table=table, log=log, risk=risk, diag=diag, picked=picked)
