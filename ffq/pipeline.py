"""Load prices, factors and point-in-time fundamentals in one call."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import fundamentals as FU
from .data import YahooProvider, build_market
from .factors import FactorEngine


def _merge(sec_per: list, yh_per: list) -> list:
    """SEC periods (point-in-time) with any missing field filled from Yahoo's same period."""
    out = []
    for p in sec_per:
        match = [q for q in yh_per if abs((q.end - p.end).days) <= 7]
        if match:
            for k, v in match[0].v.items():
                if np.isnan(p.v.get(k, np.nan)) and not np.isnan(v):
                    p.v[k] = v
        out.append(p)
    # Yahoo years newer than the last SEC filing (EDGAR lag) are kept
    last = out[-1].end if out else pd.Timestamp.min
    out += [q for q in yh_per if q.end > last + pd.Timedelta(days=7)]
    return out


def load(provider=None, start: str = "2012-01-01", use_sec: bool = False, stocks=None,
         verbose: bool = True):
    provider = provider or YahooProvider(verbose=verbose)
    stocks = stocks or sorted(set(C.SG + C.US_POOL))
    m = build_market(provider, stocks, start, verbose=verbose)
    fe = FactorEngine(m)
    accounts, notes = {}, list(m.problems)
    if verbose:
        print(f"Loading annual and quarterly accounts for {len(m.tickers)} stocks", flush=True)
    for n, tk in enumerate(m.tickers, 1):
        try:
            st = provider.statements(tk, quarterly=True)
            annual = FU.periods_from_tables(st)
            quarters = FU.periods_from_tables(st, lag_days=C.QTR_LAG_DAYS, prefix="q_")
        except Exception as exc:
            notes.append(f"{tk}: accounts unavailable ({type(exc).__name__})")
            annual, quarters = [], []
        acc = FU.Accounts(annual=annual, quarters=quarters, ttm=FU.ttm_from_quarters(quarters))
        last_px = m.raw_local[tk].dropna()
        if annual and len(last_px):
            notes += [f"{tk}: {x}" for x in FU.fix_share_basis(acc, m.info.get(tk, {}), float(last_px.iloc[-1]))]
        accounts[tk] = acc
        if verbose and n % 25 == 0:
            print(f"  accounts: {n}/{len(m.tickers)}", flush=True)
    if use_sec:
        from .sec import load_sec_periods
        if verbose:
            print("Loading point-in-time US accounts from SEC EDGAR", flush=True)
        sec = load_sec_periods([t for t in m.tickers if not t.endswith(".SI")], m.splits, verbose)
        for tk, sa in sec.items():
            yh = accounts.get(tk)
            sa.annual = _merge(sa.annual, yh.annual if yh else [])
            if yh:
                # Yahoo quarters newer than the last SEC record stay usable (EDGAR processing lag)
                last = max(p.end for p in sa.ttm) if sa.ttm else pd.Timestamp.min
                sa.ttm += [q for q in yh.ttm if q.end > last + pd.Timedelta(days=20)]
                sa.quarters = yh.quarters
            accounts[tk] = sa
        notes.append(f"SEC EDGAR accounts loaded for {len(sec)} US stocks")
    source = {tk: a.source for tk, a in accounts.items() if a.annual}
    missing = [tk for tk in m.tickers if not accounts.get(tk) or not accounts[tk].annual]
    if missing:
        notes.append(f"no annual accounts for: {', '.join(missing)}")
    return m, fe, accounts, source, notes
