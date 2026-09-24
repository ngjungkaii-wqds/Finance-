"""
The original ACC2026 momentum rulebook (weekly_screener_v2.py, section 19 of the
complete record), rebuilt on the same data panel so both strategies are tested
on identical games.

    12-1 momentum in USD; top 5 SG then top 3 US; at most 2 semiconductors;
    skip a stock whose AVERAGE daily-return correlation with names already
    picked is >= 0.55; the stronger gold miner with positive momentum replaces
    the lowest-momentum pick; equal weights.

`allowed` optionally restricts the names (used for the "legacy + fundamental gate" test).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data import Market, us_liquid

LEGACY_SEMI = {"NVDA", "AMD", "MU", "AVGO", "QCOM", "558.SI"}


def legacy_weights(m: Market, date, universe: str = "legacy", allowed: set | None = None,
                   log: list | None = None) -> pd.Series:
    i = m.loc(date)
    if i < C.FORM_DAYS + 1:
        return pd.Series(dtype=float)
    px = m.px
    mom = (px.iloc[i - C.SKIP_DAYS] / px.iloc[i - C.FORM_DAYS] - 1.0).dropna()
    if universe == "legacy":
        us_pool, semi = C.US_LEGACY, LEGACY_SEMI
    else:
        pool = [t for t in C.US_POOL if t in px.columns and t not in C.GOLD]
        us_pool, semi = us_liquid(m, i, pool, C.US_TOP_N), C.SEMI
    rets = px.iloc[i - C.FORM_DAYS:i].pct_change(fill_method=None).iloc[1:]
    ok = lambda t: t in mom.index and (allowed is None or t in allowed)
    picked, n_semi = [], 0
    for pool, k in ((C.SG, 5), (us_pool, 3)):
        ranked = sorted([t for t in pool if ok(t)], key=lambda t: mom[t], reverse=True)
        got = 0
        for t in ranked:
            if got == k:
                break
            if t in semi and n_semi >= 2:
                continue
            if picked:
                c = rets[picked + [t]].corr().loc[t, picked]
                if np.nanmean(c.values) >= 0.55:
                    continue
            picked.append(t)
            got += 1
            n_semi += t in semi
    book = list(picked)
    golds = [g for g in C.GOLD if ok(g) and mom[g] > 0]
    if golds and book:
        weakest = min(book, key=lambda t: mom[t])
        best = max(golds, key=lambda t: mom[t])
        book = [t for t in book if t != weakest] + [best]
    if log is not None:
        log.extend(book)
    if not book:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / 8.0, index=book)   # USD 40k per name each round: 1/8 of the sleeve
