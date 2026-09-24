"""
Backtest engine: overlapping 9-week "games", the same design as the complete record
(section 17): a new game starts every 5 trading days, buys at the next open, holds
45 trading days, pays 0.25% commission (minimum USD 25) on every buy and sell, and
applies the -25% per-lot stop (sell at the next open, never rebuy; the money stays in cash).

Cash earns 0%, as in the competition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data import Market


def _commission(value_usd: float, capital: float) -> float:
    return max(C.COMMISSION_MIN_USD, C.COMMISSION * value_usd) / capital if value_usd > 0 else 0.0


def simulate_game(m: Market, weights: pd.Series, i: int, hold: int = C.HOLD_DAYS,
                  stop: float | None = C.STOP, capital: float = C.ACCOUNT_USD) -> dict | None:
    """One game from the close of day i. Returns the net return and the daily NAV path."""
    e, x = i + 1, min(i + hold, len(m.dates) - 1)
    if e > x:
        return None
    w = weights[weights > 0]
    if len(w) == 0:
        nav = pd.Series(1.0, index=m.dates[e:x + 1])
        return {"ret": 0.0, "nav": nav, "stops": [], "commission": 0.0}
    names = list(w.index)
    entry = m.open_.iloc[e][names]
    entry = entry.where(entry.notna() & (entry > 0), m.px.iloc[e][names])
    entry = entry.where(entry.notna() & (entry > 0), m.px.iloc[i][names])
    path = (m.px.iloc[e:x + 1][names].ffill() / entry).fillna(1.0)
    opn = (m.open_.iloc[e:x + 1][names] / entry)
    comm = sum(_commission(wt * capital, capital) for wt in w.values)
    stops = []
    if stop is not None:
        for tk in names:
            p = path[tk].values
            hit = np.where(p[:-1] <= stop)[0]
            if len(hit):
                d = hit[0]
                px_exit = opn[tk].iloc[d + 1]
                if not np.isfinite(px_exit) or px_exit <= 0:
                    px_exit = p[d + 1]
                path.iloc[d + 1:, path.columns.get_loc(tk)] = px_exit
                stops.append((tk, str(path.index[d].date())))
    values = path.mul(w, axis=1)
    final = values.iloc[-1]
    comm += sum(_commission(v * capital, capital) for v in final.values)
    cash = 1.0 - w.sum()
    nav = values.sum(axis=1) + cash - _commission_path(w, capital, len(values))
    nav.iloc[-1] = final.sum() + cash - comm
    return {"ret": float(nav.iloc[-1] - 1.0), "nav": nav, "stops": stops, "commission": comm,
            "entry_date": m.dates[e], "exit_date": m.dates[x]}


def _commission_path(w: pd.Series, capital: float, n: int) -> np.ndarray:
    buy = sum(_commission(v * capital, capital) for v in w.values)
    return np.full(n, buy)


def benchmarks(m: Market, i: int, hold: int, universe: list[str]) -> dict:
    x = min(i + hold, len(m.dates) - 1)
    out = {}
    for lab, s in (("spy_tr", m.etf.get("SPY")), ("sp500", m.index_px.get("^GSPC")),
                   ("sti", m.index_px.get("^STI"))):
        if s is not None and pd.notna(s.iloc[i]) and pd.notna(s.iloc[x]):
            out[lab] = float(s.iloc[x] / s.iloc[i] - 1.0)
        else:
            out[lab] = np.nan
    u = [t for t in universe if t in m.px.columns]
    r = (m.px.iloc[x][u] / m.px.iloc[i][u] - 1.0).dropna()
    out["ew_universe"] = float(r.mean()) if len(r) else np.nan
    out["rf_game"] = float(m.rf_annual.iloc[i] * hold / 252.0)
    return out


def run_games(m: Market, weights_by_start: dict, universes: dict, stop=C.STOP,
              hold: int = C.HOLD_DAYS) -> tuple[pd.DataFrame, dict]:
    """Simulate every game. Returns (one row per game, {start index: NAV path})."""
    rows, paths = [], {}
    for i, w in sorted(weights_by_start.items()):
        g = simulate_game(m, w, i, hold, stop)
        if g is None:
            continue
        b = benchmarks(m, i, hold, universes.get(i, []))
        rows.append({"start": m.dates[i], "i": i, "ret": g["ret"], "n": int((w > 0).sum()),
                     "invested": float(w.sum()), "stops": len(g["stops"]),
                     "commission": g["commission"], "names": " ".join(w.index), **b})
        paths[i] = g["nav"]
    return pd.DataFrame(rows).set_index("start") if rows else pd.DataFrame(), paths


def chain_nav(paths: dict, hold: int = C.HOLD_DAYS, step: int = C.STEP_DAYS, phase: int = 0) -> pd.Series:
    """Non-overlapping games chained into one daily NAV: rebalance every `hold` days."""
    starts = sorted(paths)
    if not starts:
        return pd.Series(dtype=float)
    first = starts[0] + phase
    want = [s for s in starts if s >= first and (s - first) % hold == 0]
    level, pieces = 1.0, []
    for s in want:
        p = paths[s] * level
        pieces.append(p)
        level = float(p.iloc[-1])
    nav = pd.concat(pieces)
    return nav[~nav.index.duplicated(keep="last")]
