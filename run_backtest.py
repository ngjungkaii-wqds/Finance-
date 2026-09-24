"""
Backtest the FFQ strategy against the legacy momentum rulebook on identical 9-week games.

    python run_backtest.py                 # Yahoo Finance only (as the brief asks)
    python run_backtest.py --sec           # also SEC EDGAR accounts: long US test 2015-2026
    python run_backtest.py --quick         # every 3rd game only, for a fast first look
    python run_backtest.py --synthetic     # offline self-test on a synthetic market

Writes reports/BACKTEST_REPORT.md, reports/figures/*.png and reports/backtest_*.csv.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

from ffq import config as C
from ffq import fundamentals as FU
from ffq.backtest import run_games
from ffq.legacy import legacy_weights
from ffq.pipeline import load
from ffq.strategy import FFQStrategy

HELD_COLS = ["gate_pass", "roe", "k_e", "fscore", "ni_growth", "pe", "de", "net_margin"]

ALL_GATES = ("G1", "G2", "G3", "G4", "G5", "G6")
# Robustness run: each line changes ONE rule. Reported to show how much each rule matters;
# never used to choose the rules (they were fixed on 24 Sep 2026).
SENSITIVITY = [
    ("Base (rules as specified)", {}, C.STOP),
    ("Without G2 (ROE >= SML cost of equity)", {"GATES": tuple(g for g in ALL_GATES if g != "G2")}, C.STOP),
    ("Without G3 (Piotroski F-score)", {"GATES": tuple(g for g in ALL_GATES if g != "G3")}, C.STOP),
    ("Without G4 (profit not falling)", {"GATES": tuple(g for g in ALL_GATES if g != "G4")}, C.STOP),
    ("Without G6 (P/E cap)", {"GATES": tuple(g for g in ALL_GATES if g != "G6")}, C.STOP),
    ("Score: residual momentum only", {"W_MOM": 1.0, "W_QUAL": 0.0, "W_VALUE": 0.0}, C.STOP),
    ("Score: equal thirds", {"W_MOM": 1 / 3, "W_QUAL": 1 / 3, "W_VALUE": 1 / 3}, C.STOP),
    ("Correlation limit 0.55", {"CORR_MAX_PAIR": 0.55}, C.STOP),
    ("Correlation limit 0.75", {"CORR_MAX_PAIR": 0.75}, C.STOP),
    ("No volatility target (fully invested)", {"SIGMA_TARGET": 10.0}, C.STOP),
    ("No -25% stop", {}, None),
    ("Market risk premium 4.5%", {"MRP": 0.045}, C.STOP),
    ("Market risk premium 6.5%", {"MRP": 0.065}, C.STOP),
]


def run_sensitivity(m, fe, periods, starts, universes, countries=("SG", "US"), verbose=True) -> dict:
    out = {}
    for n, (label, over, stop) in enumerate(SENSITIVITY, 1):
        saved = {k: getattr(C, k) for k in over}
        try:
            for k, v in over.items():
                setattr(C, k, v)
            strat = FFQStrategy(m, fe, periods, countries=countries)
            W = {i: strat.book(m.dates[i]).weights for i in starts}
        finally:
            for k, v in saved.items():
                setattr(C, k, v)
        g, _ = run_games(m, W, universes, stop=stop)
        out[label] = g
        if verbose:
            print(f"  robustness {n}/{len(SENSITIVITY)}: {label}", flush=True)
    return out


def coverage(strat: FFQStrategy, i: int) -> float:
    """Share of the universe with two published years of accounts at start i.

    Measured per market and the lower share returned, so SEC accounts for US names
    cannot make the combined strategy look testable before Singapore accounts exist.
    """
    date = strat.m.dates[i]
    uni = strat.universe_at(i)
    shares = []
    for ctry in strat.countries:
        names = [t for t in uni if strat.m.country(t) == ctry]
        if not names:
            continue
        ok = sum(FU.asof(strat.periods[tk].annual if tk in strat.periods else [], date)[1] is not None for tk in names)
        shares.append(ok / len(names))
    return min(shares) if shares else 0.0


def holdings_quality(tab: pd.DataFrame, w: pd.Series) -> dict:
    """Average fundamentals of a book's holdings (equal-weighted across names)."""
    if len(w) == 0 or tab.empty:
        return {}
    t = tab.reindex(w.index)
    out = {"n_held": len(w)}
    gp = t["gate_pass"].astype(float) if "gate_pass" in t else pd.Series(np.nan, index=w.index)
    out["share_fail_gate"] = 1.0 - gp.mean()
    for c in ("roe", "fscore", "pe", "net_margin"):
        if c in t:
            out[f"avg_{c}"] = pd.to_numeric(t[c], errors="coerce").median() if c == "pe" else pd.to_numeric(t[c], errors="coerce").mean()
    if "ni_growth" in t:
        g = pd.to_numeric(t["ni_growth"], errors="coerce")
        out["share_profit_falling"] = float((g < 0).sum() / g.notna().sum()) if g.notna().sum() else np.nan
    if "roe" in t and "k_e" in t:
        spread = pd.to_numeric(t["roe"], errors="coerce") - pd.to_numeric(t["k_e"], errors="coerce")
        out["share_roe_below_ke"] = float((spread < 0).sum() / spread.notna().sum()) if spread.notna().sum() else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2012-01-01", help="first price date to download")
    ap.add_argument("--first-game", default="2016-01-04", help="first game start (the complete record starts in 2016)")
    ap.add_argument("--sec", action="store_true", help="add SEC EDGAR point-in-time US accounts")
    ap.add_argument("--quick", action="store_true", help="every 3rd game only")
    ap.add_argument("--synthetic", action="store_true", help="offline self-test")
    ap.add_argument("--sensitivity", action="store_true",
                    help="also re-run FFQ with one rule changed at a time (adds 15-40 minutes)")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    t0 = time.time()

    if args.synthetic:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from tests.synthetic import SyntheticProvider
        provider = SyntheticProvider()
    else:
        provider = None
    if args.sec and args.synthetic:
        from tests.synthetic import patch_sec
        patch_sec(provider)
    m, fe, periods, source, notes = load(provider, start=args.start, use_sec=args.sec)
    print(f"Data ready in {time.time() - t0:.0f}s: {len(m.tickers)} stocks, "
          f"{m.dates[0].date()} to {m.dates[-1].date()}", flush=True)

    new = FFQStrategy(m, fe, periods, name="FFQ")
    risk_engine = FFQStrategy(m, fe, periods, use_gate=False, use_fund_score=False, name="RISK_ENGINE")
    first = max(m.loc(args.first_game), C.FORM_DAYS + 5)
    last = len(m.dates) - 1 - C.HOLD_DAYS
    step = C.STEP_DAYS * (3 if args.quick else 1)
    starts = list(range(first, last + 1, step))
    print(f"{len(starts)} games from {m.dates[starts[0]].date()} to {m.dates[starts[-1]].date()}", flush=True)

    W = {k: {} for k in ["FFQ", "FFQ_EQUAL", "RISK_ENGINE", "LEGACY", "LEGACY_LIQ", "LEGACY_GATED"]}
    if args.sec:
        new_us = FFQStrategy(m, fe, periods, countries=("US",), name="FFQ_US")
        re_us = FFQStrategy(m, fe, periods, use_gate=False, use_fund_score=False, countries=("US",), name="RISK_ENGINE_US")
        for k in ("FFQ_US", "RISK_ENGINE_US", "MOM_US"):
            W[k] = {}
    universes, cov, bookinfo, quality = {}, {}, [], []
    for n, i in enumerate(starts, 1):
        date = m.dates[i]
        universes[i] = new.universe_at(i)
        cov[i] = coverage(new, i)
        full = cov[i] >= 0.6
        W["LEGACY"][i] = legacy_weights(m, date, "legacy")
        W["LEGACY_LIQ"][i] = legacy_weights(m, date, "liquid")
        rb = risk_engine.book(date)
        W["RISK_ENGINE"][i] = rb.weights
        held_tabs = [rb.table]
        if full:
            b = new.book(date)
            W["FFQ"][i] = b.weights
            W["FFQ_EQUAL"][i] = (pd.Series(min(1.0 / len(b.picked), C.W_MAX), index=b.picked)
                                 if b.picked else pd.Series(dtype=float))
            legacy_names = [t for t in C.SG + C.US_LEGACY + C.GOLD if t in m.px.columns]
            gt = new.assess(legacy_names, i)
            allowed = set(gt.index[gt["gate_pass"].astype(bool)])
            W["LEGACY_GATED"][i] = legacy_weights(m, date, "legacy", allowed=allowed)
            held = set().union(*[set(W[k][i].index) for k in ("FFQ", "RISK_ENGINE", "LEGACY", "LEGACY_LIQ", "LEGACY_GATED")])
            extra = [t for t in held if t not in gt.index and t not in b.table.index]
            tabs = [b.table, gt] + ([new.assess(extra, i)] if extra else [])
            qt = pd.concat(tabs)
            qt = qt[~qt.index.duplicated(keep="first")]
            for k in ("FFQ", "FFQ_EQUAL", "RISK_ENGINE", "LEGACY", "LEGACY_LIQ", "LEGACY_GATED"):
                quality.append({"start": date, "variant": k, **holdings_quality(qt, W[k][i])})
            r = b.risk
            bookinfo.append({"start": date, "n": len(b.weights), "invested": float(b.weights.sum()),
                             "vol_exante": r.get("vol_ann"), "beta_exante": r.get("beta_spy"),
                             "avg_pair_corr": r.get("avg_pair_corr"), "max_pair_corr": r.get("max_pair_corr"),
                             "effective_n": r.get("effective_n"), "div_ratio": r.get("diversification_ratio"),
                             "n_eligible": b.diag.get("n_eligible"), "n_universe": b.diag.get("n_universe"),
                             "method": b.diag.get("method"), "notes": "; ".join(b.diag.get("notes", [])),
                             "names": " ".join(f"{t}:{w:.3f}" for t, w in b.weights.items())})
        if args.sec:
            if coverage(new_us, i) >= 0.6:
                W["FFQ_US"][i] = new_us.book(date).weights
            W["RISK_ENGINE_US"][i] = re_us.book(date).weights
            mom = fe.raw_momentum(new_us.universe_at(i), date).sort_values(ascending=False)
            W["MOM_US"][i] = pd.Series(1.0 / 8, index=mom.index[:8])
        if n % 10 == 0 or n == len(starts):
            el = time.time() - t0
            print(f"  game {n}/{len(starts)} {date.date()} coverage {cov[i]:.0%} ({el:.0f}s)", flush=True)

    games = {}
    for k, wd in W.items():
        if wd:
            g, paths = run_games(m, wd, universes)
            games[k] = (g, paths)
            print(f"  simulated {k}: {len(g)} games", flush=True)

    sens = {}
    if args.sensitivity:
        full_starts = [i for i in starts if cov[i] >= 0.6]
        if full_starts:
            print(f"Robustness run on {len(full_starts)} full-accounts games", flush=True)
            sens["Singapore + US, full-accounts period"] = run_sensitivity(m, fe, periods, full_starts, universes)
        if args.sec and W.get("FFQ_US"):
            us_starts = sorted(W["FFQ_US"])[::3]
            print(f"Robustness run on {len(us_starts)} US-only SEC games (every 3rd start)", flush=True)
            sens["US only, SEC accounts (every 3rd game)"] = run_sensitivity(m, fe, periods, us_starts, universes,
                                                                          countries=("US",))

    os.makedirs(args.out, exist_ok=True)
    state = {"sensitivity": sens,"games": {k: v[0] for k, v in games.items()}, "paths": {k: v[1] for k, v in games.items()},
             "coverage": pd.Series({m.dates[i]: c for i, c in cov.items()}),
             "bookinfo": pd.DataFrame(bookinfo), "quality": pd.DataFrame(quality),
             "notes": notes, "source": source, "synthetic": args.synthetic, "sec": args.sec,
             "rf": m.rf_annual, "spy": m.etf.get("SPY"), "dates": (m.dates[0], m.dates[-1]),
             "run": dt.datetime.now().isoformat(timespec="minutes"), "quick": args.quick,
             "n_stocks": len(m.tickers)}
    with open(os.path.join(args.out, "backtest_state.pkl"), "wb") as fh:
        pickle.dump(state, fh)
    for k, (g, _) in games.items():
        g.to_csv(os.path.join(args.out, f"backtest_games_{k}.csv"))
    state["bookinfo"].to_csv(os.path.join(args.out, "backtest_ffq_books.csv"), index=False)

    from ffq.report import write_report
    path = write_report(state, args.out)
    print(f"Report written to {path} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
