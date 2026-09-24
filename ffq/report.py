"""Write reports/BACKTEST_REPORT.md and its figures from a saved backtest state."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import config as C
from .backtest import chain_nav
from .metrics import block_bootstrap_diff, game_stats, nav_stats

LABEL = {
    "FFQ": "FFQ: new strategy",
    "FFQ_EQUAL": "FFQ picks, equal weight",
    "RISK_ENGINE": "Risk engine only (no fundamentals)",
    "LEGACY": "Legacy momentum rulebook",
    "LEGACY_LIQ": "Legacy rules, liquid US universe",
    "LEGACY_GATED": "Legacy rules + fundamental gate",
    "FFQ_US": "FFQ, US only (SEC accounts)",
    "RISK_ENGINE_US": "Risk engine, US only",
    "MOM_US": "Plain 12-1 momentum, US top 8",
    "spy_tr": "S&P 500 (SPY, total return)",
    "sp500": "S&P 500 index (price)",
    "sti": "Straits Times Index (price)",
    "ew_universe": "Equal weight, whole universe",
}
# colour follows the entity (validated categorical order; grey = benchmark)
COLOR = {"FFQ": "#2a78d6", "LEGACY": "#eb6834", "RISK_ENGINE": "#1baf7a", "LEGACY_GATED": "#eda100",
         "FFQ_EQUAL": "#e87ba4", "LEGACY_LIQ": "#008300", "FFQ_US": "#2a78d6",
         "RISK_ENGINE_US": "#1baf7a", "MOM_US": "#4a3aa7", "SPY": "#6b6a66"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def pct(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:+.{d}f}%"


def pctu(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:.{d}f}%"


def num(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _stats_rows(games: dict, keys: list, idx, bench=True):
    rows = []
    base = None
    for k in keys:
        g = games.get(k)
        if g is None or g.empty:
            continue
        g = g.loc[g.index.intersection(idx)]
        if g.empty:
            continue
        base = g if base is None else base
        s = game_stats(g)
        rows.append([LABEL.get(k, k), s["n"], pct(s["mean"]), pct(s["median"]), pctu(s["win"], 0),
                     num(s["sharpe0"]), num(s["sharpe"]), pct(s["worst"]), pct(s["p5"]), pct(s["cvar5"]),
                     pctu(s["p_lt_10"]), pctu(s.get("beat_spy", np.nan), 0), pct(s.get("excess_vs_spy", np.nan))])
    if bench and base is not None:
        for b in ("spy_tr", "sp500", "sti", "ew_universe"):
            if b in base and base[b].notna().sum() > 5:
                s = game_stats(base, b)
                rows.append([LABEL[b], s["n"], pct(s["mean"]), pct(s["median"]), pctu(s["win"], 0),
                             num(s["sharpe0"]), num(s["sharpe"]), pct(s["worst"]), pct(s["p5"]), pct(s["cvar5"]),
                             pctu(s["p_lt_10"]), "", ""])
    header = ["Strategy", "Games", "Mean 9-wk", "Median", "Win", "Sharpe (rf 0)", "Sharpe (T-bill)", "Worst",
              "5th pct", "CVaR 5%", "P(<-10%)", "Beat S&P", "Mean vs S&P"]
    return md_table(header, rows)


def _nav_block(state, keys, idx, phases):
    rows = []
    spy = state["spy"]
    for k in keys:
        paths = state["paths"].get(k)
        if not paths:
            continue
        sub = {i: p for i, p in paths.items() if p.index[0] >= idx.min() - pd.Timedelta(days=3) and p.index[0] <= idx.max() + pd.Timedelta(days=5)}
        stats = []
        for ph in phases:
            nav = chain_nav(sub, phase=ph)
            if len(nav) > 60:
                stats.append(nav_stats(nav, state["rf"], spy))
        if not stats:
            continue
        df = pd.DataFrame(stats)
        a = df.mean(numeric_only=True)
        rows.append([LABEL.get(k, k), len(stats), pct(a.get("cagr")), pctu(a.get("vol")), num(a.get("sharpe")),
                     num(a.get("sortino")), pct(a.get("max_dd")), pct(df["max_dd"].min()), num(a.get("beta")),
                     pct(a.get("jensen_alpha")), num(a.get("treynor"), 3), num(a.get("info_ratio")), pct(a.get("m2"))])
    header = ["Strategy", "Phases", "CAGR", "Volatility", "Sharpe", "Sortino", "Max drawdown (avg)",
              "Max drawdown (worst phase)", "Beta vs S&P", "Jensen alpha", "Treynor", "Info ratio", "M-squared"]
    return md_table(header, rows)


def _boot_rows(games, pairs, idx):
    rows = []
    for a, b in pairs:
        ga, gb = games.get(a), games.get(b)
        if ga is None or ga.empty:
            continue
        ra = ga.loc[ga.index.intersection(idx), "ret"]
        rb = gb.loc[gb.index.intersection(idx), "ret"] if b in games else ga.loc[ga.index.intersection(idx), b]
        r = block_bootstrap_diff(ra, rb)
        if not r:
            continue
        rows.append([f"{LABEL.get(a, a)} vs {LABEL.get(b, b)}", r["n"], pct(r["mean_diff"]),
                     f"[{pct(r['mean_ci'][0])}, {pct(r['mean_ci'][1])}]", "yes" if r["mean_sig"] else "no",
                     num(r["sharpe_diff"]), f"[{num(r['sharpe_ci'][0])}, {num(r['sharpe_ci'][1])}]",
                     "yes" if r["sharpe_sig"] else "no"])
    return md_table(["Comparison", "Games", "Mean difference", "95% CI", "Significant",
                     "Sharpe difference", "95% CI", "Significant"], rows)


def _by_year(games, keys, idx):
    frames = {}
    for k in keys:
        g = games.get(k)
        if g is None or g.empty:
            continue
        g = g.loc[g.index.intersection(idx)]
        frames[LABEL.get(k, k)] = g["ret"].groupby(g.index.year).mean()
    base = games.get(keys[0])
    if base is not None and not base.empty:
        b = base.loc[base.index.intersection(idx)]
        frames[LABEL["spy_tr"]] = b["spy_tr"].groupby(b.index.year).mean()
    df = pd.DataFrame(frames)
    rows = [[str(y)] + [pct(v) for v in r] for y, r in df.iterrows()]
    return md_table(["Year (game start)"] + list(df.columns), rows)


# ---------------------------------------------------------------------------- figures
def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig_nav(state, keys, idx, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    _style(ax)
    ends = []
    first_date = None
    for k in keys:
        paths = state["paths"].get(k)
        if not paths:
            continue
        sub = {i: p for i, p in paths.items() if idx.min() - pd.Timedelta(days=3) <= p.index[0] <= idx.max() + pd.Timedelta(days=5)}
        nav = chain_nav(sub, phase=0)
        if len(nav) < 30:
            continue
        first_date = nav.index[0] if first_date is None else min(first_date, nav.index[0])
        lw = 2.4 if k in ("FFQ", "FFQ_US") else 1.6
        ax.plot(nav.index, nav.values, color=COLOR.get(k, INK2), lw=lw, label=LABEL.get(k, k))
        ends.append((nav.index[-1], nav.values[-1], LABEL.get(k, k), COLOR.get(k, INK2)))
    spy = state["spy"]
    if spy is not None and first_date is not None:
        s = spy[spy.index >= first_date].dropna()
        if ends:
            s = s[s.index <= max(e[0] for e in ends)]
        s = s / s.iloc[0]
        ax.plot(s.index, s.values, color=COLOR["SPY"], lw=1.4, ls="--", label=LABEL["spy_tr"])
        ends.append((s.index[-1], s.values[-1], "S&P 500", COLOR["SPY"]))
    ax.set_yscale("log")
    ax.set_ylabel("Growth of USD 1 (log scale)", color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    ends.sort(key=lambda e: e[1])
    last_y = None
    for d, y, lab, col in ends:   # direct labels, nudged apart
        yy = y if last_y is None or y / last_y > 1.04 else last_y * 1.04
        ax.annotate(lab.split(":")[0].split(" (")[0], (d, yy), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=INK2, va="center")
        last_y = yy
    ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def _fig_drawdown(state, keys, idx, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3.8), facecolor=SURFACE)
    _style(ax)
    first_date, last_date = None, None
    for k in keys:
        paths = state["paths"].get(k)
        if not paths:
            continue
        sub = {i: p for i, p in paths.items() if idx.min() - pd.Timedelta(days=3) <= p.index[0] <= idx.max() + pd.Timedelta(days=5)}
        nav = chain_nav(sub, phase=0)
        if len(nav) < 30:
            continue
        first_date = nav.index[0] if first_date is None else min(first_date, nav.index[0])
        last_date = nav.index[-1] if last_date is None else max(last_date, nav.index[-1])
        dd = nav / nav.cummax() - 1
        ax.plot(dd.index, dd.values * 100, color=COLOR.get(k, INK2), lw=2.0 if k in ("FFQ", "FFQ_US") else 1.4,
                label=LABEL.get(k, k))
    spy = state["spy"]
    if spy is not None and first_date is not None:
        s = spy[(spy.index >= first_date) & (spy.index <= last_date)].dropna()
        dd = s / s.cummax() - 1
        ax.plot(dd.index, dd.values * 100, color=COLOR["SPY"], lw=1.2, ls="--", label=LABEL["spy_tr"])
    ax.set_ylabel("Drawdown from peak (%)", color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    ax.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK2, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def _fig_dist(games, keys, idx, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data, labels, cols = [], [], []
    for k in keys:
        g = games.get(k)
        if g is None or g.empty:
            continue
        r = g.loc[g.index.intersection(idx), "ret"].dropna() * 100
        if len(r) < 5:
            continue
        data.append(r.values)
        labels.append(LABEL.get(k, k))
        cols.append(COLOR.get(k, INK2))
    base = games.get(keys[0])
    if base is not None and not base.empty:
        b = base.loc[base.index.intersection(idx), "spy_tr"].dropna() * 100
        if len(b) >= 5:
            data.append(b.values)
            labels.append(LABEL["spy_tr"])
            cols.append(COLOR["SPY"])
    if not data:
        return
    fig, ax = plt.subplots(figsize=(10, 0.7 * len(data) + 1.4), facecolor=SURFACE)
    _style(ax)
    bp = ax.boxplot(data[::-1], vert=False, whis=(5, 95), widths=0.55, patch_artist=True,
                    showfliers=True, flierprops={"markersize": 3, "markerfacecolor": INK2, "markeredgecolor": "none", "alpha": 0.5},
                    medianprops={"color": INK, "lw": 1.6})
    for patch, c in zip(bp["boxes"], cols[::-1]):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
        patch.set_edgecolor(SURFACE)
    for j, d in enumerate(data[::-1], 1):
        ax.plot(np.mean(d), j, marker="D", color=INK, markersize=5)
    ax.set_yticks(range(1, len(data) + 1))
    ax.set_yticklabels(labels[::-1], fontsize=9, color=INK2)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("9-week game return (%). Box: 25th-75th pct; whiskers: 5th-95th; diamond: mean", color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------- report
def write_report(state: dict, out_dir: str) -> str:
    games = state["games"]
    figdir = os.path.join(out_dir, "figures")
    os.makedirs(figdir, exist_ok=True)
    cov = state["coverage"]
    full_idx = cov.index[cov >= 0.6]
    all_idx = games["LEGACY"].index if "LEGACY" in games else pd.DatetimeIndex([])
    phases = list(range(0, C.HOLD_DAYS, C.STEP_DAYS * (3 if state.get("quick") else 1)))
    L = []
    title = "FFQ strategy backtest"
    if state.get("synthetic"):
        title += " (SYNTHETIC SELF-TEST DATA: numbers mean nothing)"
    L.append(f"# {title}\n")
    L.append(f"Run {state['run']} · prices {state['dates'][0].date()} to {state['dates'][1].date()} · "
             f"{state['n_stocks']} stocks · games of {C.HOLD_DAYS} trading days starting every "
             f"{C.STEP_DAYS * (3 if state.get('quick') else 1)} days · commission 0.25% each way (min USD 25) · "
             f"-25% per-lot stop · cash earns 0% · all returns in USD.\n")
    if state.get("synthetic"):
        L.append("> **This report was produced from a synthetic market by the offline self-test.** "
                 "It shows that the code runs end to end. It says nothing about real stocks.\n")

    # ---- headline
    if "FFQ" in games and len(full_idx):
        f = games["FFQ"].loc[games["FFQ"].index.intersection(full_idx)]
        lg = games["LEGACY"].loc[games["LEGACY"].index.intersection(full_idx)]
        sf, sl = game_stats(f), game_stats(lg)
        q = state["quality"]
        qf = q[q["variant"] == "FFQ"].mean(numeric_only=True) if len(q) else pd.Series(dtype=float)
        ql = q[q["variant"] == "LEGACY"].mean(numeric_only=True) if len(q) else pd.Series(dtype=float)
        bt = block_bootstrap_diff(f["ret"], lg["ret"])
        L.append("## Headline (period with full point-in-time accounts)\n")
        L.append(md_table(["", "FFQ (new)", "Legacy momentum"], [
            ["Games", sf.get("n"), sl.get("n")],
            ["Mean 9-week return", pct(sf.get("mean")), pct(sl.get("mean"))],
            ["Sharpe ratio (annualised, rf 0)", num(sf.get("sharpe0")), num(sl.get("sharpe0"))],
            ["Worst game", pct(sf.get("worst")), pct(sl.get("worst"))],
            ["1 game in 20 worse than", pct(sf.get("p5")), pct(sl.get("p5"))],
            ["Holdings failing the fundamental gate", pctu(qf.get("share_fail_gate", np.nan), 0), pctu(ql.get("share_fail_gate", np.nan), 0)],
            ["Holdings with falling profits", pctu(qf.get("share_profit_falling", np.nan), 0), pctu(ql.get("share_profit_falling", np.nan), 0)],
            ["Holdings earning ROE below their SML cost of equity", pctu(qf.get("share_roe_below_ke", np.nan), 0), pctu(ql.get("share_roe_below_ke", np.nan), 0)],
        ]))
        if bt:
            L.append(f"\nPaired block bootstrap, FFQ minus legacy: mean {pct(bt['mean_diff'])} "
                     f"[{pct(bt['mean_ci'][0])}, {pct(bt['mean_ci'][1])}] "
                     f"({'significant' if bt['mean_sig'] else 'not significant'}); Sharpe {num(bt['sharpe_diff'])} "
                     f"[{num(bt['sharpe_ci'][0])}, {num(bt['sharpe_ci'][1])}] "
                     f"({'significant' if bt['sharpe_sig'] else 'not significant'}).\n")
        L.append(f"Full-accounts period: {len(full_idx)} game starts from {full_idx.min().date()} to "
                 f"{full_idx.max().date()} (a start counts once at least 60% of the universe has two "
                 f"published years of accounts).\n")
    else:
        L.append("## Headline\n\nNo period had point-in-time accounts for 60% of the universe, so the full "
                 "FFQ strategy could not be tested. See the long-history risk-engine results below.\n")

    # ---- full-accounts period
    keys_full = ["FFQ", "FFQ_EQUAL", "RISK_ENGINE", "LEGACY", "LEGACY_GATED", "LEGACY_LIQ"]
    if len(full_idx) and "FFQ" in games:
        L.append("## 1. All variants on identical games (full-accounts period)\n")
        L.append(_stats_rows(games, keys_full, full_idx))
        L.append("\nSharpe is the 9-week mean over its standard deviation, annualised by the square root of "
                 f"252/{C.HOLD_DAYS}; the rf-0 column matches the complete record's convention. "
                 "CVaR 5% is the average of the worst 5% of games.\n")
        L.append("### Chained portfolio (one game after another), averaged over start phases\n")
        L.append(_nav_block(state, keys_full, full_idx, phases))
        L.append("\n### Is the difference real? (paired moving-block bootstrap, blocks of 9 starts)\n")
        L.append(_boot_rows(games, [("FFQ", "LEGACY"), ("FFQ", "RISK_ENGINE"), ("FFQ", "FFQ_EQUAL"),
                                    ("LEGACY_GATED", "LEGACY"), ("FFQ", "spy_tr")], full_idx))
        L.append("\nRead: FFQ vs risk engine = the value of the fundamentals (gate plus quality/value score); "
                 "FFQ vs equal weight = the value of the covariance optimiser and capital allocation line; "
                 "legacy + gate vs legacy = what vetoing bad accounts does to the old screen.\n")
        q = state["quality"]
        if len(q):
            L.append("### 2. What the strategies actually owned (average across game starts)\n")
            rows = []
            for k in keys_full:
                qk = q[q["variant"] == k].mean(numeric_only=True)
                if qk.empty:
                    continue
                rows.append([LABEL.get(k, k), num(qk.get("n_held"), 1), pctu(qk.get("share_fail_gate"), 0),
                             pctu(qk.get("share_profit_falling"), 0), pctu(qk.get("share_roe_below_ke"), 0),
                             pctu(qk.get("avg_roe")), num(qk.get("avg_fscore"), 1), num(qk.get("avg_pe"), 1),
                             pctu(qk.get("avg_net_margin"))])
            L.append(md_table(["Strategy", "Names", "Fail gate", "Profit falling", "ROE < cost of equity",
                               "Avg ROE", "Avg F-score", "Median P/E", "Avg net margin"], rows))
            L.append("")
        bi = state["bookinfo"]
        if len(bi) and "FFQ" in games:
            g = games["FFQ"]
            paths = state["paths"]["FFQ"]
            rv = {g.index[j]: paths[i].pct_change().std() * np.sqrt(252) for j, i in enumerate(g["i"]) if i in paths}
            bi = bi.set_index("start")
            bi["vol_realised"] = pd.Series(rv)
            ok = bi[["vol_exante", "vol_realised"]].dropna()
            L.append("### 3. Risk management checks (FFQ)\n")
            L.append(md_table(["Measure", "Average", "Range"], [
                ["Names held", num(bi["n"].mean(), 1), f"{bi['n'].min():.0f} to {bi['n'].max():.0f}"],
                ["Share of capital invested (CAL)", pctu(bi["invested"].mean(), 0), f"{pctu(bi['invested'].min(), 0)} to {pctu(bi['invested'].max(), 0)}"],
                ["Ex-ante volatility (annual)", pctu(bi["vol_exante"].mean()), f"{pctu(bi['vol_exante'].min())} to {pctu(bi['vol_exante'].max())}"],
                ["Realised volatility in the game", pctu(ok["vol_realised"].mean()) if len(ok) else "n/a", ""],
                ["Realised / predicted volatility", num((ok["vol_realised"] / ok["vol_exante"]).mean()) if len(ok) else "n/a", "1.00 = perfectly calibrated"],
                ["Ex-ante beta to S&P 500", num(bi["beta_exante"].mean()), f"{num(bi['beta_exante'].min())} to {num(bi['beta_exante'].max())}"],
                ["Average pairwise correlation (weighted)", num(bi["avg_pair_corr"].mean()), ""],
                ["Highest pairwise correlation in the book", num(bi["max_pair_corr"].mean()), f"max {num(bi['max_pair_corr'].max())} (limit {C.CORR_MAX_PAIR})"],
                ["Effective number of holdings", num(bi["effective_n"].mean(), 1), ""],
                ["Diversification ratio", num(bi["div_ratio"].mean()), "weighted stock vol / portfolio vol"],
                ["Stocks passing the gate (of universe)", f"{bi['n_eligible'].mean():.0f} of {bi['n_universe'].mean():.0f}", ""],
            ]))
            L.append("")
        L.append("### 4. By year\n")
        L.append(_by_year(games, ["FFQ", "RISK_ENGINE", "LEGACY", "LEGACY_GATED"], full_idx))
        L.append("")
        _fig_nav(state, ["FFQ", "RISK_ENGINE", "LEGACY", "LEGACY_GATED"], full_idx,
                 os.path.join(figdir, "nav_full.png"), "Chained 9-week games, full-accounts period")
        _fig_drawdown(state, ["FFQ", "LEGACY"], full_idx, os.path.join(figdir, "drawdown_full.png"),
                      "Drawdown, full-accounts period")
        _fig_dist(games, keys_full, full_idx, os.path.join(figdir, "games_full.png"),
                  "Distribution of 9-week game returns, full-accounts period")
        L.append("![Chained games](figures/nav_full.png)\n")
        L.append("![Drawdown](figures/drawdown_full.png)\n")
        L.append("![Game distribution](figures/games_full.png)\n")

    # ---- long history
    keys_long = ["RISK_ENGINE", "LEGACY", "LEGACY_LIQ"]
    if len(all_idx):
        L.append(f"## 5. Long history, price-only components ({all_idx.min().date()} to {all_idx.max().date()})\n")
        L.append("Before about 2024 Yahoo has no point-in-time accounts, so this section tests only the parts "
                 "that need prices: FF3 residual momentum, the covariance model, the correlation gate, the "
                 "max-Sharpe optimiser and the capital allocation line.\n")
        L.append("The legacy rows are the momentum rulebook rebuilt on this data, with commission and the -25% "
                 "stop applied to every variant. The complete record's +5.06% 'selection per dollar' figure "
                 "had no stop, so expect the legacy mean here to be somewhat lower.\n")
        L.append(_stats_rows(games, keys_long, all_idx))
        L.append("\n")
        L.append(_nav_block(state, keys_long, all_idx, phases))
        L.append("\n")
        L.append(_boot_rows(games, [("RISK_ENGINE", "LEGACY"), ("RISK_ENGINE", "LEGACY_LIQ"),
                                    ("RISK_ENGINE", "spy_tr")], all_idx))
        L.append("\n")
        L.append(_by_year(games, keys_long, all_idx))
        L.append("")
        _fig_nav(state, keys_long, all_idx, os.path.join(figdir, "nav_long.png"),
                 "Chained 9-week games, long history (price-only components)")
        _fig_drawdown(state, ["RISK_ENGINE", "LEGACY"], all_idx, os.path.join(figdir, "drawdown_long.png"),
                      "Drawdown, long history")
        L.append("![Long history](figures/nav_long.png)\n")
        L.append("![Long drawdown](figures/drawdown_long.png)\n")

    # ---- SEC US-only test
    if "FFQ_US" in games and not games["FFQ_US"].empty:
        us_idx = games["FFQ_US"].index
        L.append(f"## 6. US only with SEC EDGAR accounts ({us_idx.min().date()} to {us_idx.max().date()})\n")
        L.append("Every fiscal year is used from its original 10-K filing date, as first reported. This is the "
                 "only multi-regime test of the fundamental gate that free data allows.\n")
        keys_us = ["FFQ_US", "RISK_ENGINE_US", "MOM_US"]
        L.append(_stats_rows(games, keys_us, us_idx))
        L.append("\n")
        L.append(_nav_block(state, keys_us, us_idx, phases))
        L.append("\n")
        L.append(_boot_rows(games, [("FFQ_US", "RISK_ENGINE_US"), ("FFQ_US", "MOM_US"), ("FFQ_US", "spy_tr")], us_idx))
        L.append("\n")
        L.append(_by_year(games, keys_us, us_idx))
        L.append("")
        _fig_nav(state, keys_us, us_idx, os.path.join(figdir, "nav_us_sec.png"),
                 "US only, SEC EDGAR accounts: chained 9-week games")
        L.append("![US SEC test](figures/nav_us_sec.png)\n")

    # ---- caveats
    L.append("## 7. Limits of this backtest\n")
    L.append("\n".join([
        "- **Short fundamentals window.** Yahoo serves only the last four annual reports, so the full strategy "
        "(with the gate) can only be tested from about 2024. Use `--sec` for a 2015-2026 US-only test.",
        "- **Survivorship.** The universe is today's listed companies; delisted names are invisible, so absolute "
        "returns are overstated for every variant. Compare strategies with each other, not with zero.",
        "- **Factor proxies.** FF3 factors are built from ETFs (IWM-IWB, IWD-IWF, SCZ-EFA, EFV-EFG), not from "
        "Kenneth French's data library, so they are close to but not identical to the academic factors.",
        f"- **Accounts lag.** A fiscal year is used {C.FUND_LAG_DAYS} days after its year end (Yahoo) or from its "
        "filing date (SEC). Restated Yahoo figures may differ slightly from what was first published.",
        "- **Fills.** Buys at the next day's open (Yahoo adjusted), no bid-ask spread or market impact.",
        "- **Rules were fixed before this backtest was run** (ffq/config.py, 24 Sep 2026) and were not tuned "
        "to it. If they are changed after seeing these results, the results become in-sample.",
    ]))
    notes = state.get("notes") or []
    if notes:
        L.append("\n## 8. Data notes\n")
        L.append("\n".join(f"- {n}" for n in notes[:80]))
    path = os.path.join(out_dir, "BACKTEST_REPORT.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return path
