"""
FFQ weekly screen: the book to hold, why each stock is in it, and how risky it is.

    python screener.py                                   # book for a USD 1,000,000 account
    python screener.py --holdings holdings.csv           # plus orders from what you hold now
    python screener.py --deploy 0.66                     # invest only 66% of the account this round
    python screener.py --exclude CAT 558.SI              # red-flag vetoes (re-runs without them)
    python screener.py --asof 2026-06-30                 # the screen as it would have looked then
    python screener.py --synthetic                       # offline self-test

Writes reports/SCREEN_<date>.md (+ .csv and figures). Every number comes from
Yahoo Finance; verify prices and accounts on Bloomberg before trading.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

from ffq import config as C
from ffq.legacy import legacy_weights
from ffq.pipeline import load
from ffq.strategy import FFQStrategy

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def bbg(tk: str) -> str:
    return tk.replace(".SI", "") + " SP Equity" if tk.endswith(".SI") else tk.replace("-", "/") + " US Equity"


def f_pct(x, d=1, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def f_num(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{d}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def name_of(m, tk):
    inf = m.info.get(tk, {})
    return inf.get("shortName") or inf.get("longName") or tk


def leverage_text(tk, r):
    if tk in C.BANK_LIKE:
        return f"equity/assets {f_pct(r.get('eq_assets'))}"
    if tk in C.REIT_LIKE:
        return f"gearing {f_pct(r.get('gearing'))}"
    return f"debt/equity {f_num(r.get('de'))}"


def thesis(m, tk, r, w, rc, maxc, maxc_name) -> str:
    """One paragraph a group member can read out in the weekly recording."""
    ctry = "Singapore" if tk.endswith(".SI") else "United States"
    g_basis = r.get("growth_basis") or "latest period"
    parts = [
        f"**{name_of(m, tk)} ({bbg(tk).replace(' Equity', '')})**, {r.get('sector')}, {ctry}.",
        f"Accounts: {r.get('basis')}.",
        f"Return on equity {f_pct(r.get('roe'))} against a security-market-line cost of equity of "
        f"{f_pct(r.get('k_e'))} (adjusted beta {f_num(r.get('beta_adj'))}), so it earns "
        f"{f_pct((r.get('roe') or np.nan) - (r.get('k_e') or np.nan), sign=True)} a year above what shareholders require.",
        f"Profit {f_pct(r.get('ni_growth'), sign=True)} and revenue {f_pct(r.get('rev_growth'), sign=True)} ({g_basis}); "
        f"Piotroski F-score {f_num(r.get('fscore'), 1)}/9; {leverage_text(tk, r)}.",
        f"Valuation: P/E {f_num(r.get('pe'), 1)}, earnings yield {f_pct(r.get('ep'))}, book/price {f_num(r.get('bp'))}.",
        f"After allowing for its Fama-French market ({f_num(r.get('ff_mkt'))}), size ({f_num(r.get('ff_smb'))}) and "
        f"value ({f_num(r.get('ff_hml'))}) exposures, it returned {f_pct(r.get('ff_alpha_form_ann'), sign=True)} a year "
        f"above its factor benchmark over the 11 months to one month ago (appraisal ratio {f_num(r.get('ff_mom'))}).",
        f"Weight {f_pct(w)} of the account, {f_pct(rc)} of portfolio risk; its highest correlation with another "
        f"holding is {f_num(maxc)} ({maxc_name}).",
    ]
    if r.get("flags"):
        parts.append(f"Check before trading: {r['flags']}.")
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=C.ACCOUNT_USD, help="account value in USD")
    ap.add_argument("--deploy", type=float, default=1.0, help="share of the account to invest now")
    ap.add_argument("--holdings", help="CSV with columns ticker,shares (current positions)")
    ap.add_argument("--exclude", nargs="*", default=[], help="tickers vetoed by the red-flag check")
    ap.add_argument("--asof", help="run the screen as of this date (YYYY-MM-DD)")
    ap.add_argument("--sec", action="store_true", help="use SEC EDGAR accounts for US names")
    ap.add_argument("--synthetic", action="store_true", help="offline self-test")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    provider = None
    if args.synthetic:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from tests.synthetic import SyntheticProvider
        provider = SyntheticProvider()
    start = (pd.Timestamp(args.asof or dt.date.today()) - pd.Timedelta(days=365 * 4 + 60)).strftime("%Y-%m-%d")
    m, fe, acc, source, notes = load(provider, start=start, use_sec=args.sec)
    date = pd.Timestamp(args.asof) if args.asof else m.dates[-1]
    i = m.loc(date)
    date = m.dates[i]
    strat = FFQStrategy(m, fe, acc, exclude=args.exclude)
    book = strat.book(date)
    tab, w, risk = book.table, book.weights, book.risk
    rf = float(m.rf_annual.iloc[i])
    fx_sgd = float(m.fx["SGD"].iloc[i]) if m.fx.get("SGD") is not None else np.nan
    last_sg = max((m.px[t].last_valid_index() for t in m.tickers if t.endswith(".SI")), default=None)
    last_us = max((m.px[t].last_valid_index() for t in m.tickers if not t.endswith(".SI")), default=None)
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)
    tag = date.strftime("%Y-%m-%d")
    L = []
    title = f"FFQ weekly screen, {tag}"
    if args.synthetic:
        title += " (SYNTHETIC SELF-TEST: not real stocks)"
    L.append(f"# {title}\n")
    L.append(f"Run {dt.datetime.now():%Y-%m-%d %H:%M} · latest closes SG {last_sg.date() if last_sg is not None else 'n/a'}, "
             f"US {last_us.date() if last_us is not None else 'n/a'} · USD per SGD {f_num(fx_sgd, 4)} · "
             f"T-bill {f_pct(rf, 2)} · market risk premium {f_pct(C.MRP)} · account USD {args.capital:,.0f}, "
             f"deploying {args.deploy:.0%}.\n")
    L.append(f"Universe {book.diag.get('n_universe')} stocks; **{book.diag.get('n_eligible')} pass the fundamental "
             f"gate**; {book.diag.get('n_candidates')} candidates after ranking; {len(w)} held.\n")

    # ---------------------------------------------------------------- the book
    budget = args.capital * args.deploy
    rows, csv = [], []
    rc = risk.get("risk_contrib", pd.Series(dtype=float))
    from ffq.risk import covariance
    Sigma, cinfo = covariance(fe, list(w.index), date) if len(w) else (pd.DataFrame(), {})
    corr = cinfo.get("lw_corr", pd.DataFrame())
    for tk in w.sort_values(ascending=False).index:
        r = tab.loc[tk]
        px_local = float(m.raw_local[tk].iloc[i])
        fxq = float(m.fx[m.quote_ccy[tk]].iloc[i]) if m.fx.get(m.quote_ccy[tk]) is not None else 1.0
        usd = w[tk] * budget
        sh = usd / (px_local * fxq)
        sh = int(sh // 100 * 100) if tk.endswith(".SI") else int(sh)
        cost = sh * px_local * fxq
        others = [o for o in w.index if o != tk and o in corr.index]
        maxc = float(corr.loc[tk, others].max()) if others and tk in corr.index else np.nan
        maxn = corr.loc[tk, others].idxmax() if others and tk in corr.index else "-"
        rows.append([tk, bbg(tk), name_of(m, tk)[:28], f_pct(w[tk]), f"{cost:,.0f}", f"{sh:,}",
                     f"{m.quote_ccy[tk]} {px_local:,.3f}", f"{px_local * C.STOP:,.3f}", f_pct(rc.get(tk)),
                     f"{f_num(maxc)} ({maxn})"])
        csv.append({"ticker": tk, "bloomberg": bbg(tk), "name": name_of(m, tk), "weight": w[tk], "usd": cost,
                    "shares": sh, "price_local": px_local, "currency": m.quote_ccy[tk],
                    "stop_local": px_local * C.STOP, "risk_contribution": rc.get(tk), "max_corr": maxc,
                    **{k: r.get(k) for k in ("roe", "k_e", "ni_growth", "rev_growth", "fscore", "pe", "ep", "bp",
                                             "de", "gearing", "ff_mkt", "ff_smb", "ff_hml", "ff_alpha_form_ann",
                                             "ff_mom", "mom_12_1", "mu", "basis")}})
    L.append("## 1. The book\n")
    L.append(md_table(["Ticker", "Bloomberg", "Company", "Weight", "Cost USD", "Shares", "Last close",
                       "Stop (x0.75)", "Risk share", "Max corr (with)"], rows))
    cash = args.capital - sum(c["usd"] for c in csv)
    L.append(f"\nInvested USD {sum(c['usd'] for c in csv):,.0f}; cash USD {cash:,.0f}. "
             f"Capital allocation line: book volatility {f_pct(book.diag.get('book_vol'))} vs target "
             f"{f_pct(C.SIGMA_TARGET)}, so {f_pct(book.diag.get('cal_y'), 0)} of the risky book is held. "
             f"Weighting method: {book.diag.get('method')}. {'; '.join(book.diag.get('notes', []))}\n")
    L.append("Share counts use the latest close. Recalculate at the live price; the real stop is 0.75 x your fill.\n")

    # ---------------------------------------------------------------- justification
    L.append("## 2. Why each stock is in the book\n")
    jrows = []
    for tk in w.sort_values(ascending=False).index:
        r = tab.loc[tk]
        spread = (r.get("roe") or np.nan) - (r.get("k_e") or np.nan)
        jrows.append([tk, r.get("basis"), f_pct(r.get("roe")), f_pct(r.get("k_e")), f_pct(spread, sign=True),
                      f_pct(r.get("ni_growth"), sign=True), f_num(r.get("fscore"), 1), leverage_text(tk, r),
                      f_num(r.get("pe"), 1), f_num(r.get("ff_mkt")), f_num(r.get("ff_smb")), f_num(r.get("ff_hml")),
                      f_pct(r.get("ff_alpha_form_ann"), sign=True), f_num(r.get("ff_mom")), f_pct(r.get("mom_12_1"), 0, True),
                      f_pct(r.get("mu"))])
    L.append(md_table(["Ticker", "Accounts", "ROE", "Cost of equity (SML)", "ROE - k", "Profit growth",
                       "F-score", "Leverage", "P/E", "FF beta MKT", "SMB", "HML", "FF alpha (11m, ann.)",
                       "Appraisal ratio", "12-1 mom.", "Expected return"], jrows))
    L.append("\nExpected return = SML required return + alpha, alpha = 0.05 x residual volatility x score "
             "(Grinold-Kahn). Growth compares the latest trailing twelve months with a year earlier where the "
             "data allow; otherwise the latest quarter with the same quarter last year, otherwise fiscal years.\n")
    for tk in w.sort_values(ascending=False).index:
        others = [o for o in w.index if o != tk and o in corr.index]
        maxc = float(corr.loc[tk, others].max()) if others and tk in corr.index else np.nan
        maxn = corr.loc[tk, others].idxmax() if others and tk in corr.index else "-"
        L.append("- " + thesis(m, tk, tab.loc[tk], w[tk], rc.get(tk, np.nan), maxc, maxn))
    L.append("")

    # ---------------------------------------------------------------- analyst view (information only)
    arow = []
    for tk in w.index:
        inf = m.info.get(tk, {})
        tgt, px = inf.get("targetMeanPrice"), float(m.raw_local[tk].iloc[i])
        if tgt:
            up = tgt / px - 1
            dy = tab.loc[tk].get("div_yield")
            exp_r = up + (dy if dy is not None and np.isfinite(dy) else 0.0)
            arow.append([tk, f"{tgt:,.2f}", f_pct(up, sign=True), f_pct(dy), f_pct(tab.loc[tk].get("k_e")),
                         f_pct(exp_r - tab.loc[tk].get("k_e"), sign=True), inf.get("numberOfAnalystOpinions", "")])
    if arow:
        L.append("### Analysts' view against the SML (information only, not a rule)\n")
        L.append(md_table(["Ticker", "Mean target", "Upside", "Dividend yield", "Required return (SML)",
                           "Alpha implied by analysts", "Analysts"], arow))
        L.append("\nA positive implied alpha means the stock plots above the security market line on analysts' "
                 "12-month forecasts. Targets are not point-in-time, so this is not part of the tested rules.\n")

    # ---------------------------------------------------------------- portfolio risk
    L.append("## 3. Portfolio risk\n")
    L.append(md_table(["Measure", "Value", "What it means"], [
        ["Expected volatility (annual)", f_pct(risk.get("vol_ann")), f"target {f_pct(C.SIGMA_TARGET)}"],
        ["Beta to the S&P 500", f_num(risk.get("beta_spy")), f"limit {C.BETA_SPY_MAX:.2f}"],
        ["Expected return (annual, SML + alpha)", f_pct(risk.get("exp_return_ann")), ""],
        ["Ex-ante Sharpe ratio", f_num(risk.get("sharpe_ex_ante")), "model estimate, not a promise"],
        ["9-week VaR 95%", f_pct(risk.get("var95_9wk")), "1 game in 20 loses more than this (normal model)"],
        ["9-week CVaR 95%", f_pct(risk.get("cvar95_9wk")), "average loss in that worst 5%"],
        ["Diversification ratio", f_num(risk.get("diversification_ratio")), "weighted stock vol / portfolio vol"],
        ["Effective number of holdings", f_num(risk.get("effective_n"), 1), "1 / sum of squared weights"],
        ["Average pairwise correlation", f_num(risk.get("avg_pair_corr")), "weighted by position sizes"],
        ["Highest pairwise correlation", f_num(risk.get("max_pair_corr")), f"limit {C.CORR_MAX_PAIR}"],
        ["Share of risk from common factors", f_pct(risk.get("factor_risk_share")), "rest is stock-specific"],
    ]))
    expo = risk.get("apt_exposure")
    if expo is not None:
        desc = {"MKT_US": "S&P 500 beyond T-bills", "MKT_SG": "Singapore market (EWS)", "SMB_US": "US small minus big",
                "HML_US": "US value minus growth", "RATES": "7-10y Treasuries (a rise = yields fall)",
                "OIL": "oil (USO)", "GOLD_F": "gold (GLD)", "USD": "US dollar index (UUP)"}
        L.append("\n**APT macro-factor exposures of the book** (return of the book per 1% factor move):\n")
        L.append(md_table(["Factor", "Exposure", "Meaning"], [[k, f_num(v), desc.get(k, "")] for k, v in expo.items()]))
    L.append("")

    # correlation heat map and risk budget figures
    if len(w) >= 2 and not corr.empty:
        try:
            _fig_corr(corr.loc[w.index, w.index], os.path.join(args.out, "figures", f"corr_{tag}.png"), tag)
            _fig_risk(w, rc, os.path.join(args.out, "figures", f"risk_{tag}.png"), tag)
            L.append(f"![Correlation matrix](figures/corr_{tag}.png)\n")
            L.append(f"![Weight vs risk](figures/risk_{tag}.png)\n")
        except Exception as exc:  # figures are optional
            notes.append(f"figure failed: {exc}")

    # ---------------------------------------------------------------- why not
    L.append("## 4. What the old momentum screen would buy, and why FFQ does not\n")
    lw = legacy_weights(m, date, "legacy")
    leg_names = list(lw.index)
    extra = [t for t in leg_names if t not in tab.index]
    lt = strat.assess(extra, i) if extra else pd.DataFrame()
    allt = pd.concat([tab, lt]) if len(lt) else tab
    skip_why = {t: why for t, act, why in book.log if act == "skip"}
    n_elig = int(tab["eligible"].sum()) if "eligible" in tab else 0
    wr = []
    for tk in leg_names:
        r = allt.loc[tk]
        if tk in w.index:
            status, why = "held", ""
        elif bool(r.get("gate_pass")):
            status = "passes gate, not held"
            if tk in skip_why:
                why = skip_why[tk]
            elif pd.notna(r.get("score")) and r.get("score") < C.SCORE_MIN:
                why = f"below-average combined score ({f_num(r.get('score'))}) among the {n_elig} that pass"
            elif pd.notna(r.get("rank")):
                why = f"ranked {int(r.get('rank'))} of {n_elig} by combined score; the optimiser gave it under {C.W_MIN:.0%}"
            else:
                why = "no factor-model estimate (short price history)"
        else:
            status, why = "FAILS gate", r.get("gate_fails") or ""
        wr.append([tk, f_pct(r.get("mom_12_1"), 0, True), status, why])
    L.append(md_table(["Legacy pick", "12-1 momentum", "FFQ status", "Reason"], wr))
    top = tab[tab["mom_12_1"].notna()].sort_values("mom_12_1", ascending=False).head(20)
    fails = [[tk, f_pct(r["mom_12_1"], 0, True), r.get("gate_fails")] for tk, r in top.iterrows()
             if not bool(r.get("gate_pass"))]
    if fails:
        L.append("\nStrongest 12-1 momentum names in the universe that fail the fundamental gate:\n")
        L.append(md_table(["Ticker", "12-1 momentum", "Why it fails"], fails))
    skipped = [(t, why) for t, act, why in book.log if act == "skip"]
    if skipped:
        L.append("\nPassed the gate and ranked high, but skipped for risk reasons:\n")
        L.append(md_table(["Ticker", "Reason"], [[t, why] for t, why in skipped]))
    L.append("")

    # ---------------------------------------------------------------- orders
    if args.holdings:
        hold = pd.read_csv(args.holdings)
        hold["ticker"] = hold["ticker"].str.strip()
        cur = hold.groupby("ticker")["shares"].sum()
        ranks = tab["score"].rank(ascending=False)
        orows = []
        for tk in sorted(set(cur.index) | set(w.index)):
            if tk not in m.raw_local.columns:
                orows.append([tk, cur.get(tk, 0), "", "", "", "no price data"])
                continue
            px_local = float(m.raw_local[tk].iloc[i])
            fxq = float(m.fx[m.quote_ccy[tk]].iloc[i]) if m.fx.get(m.quote_ccy[tk]) is not None else 1.0
            tgt_sh = next((c["shares"] for c in csv if c["ticker"] == tk), 0)
            have = int(cur.get(tk, 0))
            r = allt.loc[tk] if tk in allt.index else strat.assess([tk], i).iloc[0]
            if tk not in w.index and have > 0:
                if bool(r.get("gate_pass")) and ranks.get(tk, 999) <= 2 * C.N_MAX:
                    orows.append([tk, have, have, 0, "", "hold: passes gate, still ranked in top 24 (no new money)"])
                    continue
                reason = r.get("gate_fails") or "no longer ranked"
                orows.append([tk, have, 0, -have, f"{-have * px_local * fxq:,.0f}", f"SELL: {reason}"])
                continue
            d = tgt_sh - have
            val = d * px_local * fxq
            if abs(val) < max(2000.0, 0.01 * args.capital):
                orows.append([tk, have, tgt_sh, 0, "", "no trade (inside the 1% band)"])
            else:
                orows.append([tk, have, tgt_sh, d, f"{val:,.0f}", "BUY" if d > 0 else "TRIM"])
        L.append("## 5. Orders from current holdings\n")
        L.append(md_table(["Ticker", "Held", "Target", "Trade (shares)", "Trade USD", "Action"], orows))
        L.append("\nSells are of stocks that now fail the fundamental gate or have dropped out of the top 24. "
                 "Trades smaller than 1% of the account are skipped to save commission (0.25% each way).\n")

    # ---------------------------------------------------------------- checks
    L.append("## 6. Before you trade\n")
    L.append("\n".join([
        "- Confirm each company name, price (within 10% of the close above) and currency on Bloomberg (`DES`, `GP`).",
        "- Red-flag check on `CN`: fraud, restatement, auditor resignation, regulatory action, suspension, or a "
        "guidance cut with 10+ points underperformance. If found, re-run with `--exclude TICKER`.",
        "- Earnings inside the game: look up `ERN`; default is hold (earnings protocol, rulebook section 19).",
        "- Verify the accounts figures above against Bloomberg `FA`; Yahoo summary fields are unreliable (error E20).",
    ]))
    ern = []
    for tk in w.index:
        ts = m.info.get(tk, {}).get("earningsTimestamp") or m.info.get(tk, {}).get("earningsTimestampStart")
        if ts:
            try:
                d = pd.Timestamp(int(float(ts)), unit="s").date()
                if d >= date.date():
                    ern.append([tk, str(d)])
            except Exception:
                pass
    if ern:
        L.append("\nNext results dates reported by Yahoo:\n")
        L.append(md_table(["Ticker", "Results date"], ern))
    if notes:
        L.append("\n### Data notes\n")
        L.append("\n".join(f"- {n}" for n in notes[:60]))
    path = os.path.join(args.out, f"SCREEN_{tag}.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    pd.DataFrame(csv).to_csv(os.path.join(args.out, f"SCREEN_{tag}.csv"), index=False)
    full = tab.drop(columns=[c for c in tab.columns if c.startswith("fscore_signals")], errors="ignore")
    full.to_csv(os.path.join(args.out, f"SCREEN_{tag}_all_stocks.csv"))
    print("\n".join(L[:4]))
    print(md_table(["Ticker", "Bloomberg", "Company", "Weight", "Cost USD", "Shares", "Last close", "Stop (x0.75)",
                    "Risk share", "Max corr (with)"], rows))
    print(f"\nFull screen written to {path}")


def _fig_corr(corr: pd.DataFrame, path: str, tag: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    n = len(corr)
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * n, 0.8 + 0.55 * n), facecolor=SURFACE)
    ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8, color=INK2)
    ax.set_yticklabels(corr.index, fontsize=8, color=INK2)
    for a in range(n):
        for b in range(n):
            ax.text(b, a, f"{corr.values[a, b]:.2f}", ha="center", va="center", fontsize=7, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Correlation of weekly USD returns, book of {tag}\n(blue negative, grey zero, red positive)",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def _fig_risk(w: pd.Series, rc: pd.Series, path: str, tag: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(w.sort_values().index)
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(names) + 1.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh(y + 0.2, w[names].values / w.sum() * 100, height=0.38, color="#2a78d6",
            label="Share of money invested (%)")
    ax.barh(y - 0.2, rc.reindex(names).values * 100, height=0.38, color="#eb6834",
            label="Share of portfolio risk (%)")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8, color=INK2)
    ax.grid(True, axis="x", color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    ax.set_title(f"Where the money is and where the risk is, {tag}", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
