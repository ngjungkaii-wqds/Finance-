"""
Point-in-time fundamentals, the Piotroski F-score, and the fundamental gate.

A fiscal year's accounts are only used once they were public: Yahoo accounts
from fiscal year end + FUND_LAG_DAYS, SEC EDGAR accounts from their filing date.
Everything here is computed from the annual income statement, balance sheet
and cash-flow statement, so the same code serves the live screen and the backtest.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C

# Yahoo field names, in order of preference
FIELDS = {
    "revenue": ["TotalRevenue", "OperatingRevenue"],
    "cogs": ["CostOfRevenue", "ReconciledCostOfRevenue"],
    "gross_profit": ["GrossProfit"],
    "net_income": ["NetIncomeCommonStockholders", "NetIncome",
                   "NetIncomeFromContinuingOperationNetMinorityInterest",
                   "NetIncomeContinuousOperations"],
    "eps": ["DilutedEPS", "BasicEPS"],
    "shares_avg": ["DilutedAverageShares", "BasicAverageShares"],
    "ocf": ["OperatingCashFlow", "CashFlowFromContinuingOperatingActivities"],
    "capex": ["CapitalExpenditure", "PurchaseOfPPE"],
    "fcf": ["FreeCashFlow"],
    "dividends_paid": ["CashDividendsPaid", "CommonStockDividendPaid"],
    "equity": ["StockholdersEquity", "CommonStockEquity", "TotalEquityGrossMinorityInterest"],
    "assets": ["TotalAssets"],
    "debt": ["TotalDebt"],
    "lt_debt": ["LongTermDebt", "LongTermDebtAndCapitalLeaseObligation"],
    "current_debt": ["CurrentDebt", "CurrentDebtAndCapitalLeaseObligation"],
    "current_assets": ["CurrentAssets"],
    "current_liabilities": ["CurrentLiabilities"],
    "shares": ["OrdinarySharesNumber", "ShareIssued"],
    "cash": ["CashAndCashEquivalents", "CashCashEquivalentsAndShortTermInvestments"],
}


def _num(x) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _pick(table: pd.DataFrame, col, names) -> float:
    if table is None or table.empty or col not in table.columns:
        return np.nan
    for n in names:
        if n in table.index:
            v = _num(table.at[n, col])
            if not np.isnan(v):
                return v
    return np.nan


@dataclass
class Period:
    end: pd.Timestamp       # fiscal period end
    avail: pd.Timestamp     # first date the accounts may be used
    v: dict                 # raw fields for the year


FLOW_KEYS = ["revenue", "cogs", "gross_profit", "net_income", "ocf", "capex", "fcf", "dividends_paid"]
STOCK_KEYS = ["equity", "assets", "debt", "lt_debt", "current_debt", "current_assets",
              "current_liabilities", "shares", "cash"]


def _finish(v: dict) -> dict:
    """Derived fields shared by every source."""
    if np.isnan(v["gross_profit"]) and not np.isnan(v["revenue"]) and not np.isnan(v["cogs"]):
        v["gross_profit"] = v["revenue"] - v["cogs"]
    if np.isnan(v["fcf"]) and not np.isnan(v["ocf"]) and not np.isnan(v["capex"]):
        v["fcf"] = v["ocf"] + v["capex"] if v["capex"] <= 0 else v["ocf"] - v["capex"]
    if np.isnan(v["debt"]):
        parts = [x for x in (v["lt_debt"], v["current_debt"]) if not np.isnan(x)]
        v["debt"] = sum(parts) if parts else np.nan
    return v


def periods_from_tables(stmts: dict, lag_days: int = C.FUND_LAG_DAYS,
                        avail_override: dict | None = None, prefix: str = "") -> list[Period]:
    """Merge Yahoo's three tables into one record per period, oldest first.

    prefix "" reads the annual tables, "q_" the quarterly (or half-yearly) ones.
    """
    inc, bal, cf = (stmts.get(prefix + k, pd.DataFrame()) for k in ("income", "balance", "cashflow"))
    ends = set()
    for t in (inc, bal, cf):
        if t is not None and not t.empty:
            ends.update(pd.to_datetime(t.columns))
    out = []
    for end in sorted(ends):
        v = {}
        for key, names in FIELDS.items():
            val = np.nan
            for t in (inc, bal, cf):
                if t is None or t.empty:
                    continue
                cols = [c for c in t.columns if pd.Timestamp(c) == end]
                if cols:
                    val = _pick(t, cols[0], names)
                    if not np.isnan(val):
                        break
            v[key] = val
        v = _finish(v)
        # a period with neither profit nor equity is an empty column Yahoo sometimes returns
        if np.isnan(v["net_income"]) and np.isnan(v["equity"]):
            continue
        avail = (avail_override or {}).get(end, end + pd.Timedelta(days=lag_days))
        out.append(Period(end=pd.Timestamp(end), avail=pd.Timestamp(avail), v=v))
    return out


def ttm_from_quarters(qper: list[Period]) -> list[Period]:
    """Trailing-twelve-month records built from consecutive quarters or half-years.

    Flows (revenue, profit, cash flow) are summed over the periods covering the last
    ~365 days; balance-sheet items are taken at the latest period end.
    """
    q = sorted(qper, key=lambda p: p.end)
    if len(q) < 2:
        return []
    spans = [np.nan] + [(q[k].end - q[k - 1].end).days for k in range(1, len(q))]
    ok_spans = [s for s in spans[1:] if 80 <= s <= 100 or 170 <= s <= 195]
    if ok_spans and len(set(round(s / 91) for s in ok_spans)) == 1:
        spans[0] = int(np.median(ok_spans))       # the oldest column is the same kind of period
    out = []
    for k in range(len(q) - 1, -1, -1):
        total, comps, j = 0, [], k
        while j >= 0 and total < 350:
            s = spans[j]
            if not (80 <= s <= 100 or 170 <= s <= 195):
                break
            total += s
            comps.append(q[j])
            j -= 1
        if not 350 <= total <= 380:
            continue
        v = {}
        for key in FLOW_KEYS:
            vals = [c.v.get(key, np.nan) for c in comps]
            v[key] = float(np.sum(vals)) if not any(np.isnan(x) for x in vals) else np.nan
        for key in STOCK_KEYS:
            v[key] = q[k].v.get(key, np.nan)
        v["eps"] = np.nan
        v["shares_avg"] = q[k].v.get("shares_avg", np.nan)
        v["n_parts"] = len(comps)
        out.append(Period(end=q[k].end, avail=q[k].avail, v=_finish({**{f: np.nan for f in FIELDS}, **v})))
    return sorted(out, key=lambda p: p.end)


def fix_share_basis(acc: "Accounts", info: dict, price_local_now: float) -> list[str]:
    """Check the latest share count against Yahoo's quoted market value.

    If they differ by a stock-split ratio (accounts on the old share basis), restate
    every period's share count and EPS to today's basis so market values are right.
    """
    mc = (info or {}).get("marketCap")
    recs = acc.annual + acc.quarters + acc.ttm
    if not acc.annual or not mc or not price_local_now or price_local_now <= 0:
        return []
    latest = max(recs, key=lambda p: p.end)
    v = latest.v if not np.isnan(latest.v.get("shares", np.nan)) else acc.annual[-1].v
    sh = v["shares"] if not np.isnan(v["shares"]) else v["shares_avg"]
    if np.isnan(sh) or sh <= 0:
        return []
    r = sh * price_local_now / float(mc)
    if 0.6 < r < 1.6:
        return []
    # off by a split ratio (old share basis) or by a share-class convention (e.g. A vs B
    # shares): restate every period to the basis of Yahoo's quoted market value
    for p in recs:
        for k in ("shares", "shares_avg"):
            if not np.isnan(p.v.get(k, np.nan)):
                p.v[k] /= r
        if not np.isnan(p.v.get("eps", np.nan)):
            p.v["eps"] *= r
    kind = "a stock split after period end" if _split_like(r) else "a share-class or unit mismatch"
    return [f"share count restated by {1 / r:.3g}x to match Yahoo's market value ({kind})"]


def check_ttm(acc: "Accounts") -> list[str]:
    """Drop TTM records whose revenue is implausible against the annual report.

    Some Singapore companies report half-yearly; if Yahoo labels the periods
    inconsistently, a sum of 'quarters' can double-count. A TTM revenue outside
    0.6-1.8x the latest annual revenue is treated as a data error.
    """
    if not acc.ttm or not acc.annual:
        return []
    bad = []
    for t in acc.ttm:
        prior = [p for p in acc.annual if p.end <= t.end]
        if not prior:
            continue
        ra, rt = prior[-1].v.get("revenue", np.nan), t.v.get("revenue", np.nan)
        if np.isnan(ra) or np.isnan(rt) or ra <= 0:
            continue
        if not 0.6 <= rt / ra <= 1.8:
            bad.append(t)
    if bad:
        acc.ttm = [t for t in acc.ttm if t not in bad]
        return [f"{len(bad)} trailing-twelve-month record(s) dropped: revenue inconsistent with the annual report"]
    return []


def asof(periods: list[Period], date) -> tuple:
    """Latest fiscal year public by `date`, with the one or two years before it."""
    date = pd.Timestamp(date)
    known = [p for p in periods if p.avail <= date]
    if not known:
        return None, None, None
    cur = known[-1]
    prev = known[-2] if len(known) >= 2 and (cur.end - known[-2].end).days < 550 else None
    prev2 = known[-3] if prev is not None and len(known) >= 3 and (prev.end - known[-3].end).days < 550 else None
    return cur, prev, prev2


def _ratio(a, b):
    if a is None or b is None or np.isnan(a) or np.isnan(b) or b == 0:
        return np.nan
    return a / b


def _avg(a, b):
    xs = [x for x in (a, b) if x is not None and not np.isnan(x)]
    return float(np.mean(xs)) if xs else np.nan


def _split_like(r: float) -> bool:
    """True if a share-count ratio looks like an unadjusted stock split."""
    if np.isnan(r):
        return False
    for f in (2, 3, 4, 5, 10, 20, 0.5, 1 / 3, 0.25, 0.2, 0.1, 0.05):
        if abs(r / f - 1) < 0.06:
            return True
    return False


def fscore(cur: dict, prev: dict | None, prev2: dict | None, bank: bool) -> tuple[float, int, dict]:
    """Piotroski (2000) F-score, scaled to 9 when some signals cannot be computed.

    Returns (score out of 9, number of signals available, per-signal results).
    Banks and insurers skip the OCF, accrual, current-ratio and gross-margin
    signals, which do not describe a lending business.
    """
    s = {}
    a_beg = prev["assets"] if prev else np.nan
    roa = _ratio(cur["net_income"], a_beg if not np.isnan(a_beg) else cur["assets"])
    s["roa_pos"] = None if np.isnan(roa) else roa > 0
    if not bank:
        s["ocf_pos"] = None if np.isnan(cur["ocf"]) else cur["ocf"] > 0
        cfoa = _ratio(cur["ocf"], a_beg if not np.isnan(a_beg) else cur["assets"])
        s["accrual"] = None if (np.isnan(cfoa) or np.isnan(roa)) else cfoa > roa
    if prev is not None:
        a_beg_prev = prev2["assets"] if prev2 else np.nan
        roa_prev = _ratio(prev["net_income"], a_beg_prev if not np.isnan(a_beg_prev) else prev["assets"])
        if np.isnan(a_beg_prev):  # compare like with like when the older year is missing
            roa = _ratio(cur["net_income"], cur["assets"])
        s["droa"] = None if (np.isnan(roa) or np.isnan(roa_prev)) else roa > roa_prev
        lev, lev_p = _ratio(cur["debt"], cur["assets"]), _ratio(prev["debt"], prev["assets"])
        s["dlever"] = None if (np.isnan(lev) or np.isnan(lev_p)) else lev <= lev_p
        if not bank:
            cr, cr_p = (_ratio(cur["current_assets"], cur["current_liabilities"]),
                        _ratio(prev["current_assets"], prev["current_liabilities"]))
            s["dliquid"] = None if (np.isnan(cr) or np.isnan(cr_p)) else cr > cr_p
            gm, gm_p = _ratio(cur["gross_profit"], cur["revenue"]), _ratio(prev["gross_profit"], prev["revenue"])
            s["dmargin"] = None if (np.isnan(gm) or np.isnan(gm_p)) else gm > gm_p
        r = _ratio(cur["shares"], prev["shares"])
        s["no_dilution"] = None if (np.isnan(r) or _split_like(r)) else r <= 1.005
        at, at_p = _ratio(cur["revenue"], cur["assets"]), _ratio(prev["revenue"], prev["assets"])
        s["dturnover"] = None if (np.isnan(at) or np.isnan(at_p)) else at > at_p
    avail = [v for v in s.values() if v is not None]
    n = len(avail)
    score = 9.0 * sum(avail) / n if n else np.nan
    return score, n, s


@dataclass
class Accounts:
    """Everything known about one company's accounts, each record with its availability date."""
    annual: list                                   # fiscal years (Period), oldest first
    quarters: list = field(default_factory=list)   # quarters or half-years
    ttm: list = field(default_factory=list)        # trailing-twelve-month records
    source: str = "Yahoo"


def _growth(a, b):
    if a is None or b is None or np.isnan(a) or np.isnan(b):
        return np.nan
    if b > 0:
        return a / b - 1
    return 1.0 if a > 0 else -1.0   # from a loss: back to profit counts +100%, deeper loss -100%


def metrics(tk: str, acc: "Accounts", date, price_local: float, fx_quote: float,
            fx_fin: float, divs_12m_local: float = np.nan) -> dict:
    """All fundamental measures for `tk` as they were knowable on `date`.

    Profit, cash flow and valuation use the latest trailing twelve months when a
    quarter or half-year newer than the last annual report has been published;
    the Piotroski F-score always uses annual reports, as Piotroski (2000) did.

    price_local: split-adjusted (not dividend-adjusted) price in the quote currency.
    fx_quote / fx_fin: USD per unit of quote currency / of reporting currency on `date`.
    """
    date = pd.Timestamp(date)
    cur, prev, prev2 = asof(acc.annual, date)
    m = {"fy_end": None, "fy_avail": None, "has_prev": False, "basis": None}
    if cur is None:
        return m
    bank = tk in C.BANK_LIKE
    ttm_known = [t for t in acc.ttm if t.avail <= date]
    ttm = ttm_known[-1] if ttm_known else None
    use_ttm = (ttm is not None and ttm.end > cur.end + pd.Timedelta(days=20)
               and not np.isnan(ttm.v.get("net_income", np.nan)))
    if use_ttm:
        L = dict(ttm.v)
        for k in STOCK_KEYS:             # a balance-sheet item missing from the quarter: use the year's
            if np.isnan(L.get(k, np.nan)):
                L[k] = cur.v.get(k, np.nan)
        target = ttm.end - pd.Timedelta(days=365)
        py = [t for t in ttm_known if abs((t.end - target).days) <= 20]
        if not py:
            py = [p for p in acc.annual if p.avail <= date and abs((p.end - target).days) <= 20]
        L_py = py[-1].v if py else None
        m["basis"] = f"TTM to {ttm.end.date()}"
    else:
        L, L_py = cur.v, (prev.v if prev else None)
        m["basis"] = f"FY to {cur.end.date()}"
    c, p = cur.v, (prev.v if prev else None)
    p2 = prev2.v if prev2 else None
    m.update(fy_end=cur.end.date(), fy_avail=cur.avail.date(), has_prev=p is not None)
    ni, eq = L["net_income"], L["equity"]
    m["net_income"], m["revenue"], m["ocf"], m["fcf"] = ni, L["revenue"], L["ocf"], L["fcf"]
    eq_py = L_py["equity"] if L_py else np.nan
    eq_avg = _avg(eq, eq_py) if (eq_py is not None and eq_py > 0 and eq > 0) else eq
    m["roe"] = _ratio(ni, eq_avg) if (eq_avg is not None and eq_avg > 0) else np.nan
    m["roa"] = _ratio(ni, L["assets"])
    m["gpa"] = np.nan if bank else _ratio(L["gross_profit"], L["assets"])
    m["gross_margin"] = np.nan if bank else _ratio(L["gross_profit"], L["revenue"])
    m["net_margin"] = _ratio(ni, L["revenue"])
    a_avg = _avg(L["assets"], L_py["assets"] if L_py else np.nan)
    m["accruals"] = np.nan if bank else _ratio(ni - L["ocf"], a_avg)
    m["de"] = _ratio(L["debt"], eq) if (eq is not None and eq > 0) else np.nan
    m["gearing"] = _ratio(L["debt"], L["assets"])
    m["eq_assets"] = _ratio(eq, L["assets"])
    m["payout"] = _ratio(-L["dividends_paid"], ni) if (ni and ni > 0 and not np.isnan(L["dividends_paid"])) else np.nan

    # growth: the most recent like-for-like comparison available
    if L_py is not None:
        m["ni_growth"] = _growth(ni, L_py["net_income"])
        m["rev_growth"] = _growth(L["revenue"], L_py["revenue"])
        m["ocf_growth"] = _growth(L["ocf"], L_py["ocf"]) if (L_py["ocf"] or 0) > 0 else np.nan
        m["asset_growth"] = _growth(L["assets"], L_py["assets"])
        m["growth_basis"] = "TTM vs a year earlier" if use_ttm else "fiscal year vs prior year"
    else:
        qk = [q for q in acc.quarters if q.avail <= date]
        qpy = [q for q in qk if abs((q.end - (qk[-1].end - pd.Timedelta(days=365))).days) <= 20] if qk else []
        if use_ttm and qpy and qk[-1].end > cur.end:
            ql, qp = qk[-1].v, qpy[-1].v
            m["ni_growth"] = _growth(ql["net_income"], qp["net_income"])
            m["rev_growth"] = _growth(ql["revenue"], qp["revenue"])
            m["ocf_growth"] = np.nan
            m["asset_growth"] = _growth(ql["assets"], qp["assets"])
            m["growth_basis"] = "latest quarter vs same quarter last year"
        elif p is not None:
            m["ni_growth"] = _growth(c["net_income"], p["net_income"])
            m["rev_growth"] = _growth(c["revenue"], p["revenue"])
            m["ocf_growth"] = _growth(c["ocf"], p["ocf"]) if (p["ocf"] or 0) > 0 else np.nan
            m["asset_growth"] = _growth(c["assets"], p["assets"])
            m["growth_basis"] = "fiscal year vs prior year"
        else:
            for k in ("ni_growth", "rev_growth", "ocf_growth", "asset_growth"):
                m[k] = np.nan
            m["growth_basis"] = None
    if p is not None and not bank:
        gm, gm_p = _ratio(c["gross_profit"], c["revenue"]), _ratio(p["gross_profit"], p["revenue"])
        m["d_gross_margin"] = gm - gm_p if not (np.isnan(gm) or np.isnan(gm_p)) else np.nan
    else:
        m["d_gross_margin"] = np.nan
    fs, fn, sig = fscore(c, p, p2, bank)
    m["fscore"], m["fscore_n"], m["fscore_signals"] = fs, fn, sig

    # valuation at `date`, in USD on both sides
    shares = L["shares"] if not np.isnan(L["shares"]) else (c["shares"] if not np.isnan(c["shares"]) else c["shares_avg"])
    mcap_usd = shares * price_local * fx_quote if (shares and shares > 0 and price_local > 0) else np.nan
    m["mcap_usd"] = mcap_usd
    ni_usd, eq_usd, fcf_usd = ni * fx_fin, eq * fx_fin, L["fcf"] * fx_fin
    ep = _ratio(ni_usd, mcap_usd)
    if np.isnan(ep) and not use_ttm and not np.isnan(c["eps"]) and price_local > 0:
        ep = c["eps"] * fx_fin / (price_local * fx_quote)
    m["ep"] = ep
    m["pe"] = 1.0 / ep if (ep is not None and not np.isnan(ep) and ep > 0) else np.nan
    m["bp"] = _ratio(eq_usd, mcap_usd)
    m["fcf_yield"] = np.nan if bank else _ratio(fcf_usd, mcap_usd)
    m["div_yield"] = divs_12m_local / price_local if (price_local > 0 and not np.isnan(divs_12m_local)) else np.nan
    return m


def gate(tk: str, m: dict, k_e: float) -> tuple[bool, list, list]:
    """The fundamental gate. Returns (passed, reasons for failing, soft flags).

    A stock with missing data fails: we do not buy what we cannot justify.
    """
    fails, flags = [], []
    bank, reit = tk in C.BANK_LIKE, tk in C.REIT_LIKE
    if m.get("fy_end") is None:
        return False, ["no published annual accounts"], flags
    ni, ocf, roe = m.get("net_income", np.nan), m.get("ocf", np.nan), m.get("roe", np.nan)
    # G1 profitable, and profits backed by cash
    if np.isnan(ni) or ni <= 0:
        fails.append("G1 not profitable (net income <= 0)")
    elif not bank and (np.isnan(ocf) or ocf <= 0):
        fails.append("G1 operating cash flow not positive")
    # G2 value creation: return on equity at least the SML cost of equity
    if np.isnan(roe):
        fails.append("G2 ROE unavailable (negative or missing equity)")
    elif np.isnan(k_e):
        fails.append("G2 cost of equity unavailable (beta needs 78 weeks of prices)")
    elif roe < k_e:
        fails.append(f"G2 ROE {roe:.1%} below SML cost of equity {k_e:.1%}")
    # G3 accounting quality: Piotroski F-score
    fs, fn = m.get("fscore", np.nan), m.get("fscore_n", 0)
    if not m.get("has_prev"):
        fails.append("G3 F-score needs two years of accounts")
    elif fn < C.FSCORE_MIN_SIGNALS:
        fails.append(f"G3 only {fn} of 9 F-score signals computable")
    elif fs < C.FSCORE_MIN:
        fails.append(f"G3 F-score {fs:.1f}/9 below {C.FSCORE_MIN}")
    # G4 earnings trend
    g = m.get("ni_growth", np.nan)
    if np.isnan(g):
        fails.append("G4 profit growth unavailable")
    elif g < C.NI_GROWTH_MIN:
        fails.append(f"G4 profit fell {g:.1%} (limit {C.NI_GROWTH_MIN:.0%})")
    # G5 balance sheet
    if bank:
        ea = m.get("eq_assets", np.nan)
        if np.isnan(ea) or ea < C.BANK_EQ_ASSETS_MIN:
            fails.append(f"G5 equity/assets {ea:.1%} below {C.BANK_EQ_ASSETS_MIN:.0%}")
    elif reit:
        gr = m.get("gearing", np.nan)
        if np.isnan(gr) or gr > C.REIT_GEARING_MAX:
            fails.append(f"G5 gearing {gr:.1%} above {C.REIT_GEARING_MAX:.0%}")
    else:
        de = m.get("de", np.nan)
        if np.isnan(de):
            fails.append("G5 debt/equity unavailable (negative equity?)")
        elif de > C.DE_MAX:
            fails.append(f"G5 debt/equity {de:.2f} above {C.DE_MAX:.1f}")
    # G6 valuation sanity
    pe = m.get("pe", np.nan)
    if np.isnan(pe):
        fails.append("G6 P/E unavailable or negative")
    elif pe > C.PE_MAX:
        fails.append(f"G6 P/E {pe:.0f} above {C.PE_MAX:.0f}")
    # soft flags for the human red-flag check (not rules)
    if not np.isnan(m.get("accruals", np.nan)) and m["accruals"] > 0.10:
        flags.append(f"high accruals {m['accruals']:.1%} of assets (profit well above cash flow)")
    og = m.get("ocf_growth", np.nan)
    if not np.isnan(og) and og < -0.40 and not bank:
        flags.append(f"operating cash flow fell {og:.0%}")
    rg = m.get("rev_growth", np.nan)
    if not np.isnan(rg) and rg < -0.15:
        flags.append(f"revenue fell {rg:.0%}")
    return len(fails) == 0, fails, flags


def quality_value_parts(m: dict, k_e: float) -> dict:
    """Raw inputs for the quality and value scores (z-scored later, cross-sectionally)."""
    roe = m.get("roe", np.nan)
    return {
        "q_spread": roe - k_e if not (np.isnan(roe) or np.isnan(k_e)) else np.nan,  # economic profit
        "q_fscore": m.get("fscore", np.nan),
        "q_growth": np.clip(m.get("ni_growth", np.nan), -1.0, 1.0) if not np.isnan(m.get("ni_growth", np.nan)) else np.nan,
        "q_accruals": -m.get("accruals", np.nan) if not np.isnan(m.get("accruals", np.nan)) else np.nan,
        "v_ep": m.get("ep", np.nan),
        "v_bp": m.get("bp", np.nan),
        "v_fcf": m.get("fcf_yield", np.nan),
    }
