"""
Optional: point-in-time US fundamentals from SEC EDGAR (XBRL "companyfacts").

Yahoo serves only four years of accounts, so a Yahoo-only backtest of the
fundamental gate starts in 2024. EDGAR has every 10-K since 2009 WITH its filing
date, so for US stocks the gate can be tested over 2015-2026 without look-ahead:
each fiscal year is used from the day its original 10-K was filed, with the
figures as first reported (later restatements are ignored).

The SEC asks every client to identify itself. Set SEC_USER_AGENT, e.g.
    set SEC_USER_AGENT=YourName your.email@school.edu        (Windows)
    export SEC_USER_AGENT="YourName your.email@school.edu"   (Mac/Linux)
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from .data import _cache_path, _fresh
from .fundamentals import FIELDS, FLOW_KEYS, STOCK_KEYS, Accounts, Period

UA = os.environ.get("SEC_USER_AGENT", "ACC2026 backtest research@example.com")
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
ANNUAL_FORMS = {"10-K", "10-K/A", "10-KT", "20-F", "20-F/A", "40-F", "40-F/A"}
QUARTER_FORMS = {"10-Q", "10-Q/A", "6-K"}

G, I, D = "us-gaap", "ifrs-full", "dei"
TAGS = {
    "revenue": [(G, "Revenues"), (G, "RevenueFromContractWithCustomerExcludingAssessedTax"),
                (G, "SalesRevenueNet"), (G, "RevenueFromContractWithCustomerIncludingAssessedTax"),
                (G, "RevenuesNetOfInterestExpense"), (I, "Revenue")],
    "cogs": [(G, "CostOfRevenue"), (G, "CostOfGoodsAndServicesSold"), (G, "CostOfGoodsSold"),
             (I, "CostOfSales")],
    "gross_profit": [(G, "GrossProfit"), (I, "GrossProfit")],
    "net_income": [(G, "NetIncomeLossAvailableToCommonStockholdersBasic"), (G, "NetIncomeLoss"),
                   (G, "ProfitLoss"), (I, "ProfitLossAttributableToOwnersOfParent"), (I, "ProfitLoss")],
    "eps": [(G, "EarningsPerShareDiluted"), (G, "EarningsPerShareBasic"),
            (I, "DilutedEarningsLossPerShare")],
    "shares_avg": [(G, "WeightedAverageNumberOfDilutedSharesOutstanding"),
                   (G, "WeightedAverageNumberOfSharesOutstandingBasic")],
    "ocf": [(G, "NetCashProvidedByUsedInOperatingActivities"),
            (G, "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
            (I, "CashFlowsFromUsedInOperatingActivities")],
    "capex": [(G, "PaymentsToAcquirePropertyPlantAndEquipment"), (G, "PaymentsToAcquireProductiveAssets"),
              (I, "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities")],
    "dividends_paid": [(G, "PaymentsOfDividendsCommonStock"), (G, "PaymentsOfDividends"),
                       (I, "DividendsPaidClassifiedAsFinancingActivities")],
    "equity": [(G, "StockholdersEquity"),
               (G, "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
               (I, "EquityAttributableToOwnersOfParent"), (I, "Equity")],
    "assets": [(G, "Assets"), (I, "Assets")],
    "lt_debt": [(G, "LongTermDebt"), (G, "LongTermDebtNoncurrent"), (I, "NoncurrentPortionOfNoncurrentBorrowings"),
                (I, "Borrowings")],
    "current_debt": [(G, "LongTermDebtCurrent"), (G, "DebtCurrent"), (G, "ShortTermBorrowings"),
                     (I, "CurrentPortionOfNoncurrentBorrowings")],
    "current_assets": [(G, "AssetsCurrent"), (I, "CurrentAssets")],
    "current_liabilities": [(G, "LiabilitiesCurrent"), (I, "CurrentLiabilities")],
    "shares": [(D, "EntityCommonStockSharesOutstanding"), (G, "CommonStockSharesOutstanding")],
    "cash": [(G, "CashAndCashEquivalentsAtCarryingValue"), (I, "CashAndCashEquivalents")],
}


def _get(url: str) -> bytes:
    import requests
    for attempt in range(4):
        try:
            r = requests.get(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}, timeout=60)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return b""
        except Exception:
            pass
        time.sleep(2 ** (attempt + 1))
    return b""


def cik_map() -> dict:
    path = _cache_path("sec", "company_tickers.json")
    if not _fresh(path, 24 * 30):
        raw = _get(TICKERS_URL)
        if not raw:
            return {}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(raw)
    with open(path) as fh:
        js = json.load(fh)
    return {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in js.values()}


def companyfacts(cik: int) -> dict:
    path = _cache_path("sec", f"CIK{cik:010d}.json")
    if not _fresh(path, 24 * 7):
        raw = _get(FACTS_URL.format(cik=cik))
        time.sleep(0.15)  # SEC fair-access limit is 10 requests a second
        if not raw:
            return {}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(raw)
    with open(path) as fh:
        return json.load(fh)


def _facts(js: dict, tax: str, tag: str, forms=None) -> list:
    node = js.get("facts", {}).get(tax, {}).get(tag)
    if not node:
        return []
    forms = forms or ANNUAL_FORMS
    out = []
    for unit, arr in node.get("units", {}).items():
        for f in arr:
            if f.get("form") in forms:
                out.append({**f, "unit": unit})
    return out


def _earliest(facts, pred):
    best = None
    for f in facts:
        if pred(f) and (best is None or f["filed"] < best["filed"]):
            best = f
    return best


def _days(a, b) -> int:
    return abs((pd.Timestamp(a) - pd.Timestamp(b)).days)


NEG = {"capex", "dividends_paid"}   # SEC reports these payments as positive numbers


def ttm_from_companyfacts(js: dict, annual: list, splits: pd.Series | None = None) -> list[Period]:
    """Trailing-twelve-month records at every 10-Q, dated by the 10-Q filing date.

    TTM = last fiscal year + this year's year-to-date - last year's year-to-date,
    using only figures filed by the 10-Q date. Balance-sheet items are the 10-Q's.
    """
    if not js or not annual:
        return []
    both = ANNUAL_FORMS | QUARTER_FORMS
    cache = {}

    def facts(tax, tag):
        if (tax, tag) not in cache:
            cache[(tax, tag)] = _facts(js, tax, tag, both)
        return cache[(tax, tag)]

    filings = {}
    for tax, tag in TAGS["net_income"]:
        for f in facts(tax, tag):
            if f.get("form") not in QUARTER_FORMS or "start" not in f:
                continue
            e, fd = pd.Timestamp(f["end"]), pd.Timestamp(f["filed"])
            a = f.get("accn")
            if a not in filings or e > filings[a][0]:
                filings[a] = (e, min(fd, filings.get(a, (e, fd))[1]))
        if filings:
            break
    out = {}
    for accn, (E, F) in filings.items():
        fys = [p for p in annual if p.end < E - pd.Timedelta(days=20) and p.avail <= F + pd.Timedelta(days=5)]
        if not fys or (E - fys[-1].end).days > 300:
            continue
        fy = fys[-1]
        fy_start = fy.end + pd.Timedelta(days=1)
        lim = F + pd.Timedelta(days=5)
        v = {k: np.nan for k in FIELDS}
        for key in [k for k in FLOW_KEYS if k != "fcf"]:
            fyv = fy.v.get(key, np.nan)
            if np.isnan(fyv):
                continue
            for tax, tag in TAGS.get(key, []):
                fl = facts(tax, tag)
                ytd = _earliest(fl, lambda f: "start" in f and pd.Timestamp(f["filed"]) <= lim
                                and _days(f["end"], E) <= 7 and _days(f["start"], fy_start) <= 10)
                if ytd is None:
                    continue
                ytd_py = _earliest(fl, lambda f: "start" in f and pd.Timestamp(f["filed"]) <= lim
                                   and _days(f["end"], E - pd.DateOffset(years=1)) <= 10
                                   and _days(f["start"], fy_start - pd.DateOffset(years=1)) <= 10)
                if ytd_py is None:
                    continue
                sgn = -1.0 if key in NEG else 1.0
                v[key] = fyv + sgn * (float(ytd["val"]) - float(ytd_py["val"]))
                break
        for key in STOCK_KEYS:
            for tax, tag in TAGS.get(key, []):
                fl = facts(tax, tag)
                if key == "shares" and tax == D:
                    f = _earliest(fl, lambda f: f.get("accn") == accn)
                else:
                    f = _earliest(fl, lambda f: "start" not in f and pd.Timestamp(f["filed"]) <= lim
                                  and _days(f["end"], E) <= 7)
                if f is not None:
                    v[key] = float(f["val"])
                    break
        v["fcf"] = v["ocf"] + v["capex"] if not (np.isnan(v["ocf"]) or np.isnan(v["capex"])) else np.nan
        parts = [x for x in (v["lt_debt"], v["current_debt"]) if not np.isnan(x)]
        v["debt"] = sum(parts) if parts else np.nan
        if splits is not None and len(splits):
            later = splits[(splits.index > F) & (splits > 0)]
            factor = float(np.prod(later.values)) if len(later) else 1.0
            if not np.isnan(v["shares"]):
                v["shares"] *= factor
        if np.isnan(v["net_income"]):
            continue
        if E not in out or F < out[E].avail:
            out[E] = Period(end=E, avail=F, v=v)
    # the annual reports are TTM records too (at each fiscal year end)
    for p in annual:
        if p.end not in out:
            out[p.end] = Period(end=p.end, avail=p.avail, v=dict(p.v))
    return [out[k] for k in sorted(out)]


def periods_from_companyfacts(js: dict, splits: pd.Series | None = None) -> list[Period]:
    """One Period per fiscal year, available from the ORIGINAL 10-K filing date.

    Share counts and EPS are restated to today's share basis with Yahoo's split
    history, so that shares x today's split-adjusted price is a true market value.
    """
    if not js:
        return []
    # fiscal years = one-year net-income facts; first filing of each is the availability date
    fy = {}
    for tax, tag in TAGS["net_income"]:
        for f in _facts(js, tax, tag):
            if "start" not in f:
                continue
            s, e = pd.Timestamp(f["start"]), pd.Timestamp(f["end"])
            if not 330 <= (e - s).days <= 380:
                continue
            filed = pd.Timestamp(f["filed"])
            if e not in fy or filed < fy[e]["filed"]:
                fy[e] = {"filed": filed, "accn": f.get("accn")}
        if fy:
            break
    if not fy:
        return []
    out = []
    for end in sorted(fy):
        filed, accn = fy[end]["filed"], fy[end]["accn"]
        v = {}
        for key in FIELDS:
            v[key] = np.nan
            for tax, tag in TAGS.get(key, []):
                best = None
                for f in _facts(js, tax, tag):
                    fd = pd.Timestamp(f["filed"])
                    if fd > filed + pd.Timedelta(days=5):
                        continue   # figure not public when this year's 10-K came out
                    if key == "shares" and tax == D:
                        if f.get("accn") != accn:
                            continue
                    else:
                        if abs((pd.Timestamp(f["end"]) - end).days) > 7:
                            continue
                        if "start" in f and not 330 <= (pd.Timestamp(f["end"]) - pd.Timestamp(f["start"])).days <= 380:
                            continue
                    if best is None or fd < pd.Timestamp(best["filed"]):
                        best = f
                if best is not None:
                    v[key] = float(best["val"])
                    break
        if not np.isnan(v["capex"]) and v["capex"] > 0:
            v["capex"] = -v["capex"]            # SEC reports payments as positive numbers
        if not np.isnan(v["dividends_paid"]) and v["dividends_paid"] > 0:
            v["dividends_paid"] = -v["dividends_paid"]
        if np.isnan(v["gross_profit"]) and not np.isnan(v["revenue"]) and not np.isnan(v["cogs"]):
            v["gross_profit"] = v["revenue"] - v["cogs"]
        v["fcf"] = v["ocf"] + v["capex"] if not (np.isnan(v["ocf"]) or np.isnan(v["capex"])) else np.nan
        parts = [x for x in (v["lt_debt"], v["current_debt"]) if not np.isnan(x)]
        v["debt"] = sum(parts) if parts else np.nan
        # restate per-share figures to today's basis
        if splits is not None and len(splits):
            later = splits[(splits.index > filed) & (splits > 0)]
            factor = float(np.prod(later.values)) if len(later) else 1.0
            for k in ("shares", "shares_avg"):
                if not np.isnan(v[k]):
                    v[k] *= factor
            if not np.isnan(v["eps"]):
                v["eps"] /= factor
        if np.isnan(v["net_income"]) and np.isnan(v["equity"]):
            continue
        out.append(Period(end=end, avail=filed, v=v))
    return out


def load_sec_periods(tickers: list[str], splits: dict, verbose=True) -> dict:
    """{ticker: Accounts} for US tickers, empty if EDGAR is unreachable."""
    cmap = cik_map()
    if not cmap:
        if verbose:
            print("  SEC EDGAR unreachable: using Yahoo fundamentals only", flush=True)
        return {}
    out = {}
    for n, tk in enumerate(tickers, 1):
        if tk.endswith(".SI"):
            continue
        cik = cmap.get(tk.upper())
        if cik is None:
            continue
        try:
            js = companyfacts(cik)
            per = periods_from_companyfacts(js, splits.get(tk))
            ttm = ttm_from_companyfacts(js, per, splits.get(tk)) if per else []
        except Exception as exc:
            if verbose:
                print(f"  SEC {tk}: {type(exc).__name__}: {exc}", flush=True)
            continue
        if per:
            out[tk] = Accounts(annual=per, ttm=ttm, source="SEC EDGAR")
        if verbose and n % 20 == 0:
            print(f"  SEC: {n}/{len(tickers)}", flush=True)
    return out
