"""
A synthetic market with the same data shapes as yfinance, for offline testing.

It plants known structure so the tests can check the machinery:
  * every stock has known FF3 and macro betas;
  * a persistent latent "quality" q drives both the accounts (ROE, growth, margins)
    and a small extra return, so quality and residual momentum carry information;
  * a few stocks have strong price trends but deteriorating accounts ("re-rating bets").
None of this says anything about real markets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ffq import config as C

NOW = pd.Timestamp("2026-09-23")
START = pd.Timestamp("2012-01-02")

SG_NAMES = ["D05.SI", "O39.SI", "U11.SI", "Z74.SI", "S68.SI", "C38U.SI", "A17U.SI", "BN4.SI",
            "S63.SI", "F34.SI", "OV8.SI", "C07.SI", "H78.SI", "BS6.SI", "V03.SI", "S58.SI",
            "M44U.SI", "558.SI", "F99.SI", "E5H.SI", "C6L.SI", "U96.SI"]
US_NAMES = ["AAPL", "MSFT", "NVDA", "AMD", "MU", "AVGO", "JPM", "V", "WMT", "COST", "UNH",
            "HD", "MCD", "CAT", "GE", "LLY", "XOM", "KO", "PG", "NFLX", "AMZN", "GOOGL",
            "META", "ORCL", "BA", "QCOM", "BAC", "GS", "INTC", "AMAT", "ABBV", "CVX", "NEE",
            "UNP", "TMO", "PEP", "NEM", "B", "AMT", "PLTR"]
BAD_TREND = {"CAT", "BN4.SI", "AMD"}     # rising prices, falling profits


class SyntheticProvider:
    def __init__(self, seed: int = 11, now: pd.Timestamp = NOW):
        self.rng = np.random.default_rng(seed)
        self.now = now
        self._acc_cache = {}
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        rng = self.rng
        days = pd.bdate_range(START, self.now)
        n = len(days)
        self.days = days
        yrs = (days - days[0]).days / 365.25
        # T-bill yield path (percent)
        irx = np.interp(yrs, [0, 4, 7, 8, 9.5, 11, 12.5, 14.8], [0.05, 0.3, 2.4, 0.1, 0.05, 4.5, 5.2, 4.0])
        rf_d = irx / 100 / 252
        f = {}
        f["MKT_US"] = rng.normal(0.00035, 0.010, n)
        f["MKT_SG"] = 0.45 * f["MKT_US"] + rng.normal(0.0001, 0.007, n)
        f["SMB_US"] = rng.normal(0.0, 0.005, n)
        f["HML_US"] = rng.normal(0.0, 0.005, n)
        f["SMB_INT"] = rng.normal(0.0, 0.004, n)
        f["HML_INT"] = rng.normal(0.0, 0.004, n)
        f["RATES"] = -0.15 * f["MKT_US"] + rng.normal(0.00005, 0.0035, n)
        f["OIL"] = 0.4 * f["MKT_US"] + rng.normal(0.0, 0.02, n)
        f["GOLD_F"] = rng.normal(0.0002, 0.009, n)
        f["USD"] = -0.1 * f["MKT_US"] + rng.normal(0.0, 0.004, n)
        # a crash to exercise risk controls (a 2020-style month)
        crash = (days >= "2020-02-24") & (days <= "2020-03-20")
        f["MKT_US"][crash] -= 0.012
        f["MKT_SG"][crash] -= 0.009
        self.f = f
        lvl = lambda r: 100 * np.exp(np.cumsum(r))
        etf = {
            "SPY": lvl(f["MKT_US"] + rf_d), "IWB": lvl(f["MKT_US"] + rf_d),
            "IWF": lvl(f["MKT_US"] + rf_d + 0.00002),
            "EWS": lvl(f["MKT_SG"] + rf_d), "EFA": lvl(0.8 * f["MKT_US"] + rf_d),
            "QUAL": lvl(f["MKT_US"] + rf_d + rng.normal(0.00003, 0.002, n)),
            "MTUM": lvl(f["MKT_US"] + rf_d + rng.normal(0.00003, 0.004, n)),
            "IEF": lvl(f["RATES"] + rf_d), "GLD": lvl(f["GOLD_F"]), "USO": lvl(f["OIL"]),
            "UUP": lvl(f["USD"]),
        }
        etf["IWM"] = etf["IWB"] * np.exp(np.cumsum(f["SMB_US"]))
        etf["IWD"] = etf["IWF"] * np.exp(np.cumsum(f["HML_US"]))
        etf["SCZ"] = etf["EFA"] * np.exp(np.cumsum(f["SMB_INT"]))
        etf["EFG"] = etf["EFA"].copy()
        etf["EFV"] = etf["EFG"] * np.exp(np.cumsum(f["HML_INT"]))
        self.series = {k: pd.Series(v, index=days) for k, v in etf.items()}
        self.series["^GSPC"] = pd.Series(etf["SPY"] * 30, index=days)
        self.series["^STI"] = pd.Series(etf["EWS"] * 40, index=days)
        self.series["^IRX"] = pd.Series(irx, index=days)
        fx = {"SGDUSD=X": 0.74, "CNYUSD=X": 0.145, "THBUSD=X": 0.029, "HKDUSD=X": 0.128,
              "EURUSD=X": 1.1, "GBPUSD=X": 1.3, "JPYUSD=X": 0.0075, "MYRUSD=X": 0.22,
              "IDRUSD=X": 0.000065, "AUDUSD=X": 0.7}
        for k, v in fx.items():
            self.series[k] = pd.Series(v * np.exp(np.cumsum(rng.normal(0, 0.003, n))), index=days)

        # calendars
        us_hol = set(rng.choice(n, size=int(n * 0.035), replace=False))
        sg_hol = set(rng.choice(n, size=int(n * 0.04), replace=False))
        self.us_open = np.array([k not in us_hol for k in range(n)])
        self.sg_open = np.array([k not in sg_hol for k in range(n)])

        # stocks
        self.meta, self.px = {}, {}
        n_years = int(np.ceil(yrs[-1])) + 2
        for tk in SG_NAMES + US_NAMES:
            sg = tk.endswith(".SI")
            bank = tk in C.BANK_LIKE
            reit = tk in C.REIT_LIKE
            q = np.zeros(n_years)
            q[0] = rng.normal(0, 1)
            for y in range(1, n_years):
                q[y] = 0.8 * q[y - 1] + rng.normal(0, 0.6)
            if tk in BAD_TREND:
                q[-4:] = [-0.5, -1.0, -1.5, -1.8]
            beta = {"MKT": rng.uniform(0.6, 1.5), "SMB": rng.normal(0, 0.4), "HML": rng.normal(0, 0.4),
                    "OIL": rng.normal(0, 0.1), "RATES": rng.normal(0, 0.3), "GOLD_F": 0.0}
            if tk in C.GOLD:
                beta["GOLD_F"], beta["MKT"] = 1.6, 0.4
            if reit:
                beta["RATES"] = 1.2
            idio = rng.uniform(0.011, 0.022)
            yi = np.minimum((yrs).astype(int), n_years - 1)
            alpha = 0.00022 * q[yi]
            if tk in BAD_TREND:  # price keeps trending up while accounts deteriorate, then cracks
                alpha = alpha + np.where(yrs > yrs[-1] - 2.0, 0.0012, 0.0) - np.where(yrs > yrs[-1] - 0.4, 0.004, 0.0)
            mk = f["MKT_SG"] if sg else f["MKT_US"]
            smb, hml = (f["SMB_INT"], f["HML_INT"]) if sg else (f["SMB_US"], f["HML_US"])
            r = (rf_d + alpha + beta["MKT"] * mk + beta["SMB"] * smb + beta["HML"] * hml
                 + beta["OIL"] * f["OIL"] + beta["RATES"] * f["RATES"] + beta["GOLD_F"] * f["GOLD_F"]
                 + rng.normal(0, idio, n))
            # industry co-movement the factor model cannot see: SG banks share a shock
            if tk in ("D05.SI", "O39.SI", "U11.SI"):
                r = r + self._bank_shock()
            if tk in ("MU", "AMAT", "INTC", "NVDA", "AMD", "AVGO", "QCOM"):
                r = r + self._semi_shock()
            ccy_q = "USD" if (not sg or tk in C.SG_USD_QUOTED) else "SGD"
            fin = C.FIN_CCY_FALLBACK.get(tk, ccy_q) if sg else "USD"
            listed = START if tk != "PLTR" else pd.Timestamp("2020-09-30")
            tr = 50 * np.exp(np.cumsum(r))
            if ccy_q == "SGD":
                tr = tr / self.series["SGDUSD=X"].values   # local-currency total return index
            dy = 0.0 if tk in ("AMD", "NFLX", "AMZN", "META", "PLTR") else rng.uniform(0.005, 0.05)
            if reit:
                dy = rng.uniform(0.045, 0.065)
            self.meta[tk] = dict(q=q, beta=beta, sg=sg, bank=bank, reit=reit, ccy=ccy_q, fin=fin,
                                 listed=listed, dy=dy, fye_month=6 if tk in ("S68.SI", "MU") else 12)
            self.px[tk] = tr

    def _bank_shock(self):
        if not hasattr(self, "_bs"):
            self._bs = self.rng.normal(0, 0.008, len(self.days))
        return self._bs

    def _semi_shock(self):
        if not hasattr(self, "_ss"):
            self._ss = self.rng.normal(0, 0.012, len(self.days))
        return self._ss

    # ------------------------------------------------------------------
    def _stock_frame(self, tk):
        mt = self.meta[tk]
        days = self.days
        tr = np.asarray(self.px[tk], dtype=float)
        open_mask = self.sg_open if mt["sg"] else self.us_open
        # quarterly ex-dividend days; the raw (unadjusted) price drops by the dividend
        q_idx = np.where((days.month % 3 == 0) & (days.day >= 15) & (days.day <= 21) & (days.dayofweek == 4))[0]
        drop = np.zeros(len(days))
        seen = set()
        for k in q_idx:
            key = (days[k].year, days[k].month)
            if key not in seen and k > 0:
                seen.add(key)
                drop[k] = mt["dy"] / 4
        cf = np.cumprod(1 - drop)              # raw = total-return index x cumulative factor
        raw = tr * cf
        divs = np.where(drop > 0, drop * np.r_[raw[0], raw[:-1]] / (1 - drop) * (1 - drop), 0.0)
        adj = tr * (raw[-1] / tr[-1])
        rng = np.random.default_rng(_seed(tk + "frame"))
        opn = np.r_[raw[0], raw[:-1]] * np.exp(rng.normal(0, 0.004, len(days)))
        vol = np.exp(rng.normal(np.log(2e7 / max(raw[-1], 1)), 0.5, len(days)))
        if tk in US_NAMES[:25]:
            vol = vol * 5
        df = pd.DataFrame({"Open": opn, "Close": raw, "Adj Close": adj, "Volume": vol,
                           "Dividends": divs, "Stock Splits": 0.0}, index=days)
        keep = open_mask & (days >= mt["listed"])
        df = df[keep]
        return df

    def history(self, tickers, start, end=None):
        out = {f: {} for f in ["Open", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]}
        for tk in tickers:
            if tk in self.px:
                df = self._stock_frame(tk)
            elif tk in self.series:
                s = self.series[tk]
                df = pd.DataFrame({"Open": s, "Close": s, "Adj Close": s, "Volume": 1e6,
                                   "Dividends": 0.0, "Stock Splits": 0.0})
                if tk.endswith("=X"):
                    pass
                else:
                    df = df[self.us_open]
            else:
                continue
            df = df[df.index >= pd.Timestamp(start)]
            for f_ in out:
                out[f_][tk] = df[f_]
        return {f_: pd.DataFrame(v).sort_index() for f_, v in out.items()}

    # ------------------------------------------------------------------
    QSHARE = (0.23, 0.24, 0.26, 0.27)    # share of each fiscal year's flows earned in Q1..Q4

    def _accounts_all(self, tk):
        """Every fiscal year, published or not: {fiscal year end: fields}."""
        if tk in self._acc_cache:
            return self._acc_cache[tk]
        mt = self.meta[tk]
        rng = np.random.default_rng(_seed(tk))
        out = {}
        rev, eq, sh = 1000.0, 800.0, 100.0
        lev = 9.0 if mt["bank"] else (0.9 if mt["reit"] else rng.uniform(0.2, 1.2))
        for y, qv in enumerate(mt["q"]):
            end = pd.Timestamp(year=2011 + y, month=mt["fye_month"], day=30 if mt["fye_month"] == 6 else 31)
            g = 0.05 + 0.08 * qv + rng.normal(0, 0.05)
            rev *= (1 + g)
            margin = 0.10 + 0.06 * qv + rng.normal(0, 0.02)
            ni = rev * margin
            eq = max(eq + ni * 0.6, 50.0)
            assets = eq * (1 + lev)
            debt = assets - eq - (0.1 * assets if not mt["bank"] else 0.0)
            debt = max(debt, 0.0) * (0.6 if not mt["bank"] else 0.9)
            ocf = ni * (1.1 + rng.normal(0, 0.15) - (0.5 if (tk in BAD_TREND and y >= len(mt["q"]) - 3) else 0))
            sh *= 1 + max(0, rng.normal(0.0, 0.01))
            out[end] = {
                "TotalRevenue": rev, "CostOfRevenue": rev * (0.6 - 0.05 * qv),
                "GrossProfit": None if mt["bank"] else rev * (0.4 + 0.05 * qv),
                "NetIncomeCommonStockholders": ni, "NetIncome": ni,
                "DilutedEPS": ni / sh, "DilutedAverageShares": sh,
                "OperatingCashFlow": ocf, "CapitalExpenditure": -0.3 * abs(ni),
                "FreeCashFlow": ocf - 0.3 * abs(ni), "CashDividendsPaid": -0.4 * max(ni, 0),
                "StockholdersEquity": eq, "TotalAssets": assets, "TotalDebt": debt,
                "CurrentAssets": None if mt["bank"] else assets * 0.3,
                "CurrentLiabilities": None if mt["bank"] else assets * 0.2 * (1 - 0.1 * qv),
                "OrdinarySharesNumber": sh,
                "CashAndCashEquivalents": assets * 0.05,
            }
        # scale share counts so the P/E at the latest PUBLISHED year is sensible
        pub = [e for e in sorted(out) if e <= self.now - pd.Timedelta(days=60)]
        if pub:
            last = pub[-1]
            p = float(self._stock_frame(tk)["Close"].asof(last))
            pe = 150 if tk == "AMD" else 12 + 10 * np.random.default_rng(_seed(tk + "pe")).random()
            ni = out[last]["NetIncomeCommonStockholders"]
            target = max(ni, 1.0) * pe / p if ni > 0 else 100.0
            k = target / out[last]["OrdinarySharesNumber"]
            for e in out:
                for key in ("OrdinarySharesNumber", "DilutedAverageShares"):
                    out[e][key] *= k
                out[e]["DilutedEPS"] = out[e]["NetIncomeCommonStockholders"] / out[e]["DilutedAverageShares"]
        self._acc_cache[tk] = out
        return out

    def annual_accounts(self, tk):
        """Published fiscal years only."""
        return {e: v for e, v in self._accounts_all(tk).items() if e <= self.now - pd.Timedelta(days=60)}

    def quarterly_accounts(self, tk):
        """{quarter end: fields}: flows for the quarter alone, balance sheet at quarter end."""
        acc = self._accounts_all(tk)
        ends = sorted(acc)
        out = {}
        for n, fe in enumerate(ends):
            prev = acc[ends[n - 1]] if n > 0 else acc[fe]
            for qi in range(4):
                qend = (fe - pd.DateOffset(months=3 * (3 - qi))) + pd.offsets.MonthEnd(0)
                v = {}
                for k, x in acc[fe].items():
                    if x is None:
                        v[k] = None
                    elif k in _FLOW_Y:
                        v[k] = x * self.QSHARE[qi]
                    elif k in ("DilutedEPS",):
                        v[k] = x * self.QSHARE[qi]
                    elif k in ("OrdinarySharesNumber", "DilutedAverageShares"):
                        v[k] = x
                    else:
                        v[k] = prev[k] + (x - prev[k]) * (qi + 1) / 4 if prev[k] is not None else x
                out[qend] = v
        return out

    def statements(self, tk, quarterly=False):
        if tk not in self.meta:
            return {"income": pd.DataFrame(), "balance": pd.DataFrame(), "cashflow": pd.DataFrame()}
        rows = {"income": ["TotalRevenue", "CostOfRevenue", "GrossProfit", "NetIncomeCommonStockholders",
                           "NetIncome", "DilutedEPS", "DilutedAverageShares"],
                "balance": ["StockholdersEquity", "TotalAssets", "TotalDebt", "CurrentAssets",
                            "CurrentLiabilities", "OrdinarySharesNumber", "CashAndCashEquivalents"],
                "cashflow": ["OperatingCashFlow", "CapitalExpenditure", "FreeCashFlow", "CashDividendsPaid"]}
        out = {}
        acc = self.annual_accounts(tk)
        ends = sorted(acc)[-4:]       # Yahoo serves only the last four fiscal years
        for name, keys in rows.items():
            data = {e: [acc[e][k] for k in keys] for e in reversed(ends)}
            out[name] = pd.DataFrame(data, index=keys).astype(float).dropna(how="all")
        if quarterly:
            qa = self.quarterly_accounts(tk)
            qends = [e for e in sorted(qa) if e + pd.Timedelta(days=40) <= self.now][-5:]   # five quarters
            for name, keys in rows.items():
                data = {e: [qa[e][k] for k in keys] for e in reversed(qends)}
                out["q_" + name] = pd.DataFrame(data, index=keys).astype(float).dropna(how="all")
        return out

    def info(self, tk):
        if tk not in self.meta:
            return {}
        mt = self.meta[tk]
        return {"longName": f"Synthetic {tk}", "currency": mt["ccy"], "financialCurrency": mt["fin"],
                "sector": C.SECTOR.get(tk, "Other"), "beta": mt["beta"]["MKT"]}

    # SEC-style full history with filing dates (US names only)
    def companyfacts_frames(self, tk):
        if tk not in self.meta or self.meta[tk]["sg"]:
            return None
        return self.annual_accounts(tk)


def _seed(s: str) -> int:
    import zlib
    return zlib.crc32(s.encode())


_FLOW_Y = {"TotalRevenue", "CostOfRevenue", "GrossProfit", "NetIncomeCommonStockholders", "NetIncome",
           "OperatingCashFlow", "CapitalExpenditure", "FreeCashFlow", "CashDividendsPaid"}


# ---------------------------------------------------------------------- SEC EDGAR mock
_SEC_MAP = {
    "TotalRevenue": ("us-gaap", "Revenues", "USD", 1), "CostOfRevenue": ("us-gaap", "CostOfRevenue", "USD", 1),
    "GrossProfit": ("us-gaap", "GrossProfit", "USD", 1),
    "NetIncomeCommonStockholders": ("us-gaap", "NetIncomeLoss", "USD", 1),
    "DilutedEPS": ("us-gaap", "EarningsPerShareDiluted", "USD/shares", 1),
    "DilutedAverageShares": ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding", "shares", 1),
    "OperatingCashFlow": ("us-gaap", "NetCashProvidedByUsedInOperatingActivities", "USD", 1),
    "CapitalExpenditure": ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment", "USD", -1),
    "CashDividendsPaid": ("us-gaap", "PaymentsOfDividendsCommonStock", "USD", -1),
    "StockholdersEquity": ("us-gaap", "StockholdersEquity", "USD", 1), "TotalAssets": ("us-gaap", "Assets", "USD", 1),
    "TotalDebt": ("us-gaap", "LongTermDebt", "USD", 1), "CurrentAssets": ("us-gaap", "AssetsCurrent", "USD", 1),
    "CurrentLiabilities": ("us-gaap", "LiabilitiesCurrent", "USD", 1),
    "CashAndCashEquivalents": ("us-gaap", "CashAndCashEquivalentsAtCarryingValue", "USD", 1),
}
_FLOW = {"TotalRevenue", "CostOfRevenue", "GrossProfit", "NetIncomeCommonStockholders", "DilutedEPS",
         "DilutedAverageShares", "OperatingCashFlow", "CapitalExpenditure", "CashDividendsPaid"}


def companyfacts_json(prov: SyntheticProvider, tk: str) -> dict:
    """EDGAR-shaped facts. Each 10-K reports the year and restates the prior year (x1.02);
    each 10-Q reports year-to-date flows with prior-year comparatives."""
    acc = prov.companyfacts_frames(tk)
    if not acc:
        return {}
    allacc = prov._accounts_all(tk)
    qacc = prov.quarterly_accounts(tk)
    facts: dict = {}

    def add(tax, tag, unit, f):
        facts.setdefault(tax, {}).setdefault(tag, {"units": {}})["units"].setdefault(unit, []).append(f)

    ends = sorted(acc)
    for n, end in enumerate(ends):
        filed = end + pd.Timedelta(days=40 + (n % 3) * 5)
        accn = f"{_seed(tk) % 10**10:010d}-{end.year % 100:02d}-000001"
        for yr_end, scale in ((end, 1.0), (ends[n - 1], 1.02) if n > 0 else (None, None)):
            if yr_end is None:
                continue
            for field, (tax, tag, unit, sign) in _SEC_MAP.items():
                val = acc[yr_end].get(field)
                if val is None:
                    continue
                f = {"end": str(yr_end.date()), "val": float(val) * sign * scale, "accn": accn,
                     "fy": end.year, "fp": "FY", "form": "10-K", "filed": str(filed.date())}
                if field in _FLOW:
                    f["start"] = str((yr_end - pd.DateOffset(years=1) + pd.Timedelta(days=1)).date())
                add(tax, tag, unit, f)
        add("dei", "EntityCommonStockSharesOutstanding", "shares",
            {"end": str((filed - pd.Timedelta(days=10)).date()), "val": acc[end]["OrdinarySharesNumber"],
             "accn": accn, "fy": end.year, "fp": "FY", "form": "10-K", "filed": str(filed.date())})
    # 10-Qs for Q1-Q3 of every fiscal year whose prior year is published
    fyes = sorted(allacc)
    for n, fye in enumerate(fyes[1:], 1):
        prev_fye = fyes[n - 1]
        fy_start = prev_fye + pd.Timedelta(days=1)
        for qi in range(3):
            qend = (fye - pd.DateOffset(months=3 * (3 - qi))) + pd.offsets.MonthEnd(0)
            filed = qend + pd.Timedelta(days=35)
            if filed > prov.now:
                continue
            accn = f"{_seed(tk) % 10**10:010d}-{qend.year % 100:02d}-1{qend.month:02d}01"
            for field, (tax, tag, unit, sign) in _SEC_MAP.items():
                if field in _FLOW:
                    if field in ("DilutedAverageShares",):
                        continue
                    cur = allacc[fye].get(field)
                    py = allacc[prev_fye].get(field)
                    if cur is None or py is None:
                        continue
                    frac = sum(SyntheticProvider.QSHARE[: qi + 1])
                    for val, st, en in ((cur * frac, fy_start, qend),
                                        (py * frac, fy_start - pd.DateOffset(years=1), qend - pd.DateOffset(years=1))):
                        add(tax, tag, unit, {"start": str(st.date()), "end": str((en + pd.offsets.MonthEnd(0)).date()),
                                             "val": float(val) * sign, "accn": accn, "form": "10-Q",
                                             "fp": f"Q{qi + 1}", "filed": str(filed.date())})
                else:
                    val = qacc[qend].get(field)
                    if val is None:
                        continue
                    add(tax, tag, unit, {"end": str(qend.date()), "val": float(val) * sign, "accn": accn,
                                         "form": "10-Q", "fp": f"Q{qi + 1}", "filed": str(filed.date())})
            add("dei", "EntityCommonStockSharesOutstanding", "shares",
                {"end": str((filed - pd.Timedelta(days=10)).date()), "val": qacc[qend]["OrdinarySharesNumber"],
                 "accn": accn, "form": "10-Q", "fp": f"Q{qi + 1}", "filed": str(filed.date())})
    return {"facts": facts}


def patch_sec(prov: SyntheticProvider):
    """Point ffq.sec at the synthetic EDGAR instead of the network."""
    import ffq.sec as S
    tickers = [t for t in prov.meta if not t.endswith(".SI")]
    cmap = {t: 1000 + k for k, t in enumerate(tickers)}
    rev = {v: k for k, v in cmap.items()}
    S.cik_map = lambda: cmap
    S.companyfacts = lambda cik: companyfacts_json(prov, rev[cik])
