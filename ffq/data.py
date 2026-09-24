"""
Data layer: Yahoo Finance prices, FX, T-bill rate, factor ETFs and financial statements.

Everything is cached under ./data_cache so a second run is fast and reproducible.
A "provider" object hides where the data comes from, so the tests can swap in a
synthetic market with exactly the same shapes as yfinance returns.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C

warnings.filterwarnings("ignore")

CACHE_DIR = os.environ.get("FFQ_CACHE", os.path.join(os.getcwd(), "data_cache"))
PRICE_FIELDS = ["Open", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]


def _cache_path(*parts: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, *parts)


def _fresh(path: str, max_age_hours: float) -> bool:
    return os.path.exists(path) and (time.time() - os.path.getmtime(path)) < max_age_hours * 3600


# --------------------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------------------
class YahooProvider:
    """Downloads from Yahoo Finance through yfinance, with retries and a disk cache."""

    def __init__(self, max_age_hours: float = 12.0, fund_max_age_hours: float = 24 * 7,
                 verbose: bool = True):
        import yfinance as yf  # imported lazily so tests do not need network access
        self.yf = yf
        self.max_age = max_age_hours
        self.fund_max_age = fund_max_age_hours
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    # ---- prices ------------------------------------------------------------------
    def history(self, tickers: list[str], start: str, end: str | None = None) -> dict:
        """Return {field: DataFrame(dates x tickers)} for PRICE_FIELDS."""
        tickers = sorted(set(tickers))
        key = hashlib.md5(("|".join(tickers) + start + str(end)).encode()).hexdigest()[:12]
        path = _cache_path(f"prices_{key}.pkl")
        if _fresh(path, self.max_age):
            with open(path, "rb") as fh:
                return pickle.load(fh)
        frames = {f: [] for f in PRICE_FIELDS}
        chunks = [tickers[i:i + 40] for i in range(0, len(tickers), 40)]
        for n, chunk in enumerate(chunks, 1):
            self._log(f"  prices: chunk {n}/{len(chunks)} ({len(chunk)} tickers)")
            raw = None
            for attempt in range(4):
                try:
                    raw = self.yf.download(chunk, start=start, end=end, auto_adjust=False,
                                           actions=True, progress=False, threads=True,
                                           group_by="column", multi_level_index=True)
                    if raw is not None and not raw.empty:
                        break
                except Exception as exc:  # rate limit or network hiccup
                    self._log(f"    retry {attempt + 1}: {type(exc).__name__}: {exc}")
                time.sleep(2 ** (attempt + 1))
            if raw is None or raw.empty:
                self._log(f"    chunk {n} returned no data")
                continue
            for f in PRICE_FIELDS:
                if isinstance(raw.columns, pd.MultiIndex):
                    if f in raw.columns.get_level_values(0):
                        frames[f].append(raw[f])
                elif f in raw.columns and len(chunk) == 1:
                    frames[f].append(raw[[f]].rename(columns={f: chunk[0]}))
        out = {}
        for f in PRICE_FIELDS:
            if frames[f]:
                df = pd.concat(frames[f], axis=1)
                df = df.loc[:, ~df.columns.duplicated()]
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                out[f] = df.sort_index()
            else:
                out[f] = pd.DataFrame()
        with open(path, "wb") as fh:
            pickle.dump(out, fh)
        return out

    # ---- statements --------------------------------------------------------------
    def statements(self, ticker: str, quarterly: bool = False) -> dict:
        """Annual (and optionally quarterly) income, balance and cash-flow tables.

        Rows are Yahoo's CamelCase field names (e.g. "NetIncome"), columns are
        fiscal period end dates, newest first. Yahoo serves at most 4 years.
        """
        path = _cache_path("fund", f"{ticker.replace('^', '_')}{'_q' if quarterly else ''}.pkl")
        if _fresh(path, self.fund_max_age):
            with open(path, "rb") as fh:
                return pickle.load(fh)
        t = self.yf.Ticker(ticker)
        out = {}
        freqs = [("yearly", "")] + ([("quarterly", "q_")] if quarterly else [])
        for freq, pre in freqs:
            for name, fn in (("income", t.get_income_stmt), ("balance", t.get_balance_sheet),
                             ("cashflow", t.get_cash_flow)):
                df = pd.DataFrame()
                for attempt in range(3):
                    try:
                        df = fn(pretty=False, freq=freq)
                        break
                    except Exception as exc:
                        self._log(f"    {ticker} {pre}{name} retry {attempt + 1}: {type(exc).__name__}")
                        time.sleep(2 ** (attempt + 1))
                out[pre + name] = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(out, fh)
        time.sleep(0.2)
        return out

    def info(self, ticker: str) -> dict:
        path = _cache_path("info", f"{ticker.replace('^', '_')}.json")
        if _fresh(path, self.fund_max_age):
            with open(path) as fh:
                return json.load(fh)
        info = {}
        for attempt in range(3):
            try:
                info = dict(self.yf.Ticker(ticker).info or {})
                break
            except Exception as exc:
                self._log(f"    {ticker} info retry {attempt + 1}: {type(exc).__name__}")
                time.sleep(2 ** (attempt + 1))
        keep = ["longName", "shortName", "currency", "financialCurrency", "sector", "industry",
                "marketCap", "sharesOutstanding", "beta", "trailingPE", "forwardPE",
                "targetMeanPrice", "recommendationMean", "numberOfAnalystOpinions",
                "dividendYield", "trailingAnnualDividendYield", "earningsTimestamp",
                "earningsTimestampStart", "exchange", "quoteType"]
        info = {k: info.get(k) for k in keep if info.get(k) is not None}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(info, fh, default=str)
        time.sleep(0.2)
        return info


# --------------------------------------------------------------------------------------
# The assembled market panel
# --------------------------------------------------------------------------------------
@dataclass
class Market:
    dates: pd.DatetimeIndex
    tickers: list                  # stock tickers with usable data
    px: pd.DataFrame               # total-return (dividend-adjusted) close, USD, ffilled
    open_: pd.DataFrame            # total-return-adjusted open, USD
    raw: pd.DataFrame              # split-adjusted close without dividend adjustment, USD
    raw_local: pd.DataFrame        # same, local quote currency (for share counts)
    dollar_vol: pd.DataFrame       # USD traded value
    divs_local: pd.DataFrame       # cash dividends per share, local currency
    traded: pd.DataFrame           # True on days the stock actually printed a price
    fx: dict                       # currency -> Series of USD per unit
    rf_annual: pd.Series           # T-bill yield, decimal a year, daily
    etf: pd.DataFrame              # factor ETF total-return prices (USD)
    index_px: pd.DataFrame         # ^GSPC, ^STI price levels
    quote_ccy: dict
    fin_ccy: dict
    splits: dict = field(default_factory=dict)   # ticker -> Series of split ratios (Yahoo)
    info: dict = field(default_factory=dict)
    problems: list = field(default_factory=list)

    def country(self, tk: str) -> str:
        return "SG" if tk.endswith(".SI") else "US"

    def loc(self, date) -> int:
        """Position of the last calendar date on or before `date`."""
        i = self.dates.searchsorted(pd.Timestamp(date), side="right") - 1
        if i < 0:
            raise ValueError(f"{date} is before the start of the data")
        return int(i)


def _quote_ccy(tk: str, info: dict) -> str:
    ccy = (info or {}).get("currency")
    if ccy:
        return str(ccy).upper()
    if tk.endswith(".SI"):
        return "USD" if tk in C.SG_USD_QUOTED else "SGD"
    return "USD"


def _fin_ccy(tk: str, info: dict, quote: str) -> str:
    ccy = (info or {}).get("financialCurrency")
    if ccy:
        return str(ccy).upper()
    return C.FIN_CCY_FALLBACK.get(tk, quote)


def _clean_spikes(s: pd.Series) -> pd.Series:
    """Remove one-day price spikes that fully reverse (bad prints), not real moves."""
    r = s.pct_change()
    bad = (r.abs() > 0.5) & (r.shift(-1).abs() > 0.3) & (np.sign(r) != np.sign(r.shift(-1)))
    if bad.any():
        s = s.copy()
        s[bad] = np.nan
    return s


def build_market(provider, stocks: list[str], start: str, end: str | None = None,
                 with_info: bool = True, verbose: bool = True) -> Market:
    """Download everything and build USD panels on one combined SG+US calendar."""
    fx_tk = sorted(set(C.FX_TICKERS.values()))
    all_tk = sorted(set(stocks)) + C.FACTOR_ETFS + C.INDEX_TICKERS + fx_tk + [C.RF_TICKER]
    if verbose:
        print(f"Downloading prices for {len(all_tk)} series from {start}", flush=True)
    h = provider.history(all_tk, start=start, end=end)
    close, adj, opn = h["Close"], h["Adj Close"], h["Open"]
    vol = h["Volume"]
    divs = h["Dividends"] if not h["Dividends"].empty else pd.DataFrame(0.0, index=close.index, columns=close.columns)

    problems = []
    info = {}
    if with_info:
        for tk in stocks:
            try:
                info[tk] = provider.info(tk)
            except Exception:
                info[tk] = {}

    # FX: USD per unit of currency, forward filled on the full calendar
    fx = {"USD": None}
    for ccy, tk in C.FX_TICKERS.items():
        if tk in close.columns and close[tk].notna().sum() > 100:
            fx[ccy] = close[tk]

    good = []
    for tk in stocks:
        if tk not in close.columns or close[tk].dropna().shape[0] < 60:
            problems.append(f"{tk}: no price data")
            continue
        good.append(tk)

    # combined calendar: every weekday on which at least one stock traded
    traded_any = close[good].notna().any(axis=1)
    dates = close.index[traded_any]
    dates = dates[dates.dayofweek < 5]
    if len(dates) == 0:
        raise RuntimeError("No price data at all: is Yahoo Finance reachable?")

    def fx_series(ccy):
        if ccy == "USD" or fx.get(ccy) is None:
            return pd.Series(1.0, index=dates)
        return fx[ccy].reindex(close.index).ffill().bfill().reindex(dates).ffill().bfill()

    quote_ccy, fin_ccy = {}, {}
    px, op, rw, rwl, dv, dvl, trd = {}, {}, {}, {}, {}, {}, {}
    for tk in good:
        q = _quote_ccy(tk, info.get(tk, {}))
        if q not in fx and q != "USD":
            problems.append(f"{tk}: quote currency {q} has no FX series; treated as USD")
            q = "USD"
        quote_ccy[tk] = q
        fin_ccy[tk] = _fin_ccy(tk, info.get(tk, {}), q)
        f = fx_series(q)
        c = _clean_spikes(close[tk]).reindex(dates)
        a = adj[tk].reindex(dates) if tk in adj.columns else c
        a = a.where(c.notna())
        o = opn[tk].reindex(dates) if tk in opn.columns else c
        o = o.where((o > 0) & c.notna(), c)          # a zero or missing open falls back to the close
        ratio = (a / c).where(c > 0)
        trd[tk] = c.notna()
        first = c.first_valid_index()
        # forward fill at most 5 days (holidays, suspensions shorter than a week)
        c_f = c.ffill(limit=5)
        a_f = a.ffill(limit=5)
        px[tk] = a_f * f
        op[tk] = (o * ratio).where(c.notna()) * f
        rw[tk] = c_f * f
        rwl[tk] = c_f
        v = vol[tk].reindex(dates) if tk in vol.columns else pd.Series(np.nan, index=dates)
        dv[tk] = (v * c * f)
        dvl[tk] = divs[tk].reindex(dates).fillna(0.0) if tk in divs.columns else pd.Series(0.0, index=dates)
        if first is not None:
            for s in (px[tk], rw[tk], rwl[tk]):
                s.loc[s.index < first] = np.nan

    etf = pd.DataFrame({e: adj[e].reindex(dates).ffill(limit=5) for e in C.FACTOR_ETFS if e in adj.columns})
    idx = pd.DataFrame({e: close[e].reindex(dates).ffill(limit=5) for e in C.INDEX_TICKERS if e in close.columns})
    if C.RF_TICKER in close.columns and close[C.RF_TICKER].notna().sum() > 20:
        rf = (close[C.RF_TICKER].reindex(close.index).ffill().reindex(dates).ffill().bfill() / 100.0)
    else:
        problems.append("T-bill rate (^IRX) missing: using 4% a year")
        rf = pd.Series(0.04, index=dates)
    missing_etf = [e for e in C.FACTOR_ETFS if e not in etf.columns or etf[e].notna().sum() < 100]
    if missing_etf:
        problems.append(f"factor ETFs missing: {missing_etf}")

    fx_full = {k: fx_series(k) for k in fx}
    spl = h.get("Stock Splits", pd.DataFrame())
    splits = {}
    for tk in good:
        if tk in spl.columns:
            sp = spl[tk].dropna()
            sp = sp[sp > 0]
            if len(sp):
                splits[tk] = sp
    return Market(dates=dates, tickers=good, px=pd.DataFrame(px), open_=pd.DataFrame(op),
                  raw=pd.DataFrame(rw), raw_local=pd.DataFrame(rwl), dollar_vol=pd.DataFrame(dv),
                  divs_local=pd.DataFrame(dvl), traded=pd.DataFrame(trd), fx=fx_full,
                  rf_annual=rf.clip(lower=0.0), etf=etf, index_px=idx, quote_ccy=quote_ccy,
                  fin_ccy=fin_ccy, splits=splits, info=info, problems=problems)


def weekly_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Friday-to-Friday simple returns from a daily price panel.

    Weekly returns sidestep the time-zone problem: Singapore closes about 12 hours
    before New York, which makes daily SG-US correlations look too low.
    """
    wk = panel.resample("W-FRI").last()
    return wk.pct_change(fill_method=None)


def us_liquid(m: Market, i: int, pool: list[str], top_n: int) -> list[str]:
    """Top-N US names by 252-day median USD traded value, known at calendar position i."""
    lo = max(0, i - 251)
    dv = m.dollar_vol.iloc[lo:i + 1][[t for t in pool if t in m.dollar_vol.columns]]
    med = dv.median(skipna=True)
    enough = dv.notna().sum() >= 200
    med = med[enough].dropna().sort_values(ascending=False)
    return list(med.index[:top_n])
