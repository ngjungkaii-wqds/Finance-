"""Tests of the parts that only real data exercises: yfinance shapes, TTM checks, share bases, SEC quirks."""
import numpy as np
import pandas as pd
import pytest

from ffq import fundamentals as FU
from ffq import sec as S
from ffq.data import YahooProvider


class FakeYF:
    """Mimics yfinance.download's column layout: MultiIndex (Price, Ticker), tz-naive dates."""

    def __init__(self):
        self.calls = 0

    def download(self, tickers, start=None, end=None, auto_adjust=False, actions=True, progress=False,
                 threads=True, group_by="column", multi_level_index=True):
        self.calls += 1
        idx = pd.bdate_range("2024-01-01", periods=5)
        fields = ["Adj Close", "Close", "Dividends", "High", "Low", "Open", "Stock Splits", "Volume"]
        cols = pd.MultiIndex.from_product([fields, sorted(tickers)], names=["Price", "Ticker"])
        df = pd.DataFrame(np.arange(len(idx) * len(cols), dtype=float).reshape(len(idx), len(cols)) + 1,
                          index=idx, columns=cols)
        return df

    class Ticker:
        def __init__(self, tk):
            self.tk = tk

        def get_income_stmt(self, pretty=False, freq="yearly"):
            return pd.DataFrame({pd.Timestamp("2025-12-31"): [100.0, 10.0], pd.Timestamp("2024-12-31"): [90.0, 9.0]},
                                index=["TotalRevenue", "NetIncomeCommonStockholders"])

        def get_balance_sheet(self, pretty=False, freq="yearly"):
            return pd.DataFrame({pd.Timestamp("2025-12-31"): [50.0, 200.0], pd.Timestamp("2024-12-31"): [45.0, 190.0]},
                                index=["StockholdersEquity", "TotalAssets"])

        def get_cash_flow(self, pretty=False, freq="yearly"):
            return pd.DataFrame({pd.Timestamp("2025-12-31"): [12.0], pd.Timestamp("2024-12-31"): [11.0]},
                                index=["OperatingCashFlow"])

        @property
        def info(self):
            return {"currency": "SGD", "marketCap": 1e9, "longName": "Fake"}


def test_yahoo_provider_parses_download_shape(tmp_path, monkeypatch):
    monkeypatch.setattr("ffq.data.CACHE_DIR", str(tmp_path))
    p = YahooProvider.__new__(YahooProvider)
    p.yf, p.max_age, p.fund_max_age, p.verbose = FakeYF(), 1, 1, False
    h = p.history(["AAA", "BBB.SI", "^IRX"], start="2024-01-01")
    assert set(h) >= {"Open", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"}
    assert list(h["Close"].columns) == ["AAA", "BBB.SI", "^IRX"]
    st = p.statements("AAA", quarterly=True)
    per = FU.periods_from_tables(st)
    assert [x.end.year for x in per] == [2024, 2025]
    assert per[-1].v["net_income"] == 10.0 and per[-1].v["ocf"] == 12.0
    assert p.info("AAA")["currency"] == "SGD"


def _period(end, **kw):
    v = {k: np.nan for k in FU.FIELDS}
    v.update(kw)
    return FU.Period(end=pd.Timestamp(end), avail=pd.Timestamp(end) + pd.Timedelta(days=45), v=v)


def test_check_ttm_drops_double_counted_half_years():
    acc = FU.Accounts(annual=[_period("2025-12-31", revenue=100.0, net_income=10.0, equity=50.0)],
                      ttm=[_period("2026-06-30", revenue=200.0, net_income=20.0),
                           _period("2026-03-31", revenue=104.0, net_income=10.5)])
    notes = FU.check_ttm(acc)
    assert len(acc.ttm) == 1 and acc.ttm[0].v["revenue"] == 104.0 and notes


def test_share_basis_rescaled_to_market_value():
    acc = FU.Accounts(annual=[_period("2025-12-31", shares=2.0e6, eps=5000.0, net_income=10.0, equity=5.0)])
    notes = FU.fix_share_basis(acc, {"marketCap": 1.0e9}, 500.0)     # 2m shares x 500 = 1bn: consistent
    assert notes == []
    acc2 = FU.Accounts(annual=[_period("2025-12-31", shares=1.44e3, eps=np.nan, net_income=10.0, equity=5.0)])
    notes = FU.fix_share_basis(acc2, {"marketCap": 1.0e9}, 500.0)   # class-A count vs B price
    assert notes and acc2.annual[0].v["shares"] == pytest.approx(2.0e6)


def test_ttm_from_half_years():
    q = [_period(e, revenue=r, net_income=n, equity=50.0) for e, r, n in
         (("2024-12-31", 50, 5), ("2025-06-30", 48, 4), ("2025-12-31", 52, 6), ("2026-06-30", 55, 7))]
    ttm = FU.ttm_from_quarters(q)
    last = [t for t in ttm if t.end == pd.Timestamp("2026-06-30")][0]
    assert last.v["revenue"] == 107 and last.v["net_income"] == 13


def test_sec_multiclass_shares_and_debt():
    fact = lambda **k: {"form": "10-K", "filed": "2026-02-01", "accn": "A1", **k}
    js = {"facts": {
        "us-gaap": {
            "NetIncomeLoss": {"units": {"USD": [fact(start="2025-01-01", end="2025-12-31", val=100.0)]}},
            "StockholdersEquity": {"units": {"USD": [fact(end="2025-12-31", val=500.0)]}},
            "LongTermDebt": {"units": {"USD": [fact(end="2025-12-31", val=300.0)]}},
            "LongTermDebtCurrent": {"units": {"USD": [fact(end="2025-12-31", val=50.0)]}},
            "ShortTermBorrowings": {"units": {"USD": [fact(end="2025-12-31", val=20.0)]}},
        },
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            fact(end="2026-01-20", val=5.8e9), fact(end="2026-01-20", val=0.86e9), fact(end="2026-01-20", val=5.5e9)]}}},
    }}
    per = S.periods_from_companyfacts(js)
    assert len(per) == 1
    v = per[0].v
    assert v["shares"] == pytest.approx(12.16e9)          # classes A + B + C
    assert v["debt"] == pytest.approx(320.0)              # LongTermDebt already includes its current part
    assert per[0].avail == pd.Timestamp("2026-02-01")
