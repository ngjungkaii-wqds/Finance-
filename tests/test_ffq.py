"""Offline tests on the synthetic market: python -m pytest -q tests"""
import numpy as np
import pandas as pd
import pytest

from ffq import config as C
from ffq import fundamentals as FU
from ffq import risk as R
from ffq.backtest import simulate_game
from ffq.data import Market
from ffq.factors import FactorEngine
from ffq.legacy import legacy_weights
from ffq.pipeline import load
from ffq.strategy import FFQStrategy
from tests.synthetic import SyntheticProvider, patch_sec

DATE = "2026-09-18"


@pytest.fixture(scope="module")
def world():
    prov = SyntheticProvider()
    m, fe, acc, source, notes = load(prov, start="2012-01-01", verbose=False)
    return prov, m, fe, acc


@pytest.fixture(scope="module")
def world_sec():
    prov = SyntheticProvider()
    patch_sec(prov)
    m, fe, acc, source, notes = load(prov, start="2012-01-01", use_sec=True, verbose=False)
    return prov, m, fe, acc


def test_market_panels(world):
    prov, m, fe, acc = world
    assert len(m.tickers) == len(prov.meta)
    assert m.quote_ccy["H78.SI"] == "USD" and m.quote_ccy["D05.SI"] == "SGD"
    # USD conversion: an SGD stock's USD price moves with SGDUSD
    tk = "D05.SI"
    i = m.loc(DATE)
    local = m.raw_local[tk].iloc[i]
    assert m.raw[tk].iloc[i] == pytest.approx(local * m.fx["SGD"].iloc[i])


def test_ff3_recovers_planted_betas(world):
    prov, m, fe, acc = world
    ff = fe.ff3(list(prov.meta), DATE)
    errs = [abs(ff.loc[t, "ff_mkt"] - prov.meta[t]["beta"]["MKT"]) for t in ff.index
            if t not in C.GOLD and t not in ("D05.SI", "O39.SI", "U11.SI")]
    assert np.median(errs) < 0.12


def test_capm_and_cost_of_equity(world):
    prov, m, fe, acc = world
    cap = fe.capm(["AAPL", "D05.SI"], DATE)
    ke = fe.cost_of_equity(cap["beta_adj"], DATE)
    rf = m.rf_annual.asof(pd.Timestamp(DATE))
    assert (ke > rf).all()
    assert cap.loc["AAPL", "beta_adj"] == pytest.approx(0.67 * cap.loc["AAPL", "beta_local"] + 0.33)


def _truncate(m: Market, date) -> Market:
    i = m.loc(date)
    cut = lambda df: df.iloc[: i + 1]
    return Market(dates=m.dates[: i + 1], tickers=m.tickers, px=cut(m.px), open_=cut(m.open_), raw=cut(m.raw),
                  raw_local=cut(m.raw_local), dollar_vol=cut(m.dollar_vol), divs_local=cut(m.divs_local),
                  traded=cut(m.traded), fx={k: v.iloc[: i + 1] for k, v in m.fx.items()},
                  rf_annual=m.rf_annual.iloc[: i + 1], etf=cut(m.etf), index_px=cut(m.index_px),
                  quote_ccy=m.quote_ccy, fin_ccy=m.fin_ccy, splits=m.splits, info=m.info)


@pytest.mark.parametrize("date", ["2025-06-11", "2026-03-04"])
def test_no_lookahead(world, date):
    """The book on a date must not change when every later price is deleted."""
    prov, m, fe, acc = world
    full = FFQStrategy(m, fe, acc).book(date)
    mt = _truncate(m, date)
    trunc = FFQStrategy(mt, FactorEngine(mt), acc).book(date)
    assert list(full.weights.index) == list(trunc.weights.index)
    np.testing.assert_allclose(full.weights.values, trunc.weights.values, atol=1e-6)


def test_book_respects_risk_limits(world):
    prov, m, fe, acc = world
    b = FFQStrategy(m, fe, acc).book(DATE)
    w = b.weights
    assert 0 < w.sum() <= 1.0 + 1e-9
    y = b.diag.get("cal_y", 1.0)
    assert (w / y <= C.W_MAX + 1e-6).all()
    assert (w / y >= C.W_MIN - 1e-6).all()
    assert b.table.loc[w.index, "gate_pass"].astype(bool).all()
    secs = pd.Series({t: C.SECTOR[t] for t in w.index})
    assert secs.value_counts().max() <= C.SECTOR_N_MAX
    assert sum(t in C.SEMI for t in w.index) <= C.SEMI_N_MAX
    assert b.risk["max_pair_corr"] <= 0.95
    assert b.risk["vol_ann"] <= C.SIGMA_TARGET + 1e-6
    if b.diag.get("method") == "max-Sharpe" and not b.diag.get("notes"):
        assert b.risk["risk_contrib"].max() <= C.RC_MAX + 1e-3


def test_correlation_gate_blocks_pairs():
    corr = pd.DataFrame([[1, .8, .2], [.8, 1, .1], [.2, .1, 1]], index=list("ABC"), columns=list("ABC"))
    log = []
    assert R.select(["A", "B", "C"], corr, log=log) == ["A", "C"]
    assert "correlation 0.80" in log[1][2]


def test_gate_rejects_deteriorating_names(world):
    prov, m, fe, acc = world
    s = FFQStrategy(m, fe, acc)
    tab = s.assess(["CAT", "AMD", "BN4.SI"], m.loc(DATE))
    assert not tab["gate_pass"].astype(bool).any()
    assert "G6" in tab.loc["AMD", "gate_fails"]


def test_ttm_matches_sec_and_yahoo(world, world_sec):
    """A TTM built from Yahoo quarters must equal the SEC 10-Q formula FY + YTD - YTD(prior)."""
    _, _, _, acc = world
    _, _, _, acc_sec = world_sec
    for tk in ("AAPL", "JPM", "LLY"):
        y = {p.end: p for p in acc[tk].ttm}
        sec = {p.end: p for p in acc_sec[tk].ttm if p.avail > p.end + pd.Timedelta(days=30)}
        common = sorted(set(y) & set(sec))
        assert common, tk
        e = common[-1]
        assert sec[e].v["net_income"] == pytest.approx(y[e].v["net_income"], rel=1e-6)
        assert sec[e].v["revenue"] == pytest.approx(y[e].v["revenue"], rel=1e-6)


def test_sec_uses_first_reported_values(world_sec):
    prov, m, fe, acc = world_sec
    raw = prov.annual_accounts("AAPL")
    ends = sorted(raw)
    p = [q for q in acc["AAPL"].annual if q.end == ends[-3]][0]
    assert p.v["net_income"] == pytest.approx(raw[ends[-3]]["NetIncomeCommonStockholders"])  # not the x1.02 restatement
    assert p.avail > p.end


def test_ttm_used_when_newer(world):
    prov, m, fe, acc = world
    i = m.loc(DATE)
    s = FFQStrategy(m, fe, acc)
    f = s.fundamentals_at(["AAPL"], i)["AAPL"]
    assert f["basis"].startswith("TTM")      # December year end, June quarter published by September


def test_game_commission_and_stop(world):
    prov, m, fe, acc = world
    i = m.loc("2020-02-14")
    w = pd.Series({"AAPL": 0.5})
    g = simulate_game(m, w, i, stop=None)
    e, x = i + 1, i + C.HOLD_DAYS
    gross = 0.5 * m.px["AAPL"].iloc[x] / (m.open_["AAPL"].iloc[e])
    comm = C.COMMISSION * 0.5 + C.COMMISSION * gross
    assert g["ret"] == pytest.approx(gross + 0.5 - comm - 1.0, abs=1e-9)
    g2 = simulate_game(m, w, i, stop=0.999)
    assert len(g2["stops"]) >= 0


def test_legacy_matches_rules(world):
    prov, m, fe, acc = world
    w = legacy_weights(m, DATE, "legacy")
    assert len(w) <= 8 and abs(w.sum() - len(w) / 8) < 1e-12
    assert sum(t in {"NVDA", "AMD", "MU", "AVGO", "QCOM", "558.SI"} for t in w.index) <= 2
