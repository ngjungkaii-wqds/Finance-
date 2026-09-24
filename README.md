# ACC2026 FFQ strategy

A stock-selection and risk-management system for the ACC2026 trading competition (US and Singapore stocks,
USD 1m, 14 Sep – 13 Nov 2026). It is built on the Fama–French model, arbitrage pricing theory, the security
market line and Markowitz portfolio theory.

- **What the strategy is and why:** [STRATEGY.md](STRATEGY.md)
- **Backtest results:** `reports/BACKTEST_REPORT.md` (created when you run the backtest)
- **This week's book:** `reports/SCREEN_<date>.md` (created when you run the screen)

## Run it on your laptop (Windows, Mac or Linux)

```bash
pip install -r requirements.txt

python -m pytest -q tests                       # 1 minute, offline: checks the code works
python screener.py                              # this week's book for a USD 1m account
python screener.py --holdings holdings.csv      # ...plus buy/sell orders from what you hold now
python run_backtest.py --quick                  # first look at the backtest (every 3rd game)
python run_backtest.py --sec                    # full backtest, Yahoo + SEC EDGAR (20-40 minutes)
```

- **Holdings file:** copy `holdings_template.csv` to `holdings.csv` and type your actual share counts.
- **SEC EDGAR:** before a `--sec` run, set your contact so the SEC accepts the requests:
  - Windows: `set SEC_USER_AGENT=Your Name your.email@school.edu`
  - Mac/Linux: `export SEC_USER_AGENT="Your Name your.email@school.edu"`
- **Cache:** downloads are cached in `data_cache/`, so a second run takes seconds. Delete the folder to force fresh data.

Useful screen options:

| Option | Effect |
| --- | --- |
| `--deploy 0.66` | invest only 66% of the account this round (the rest stays cash) |
| `--capital 1012000` | the account's current value in USD |
| `--exclude CAT 558.SI` | remove stocks that failed your red-flag check on Bloomberg |
| `--asof 2026-06-30` | the screen as it would have looked on that date |

## Run it from your phone (GitHub Actions)

GitHub → this repository → **Actions** → **FFQ backtest and screen** → **Run workflow**. Choose `both`,
`screen` or `backtest`. The results are committed to `reports/` on the branch when it finishes.

- If you commit a `holdings.csv`, the screen also produces orders.
- Yahoo sometimes throttles cloud servers. If the run fails with a rate-limit error, run it again later or on your laptop.

## What is in the repository

| Path | Contents |
| --- | --- |
| `ffq/config.py` | Every rule and parameter, fixed on 24 Sep 2026 before any backtest |
| `ffq/data.py` | Yahoo prices, FX, T-bill, factor ETFs, statements; USD conversion; cache |
| `ffq/fundamentals.py` | Point-in-time accounts, TTM, Piotroski F-score, the fundamental gate |
| `ffq/sec.py` | Optional SEC EDGAR 10-K and 10-Q accounts with filing dates |
| `ffq/factors.py` | FF3 regressions and residual momentum, CAPM beta and SML, APT loadings |
| `ffq/risk.py` | Covariance model, correlation limit, max-Sharpe optimiser, capital allocation line, risk report |
| `ffq/strategy.py` | The whole strategy on one date |
| `ffq/legacy.py` | The old momentum rulebook, rebuilt for comparison |
| `ffq/backtest.py`, `ffq/metrics.py`, `ffq/report.py` | Game simulator, statistics, report and charts |
| `screener.py`, `run_backtest.py` | The two programs you run |
| `tests/` | Offline tests on a synthetic market with the same data shapes as Yahoo |

All data comes from Yahoo Finance (and, optionally, SEC EDGAR). Check prices and accounts on Bloomberg
before trading. The code prints Bloomberg tickers for that.
