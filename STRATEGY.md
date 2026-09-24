# FFQ: a Fama–French quality strategy for ACC2026

**Written:** 24 Sep 2026. **Status:** rules fixed; backtest code complete and tested offline, not yet run on
real data (the session that wrote it could not reach Yahoo Finance; see section 8 and the README).
**Code:** `ffq/` (engine), `screener.py` (weekly book), `run_backtest.py` (backtest). All parameters are in `ffq/config.py`.
**Replaces:** the selection step of the momentum rulebook (`ACC2026_COMPLETE_RECORD.md` section 19). The −25% per-lot stop, the human red-flag check and the earnings protocol are kept.

The group's priorities, in order, and where each is handled:

| Priority | How FFQ meets it | Section |
| --- | --- | --- |
| 1. A good fundamental reason to own every stock | Six-test fundamental gate; quality and value in the ranking; a written case for each holding | 3.3, 3.4, 6 |
| 2. Investment risk management | Covariance model (APT + Ledoit–Wolf), max-pairwise correlation limit, sector, beta and risk-contribution caps, volatility target, stop | 3.5–3.8, 4 |
| 3. Sharpe ratio | Constrained maximum-Sharpe (tangency) portfolio; residual momentum instead of raw momentum | 3.4, 3.6 |
| 4. Returns | Momentum is kept, but measured after Fama–French factors | 3.4 |

---

## 1. Why change

The momentum screen earns its return, but it selects on price alone. The round-2 record says so directly:
*"Only Micron, Eli Lilly, Newmont and Yangzijiang have accounts that independently justify their price
moves. UMS, OCBC and Caterpillar are re-rating bets."* The next names down were worse: Wilmar (net
margin 2.0%, ROE 5.6%), Keppel (profit −15.9%), Hongkong Land (revenue −27.7%), AMD (trailing P/E 159).
The brief awards no marks for luck, and a report must explain the fundamentals of every stock.

The project's own tests point the same way. The mean return of the top momentum names comes from a few
extreme winners, while the median and the win rate barely change (round-2 record, section 6: top 1% of
momentum mean +10.2%, median +3.0%). In a single 9-week game you get something close to the median, not
the mean. Raw momentum also piles into whatever sector is hottest, which is why the universe-size test got
worse as the universe grew.

FFQ keeps momentum as the main source of return, but requires every stock to pass a fundamental gate first.
It also measures momentum after stripping out market, size and value effects, and sizes positions from the
covariance matrix instead of equally.

---

## 2. The theory, and where each model is used

### 2.1 CAPM and the security market line (SML)

The SML gives the return shareholders require for bearing a stock's systematic risk:

> k_i = r_f + β_i × (E[R_m] − r_f)

- **r_f** is the 13-week US Treasury bill yield (Yahoo `^IRX`, point in time).
- **E[R_m] − r_f** is the market risk premium, fixed at **5.5%** (the usual textbook range is 5–6%).
- **β_i** is estimated from two years of weekly USD returns against the stock's home market: the S&P 500 (SPY) for US stocks and MSCI Singapore (EWS) for Singapore stocks.
- β is then **Blume-adjusted**, β_adj = 0.67 β + 0.33, the same adjustment Bloomberg reports. It pulls noisy estimates towards 1 (Blume, 1971).

**Where it is used**

1. **The value-creation test (gate G2).** A company whose return on equity is below its cost of equity destroys value for shareholders; its price is only justified by hope of improvement. This is the residual-income logic: price to book above 1 is justified only when ROE > k. FFQ requires **ROE ≥ k_i**. This is the formal version of "the fundamentals must justify the price".
2. **The equilibrium part of expected return.** In the optimiser, each stock's expected return is its SML return plus an alpha (2.5).
3. **Performance evaluation.** The backtest reports Jensen's alpha (the intercept against the SML), the Treynor ratio and M².

### 2.2 The Fama–French three-factor model (FF3)

> r_i − r_f = α_i + b_i·MKT + s_i·SMB + h_i·HML + ε_i  (Fama & French, 1993)

Each factor is a tradable long-short portfolio built from Yahoo ETF prices:

| Factor | US stocks | Singapore stocks |
| --- | --- | --- |
| MKT (market) | SPY − T-bill | EWS − T-bill |
| SMB (small minus big) | IWM − IWB (Russell 2000 − Russell 1000) | SCZ − EFA (developed-market small caps − large) |
| HML (value minus growth) | IWD − IWF (Russell 1000 Value − Growth) | EFV − EFG (developed-market value − growth) |

**Where it is used**

1. **Residual momentum, the return signal.** For each stock, the betas are estimated on up to 156 weeks of returns ending one month ago. The signal is the stock's factor-adjusted return (α + ε) over months t−12 to t−1, divided by its residual volatility. This is the **appraisal ratio** of Treynor and Black (1973) over the formation window.
   - It asks whether the stock beat what the Fama–French model says it should have earned, given its exposures.
   - Blitz, Huij and Martens (2011) show that this "residual momentum" earns about the same return as plain momentum with about half the volatility, and avoids momentum crashes (Daniel & Moskowitz, 2016). Those crashes come from the factor bets that raw momentum accumulates.
2. **Describing each holding.** The screen reports b, s and h per stock, e.g. "a large-cap growth stock with market beta 1.3".
3. **FF5 motivation for the fundamentals.** Fama and French (2015) added **profitability (RMW)** and **investment (CMA)** factors because profitable, conservatively investing firms earn higher returns. FFQ does not estimate RMW and CMA loadings. Instead it measures profitability and value directly from the accounts (2.4), which is where those factors come from.

### 2.3 Arbitrage pricing theory (APT)

> r_i = E[r_i] + Σ_k b_ik F_k + e_i,  and by no-arbitrage  E[r_i] − r_f = Σ_k b_ik λ_k  (Ross, 1976)

APT allows any set of systematic factors. FFQ uses eight tradable macro factors, in the spirit of Chen, Roll and Ross (1986):
- **Markets:** US market (SPY), Singapore market (EWS)
- **Style:** US size (IWM−IWB), US value (IWD−IWF)
- **Macro:** interest rates (IEF − T-bill: 7–10 year Treasuries), oil (USO), gold (GLD), the US dollar (UUP)

**Where it is used**

1. **The covariance matrix (the risk model).** The multifactor structure gives
   > Σ_APT = B Σ_F B′ + D
   - B is the stocks × factors loadings.
   - Σ_F is the factor covariance, exponentially weighted with a 52-week half-life.
   - D is the diagonal matrix of stock-specific variances.

   This needs only 8 factor variances and covariances plus one residual variance per stock, instead of hundreds of pairwise covariances. That makes it far more stable than the sample matrix from three years of data.
2. **Macro exposure of the book.** The screen reports how much the book moves for a 1% move in oil, rates, gold or the dollar. It also reports how much of the risk is common-factor versus stock-specific.

### 2.4 Markowitz portfolio theory: covariance and correlation

> σ_p² = w′Σw = Σ_i Σ_j w_i w_j ρ_ij σ_i σ_j  (Markowitz, 1952)

Portfolio risk depends on the correlations between holdings, not just on each stock's own volatility. FFQ uses this at three points:

1. **Which covariance.**
   > Σ = ½ Σ_APT + ½ Σ_Ledoit–Wolf
   - The Ledoit–Wolf half is the sample covariance shrunk towards a structured target (Ledoit & Wolf, 2004). It keeps industry co-movements that the eight factors miss, such as the three Singapore banks.
   - Weekly (Friday-to-Friday) returns are used because Singapore closes about 12 hours before New York. Daily correlations between the two markets are therefore biased towards zero.
2. **Which stocks can sit together.**
   - No two holdings may have a correlation above **0.65**.
   - This is the maximum pairwise correlation, not the average used by the old rule. The old rule let DBS in with OCBC at 0.69 because the average hid the pair (risk register item 9).
3. **How much of each.**
   - The weights are those of the constrained tangency portfolio, the point on the efficient frontier with the highest Sharpe ratio.
   - The **capital allocation line** then decides how much of the account goes into it: invest y = min(1, 15% / σ_book), the rest in cash.
   - This is the textbook separation of choosing the risky portfolio from choosing how much risk to take (Bodie, Kane & Marcus, 2021, ch. 6–7).

**Risk contributions.** Each holding's share of portfolio variance, RC_i = w_i (Σw)_i / w′Σw, is capped at 20%. A high-volatility name cannot dominate the book even when its weight is modest.

### 2.5 Combining the SML with an active view (Treynor–Black, Grinold–Kahn)

Expected return for the optimiser:

> μ_i = k_i (SML) + α_i,  α_i = IC × σ_ε,i × z_i

- **z_i** is the stock's standardised score (2.2 and 3.4).
- **σ_ε,i** is its residual volatility from the FF3 regression.
- **IC = 0.05** is a deliberately modest information coefficient: the score is assumed to explain only a small part of future returns.

This is the Grinold and Kahn (2000) rule "alpha = IC × volatility × score". It follows the Treynor and Black (1973) idea: start from the equilibrium (SML) return, then tilt towards stocks with positive alpha in proportion to alpha over residual risk. With a small IC the optimiser stays close to a diversified, SML-consistent portfolio and only leans on the scores. This protects the book from over-trusting its own forecasts (DeMiguel, Garlappi & Uppal, 2009).

---

## 3. The rules

Run `python screener.py` each buying day. Every step uses only data public at the close of the screen date.

### 3.1 Universe

- **Singapore:** the same 46 liquid SGX names as the rulebook.
- **United States:** the **40 most liquid** of a pool of about 110 large caps: the S&P 100, the legacy 28, seven liquid semiconductor/storage names and the two gold miners. Liquidity is 252-day median dollar volume, measured on the screen date.
  - This replaces the hand-picked 28 that the round-2 record found hindsight-tilted (section 5). A liquidity rule is documented, repeatable and gave the same result as the hand-picked list.
- **Returns and currencies:**
  - All returns are in US dollars.
  - Hongkong Land (H78) and Hutchison Port Trust (NS8U) are quoted in USD on SGX, so they are no longer multiplied by SGD/USD (a small error in the old screener).
  - Accounts reported in another currency (e.g. Yangzijiang in CNY, Thai Beverage in THB) are converted before computing valuation ratios.

### 3.2 Per-stock estimates

For every stock:
- CAPM beta (local and to the S&P 500) and the SML cost of equity k_i (2.1)
- FF3 loadings, residual volatility and residual momentum (2.2)
- APT loadings (2.3)
- The fundamentals below

**Which accounts are used.** Figures come from the annual income statement, balance sheet and cash-flow statement.
- If a quarter or half-year newer than the last annual report has been published, profit, cash flow and valuation use the **trailing twelve months (TTM)**. Example: Micron's last annual EPS gives a P/E of about 140 at today's price, while its TTM P/E is about 21. Using a 15-month-old number would be poor justification.
- The Piotroski F-score always uses annual reports, as in the original paper.
- Accounts are used only after they are public:
  - Yahoo annual: fiscal year end + 75 days
  - Yahoo quarterly: quarter end + 45 days
  - SEC EDGAR: the filing date

### 3.3 The fundamental gate (all six must pass)

| Test | Rule | Why | Theory |
| --- | --- | --- | --- |
| **G1 Profitable** | Net income > 0 and operating cash flow > 0 (banks: net income only) | A loss-maker cannot be valued on earnings; profit without cash is an accounting claim | — |
| **G2 Creates value** | ROE ≥ SML cost of equity k_i | Earns more on shareholders' money than they require for its risk | CAPM/SML, residual income |
| **G3 Sound accounts** | Piotroski F-score ≥ 5/9 (scaled if a signal cannot be computed; at least 6 signals needed) | Nine accounting signals of improving profitability, liquidity and efficiency. Piotroski (2000) shows low scores predict poor returns | Accounting quality |
| **G4 Profits not falling** | Profit growth ≥ −10% (latest TTM vs a year earlier, or latest quarter vs the same quarter last year, or last fiscal year) | Excludes re-rating bets whose last results show falling profits | Earnings momentum |
| **G5 Balance sheet** | Non-financials debt/equity ≤ 2.0; REITs and trusts gearing ≤ 50% (MAS limit); banks and insurers equity/assets ≥ 5% | Leverage turns a bad quarter into a solvency question | Financial risk |
| **G6 Not absurdly priced** | 0 < trailing P/E ≤ 60 | An earnings yield below 1.7% requires heroic growth; the gate refuses to pay any price | Value (HML) |

A stock with missing data fails: we do not buy what we cannot justify.

**Soft flags.** These are not rules but are printed for the human check: accruals above 10% of assets, operating cash flow down more than 40%, and revenue down more than 15%. Yangzijiang's FY2025 cash-flow fall (error E21) is the kind of thing these flag.

### 3.4 Ranking score (only among stocks that pass)

z-scores are computed within each country (Singapore and US valuations are not comparable) and winsorised at ±3.

> score = 0.50 × z(momentum) + 0.25 × z(quality) + 0.25 × z(value)

- **Momentum:** the FF3 residual-momentum appraisal ratio (2.2).
- **Quality:** the average of the available z-scores of:
  - ROE − k (economic profit)
  - the F-score
  - profit growth (capped at ±100%)
  - −accruals (Sloan, 1996: profits backed by cash persist)

  This is the FF5 profitability (RMW) idea measured from the accounts (Novy-Marx, 2013).
- **Value:** the average of the z-scores of earnings yield, book-to-price and free-cash-flow yield. This is the FF HML idea.
  - Value and momentum are negatively correlated, so combining them raises the Sharpe ratio of either alone (Asness, Moskowitz & Pedersen, 2013).

Only stocks with score ≥ 0 are candidates, i.e. at or above the average of those that passed the gate.
The top 12 per country form the candidate list.

### 3.5 Selection with risk limits

Walk down the candidate list by score. Accept a stock unless:
- its correlation with any stock already accepted is above **0.65**;
- it would be the fourth stock from one sector (**sector cap: 3 names**);
- it would be the third semiconductor or storage name (**semiconductor cap: 2**); or
- it would be the second gold miner.

Stop at 12 stocks.

### 3.6 Weights: constrained maximum-Sharpe portfolio

Maximise (w′μ − r_f) / √(w′Σw), subject to:
- Long only.
- Each stock between **4% and 15%** of the book. Names the optimiser wants below 4% are dropped and it re-solves.
- **Sector ≤ 30%**; semiconductors ≤ 20%.
- **Each country ≥ 30%.** Singapore–US correlation is low, which is the cheapest diversification available.
- **Beta to the S&P 500 ≤ 1.0.** The book is never riskier in market terms than the benchmark it is judged against.
- **Risk contribution of any stock ≤ 20%.**

If the constraints cannot all hold, they are relaxed in a fixed order and the screen says which. If the maximum-Sharpe solution fails, the fallback is minimum variance, then capped inverse-variance.

### 3.7 How much to invest: the capital allocation line

Invest y = min(1, 15% / σ_book) of the account in the book, never less than 50%; the rest stays in cash.
- In calm markets y = 1.
- When volatility spikes, exposure falls automatically. Moreira and Muir (2017) show that this kind of volatility management raises Sharpe ratios.

### 3.8 Holding, exits and weekly trades

- **Stop:** sell a purchase that closes at or below 75% of its price; never rebuy (kept from the tested rulebook).
- **Fundamental break:** at each screen, a holding that now **fails the gate** (typically after new results) is sold at the next buying day. A holding that still passes and ranks in the top 24 is kept even if it is no longer in the target book. This avoids paying 0.5% round-trip commission to swap near-equals.
- **Broken case and red flags:** as in the rulebook (fraud, restatement, auditor resignation, regulatory action, suspension, or a guidance cut with 10+ points underperformance).
- **Weekly member trades (weeks 4–9):** each ~USD 2,000 trade moves the book towards its target: buy the most under-weight holding or trim the most over-weight. The compulsory trades then reduce tracking error instead of adding noise.
- **No-trade band:** trades under 1% of the account are skipped.

---

## 4. Risk management: every risk and the control that addresses it

| Risk | Control | Where |
| --- | --- | --- |
| Owning a company whose accounts do not justify the price | Gate G1–G6; soft flags; human red-flag check | 3.3 |
| One stock blowing up | 15% weight cap; 20% risk-contribution cap; −25% stop | 3.6, 3.8 |
| Two holdings that are really one bet (DBS/OCBC 0.69, MU/SNDK 0.78) | Max pairwise correlation 0.65 | 3.5 |
| Sector crowding (six of the missing US liquid names were semis) | 3 names / 30% per sector; semis 2 names / 20% | 3.5, 3.6 |
| Market crash | Beta ≤ 1.0 to the S&P 500; country minimum 30% each | 3.6 |
| Volatility regime change | Capital allocation line, 15% volatility target | 3.7 |
| Momentum crash (factor reversal) | Residual (factor-neutral) momentum; value in the score | 2.2, 3.4 |
| Estimation error in the model | Shrunk covariance (factor + Ledoit–Wolf); Blume beta; winsorised scores; small IC; weight bounds | 2.3–2.5 |
| Stale accounts | TTM overlay; point-in-time availability dates | 3.2 |
| Data errors | Share-count vs market-value check; currency checks; Bloomberg verification | code, 6 |
| Commission drag | Hold-if-still-good buffer; 1% no-trade band | 3.8 |

---

## 5. What changes from the momentum rulebook

| | Momentum rulebook | FFQ |
| --- | --- | --- |
| Return signal | 12-1 price momentum | 12-1 **residual** momentum after FF3 factors |
| Fundamentals | Human red-flag check only | Six-test gate + quality and value in the score |
| US universe | Hand-picked 28 | 40 most liquid of ~110 large caps, by rule |
| Correlation rule | Average correlation < 0.55 | **Maximum** pairwise correlation ≤ 0.65 |
| Concentration | 2 semis max | 3 per sector, 2 semis, 1 gold; weight, sector and risk caps |
| Weights | Equal (USD 40k per name per round) | Maximum-Sharpe from the covariance matrix |
| Size of the bet | Fixed Plan D | Capital allocation line (15% volatility target) |
| Market risk | Not controlled | Beta ≤ 1.0 to the S&P 500 |
| Kept unchanged | — | −25% per-lot stop, red-flag check, earnings protocol, Bloomberg verification |

---

## 6. Indicative gate check of the current and reserve names

These rows use the figures in the round-2 record (section 11, fiscal-year accounts and Yahoo betas), not the screener. The screener's own numbers are what count: it uses TTM accounts where newer, and the local-market beta.
Cost of equity is 4.0% + β_adj × 5.5%.

| Stock | ROE | k (approx.) | Profit growth | P/E | Likely result | Note |
| --- | --- | --- | --- | --- | --- | --- |
| Micron | 15.8% (FY; TTM far higher) | 14.0% | +998% | ~24 TTM (~141 on FY EPS) | **Pass** | Passes only because TTM accounts are used |
| Eli Lilly | 77.8% | 7.7% | +95% | ~39 | **Pass** | |
| Newmont | 20.8% | 7.8% | +112% | ~15 | **Pass** | |
| SGX | 29.6% | 6.8% | +7.8% | ~27 (fwd) | **Pass** | |
| Yangzijiang | 26.8% | 10.3% | +30% | ~12.5 | **Pass, flagged** | Operating cash flow −66% (soft flag) |
| Sheng Siong | 25.3% | 6.1% | +8.7% | ~32 | **Pass** | |
| OCBC | 11.7% | 10.4% (STI beta 1.25) | −2.2% | ~18 | **Pass, marginal** | Only ~1 point above k |
| UMS | 9.1% | 8.3–9.6% | +2.4% | ~43 | **Marginal** | Passes or fails G2 depending on the beta used |
| Caterpillar | 41.7% | 11.7% | **−17.7%** | ~34 | **Fails G4** on FY2025 | Could pass if TTM profit has recovered |
| AMD | 6.9% | 14.9% | +164% | **~159** | **Fails G2, G6** | |
| Wilmar | 5.6% | 6.2% | +20.6% | ~11 (fwd) | **Fails G2** | 2.0% net margin |
| Keppel | 7.2% | 7.7% | **−15.9%** | ~19 (fwd) | **Fails G2, G4** | |
| Hongkong Land | 4.1% | 7.1% | loss | n/a | **Fails G1, G2** | |

DBS and OCBC (correlation 0.69) cannot both be held under the 0.65 maximum-pair rule.

In short, the gate removes the names the round-2 record called re-rating bets or worse (Caterpillar, AMD, Wilmar, Keppel, Hongkong Land) and keeps the names whose accounts justify their prices.

---

## 7. Switching to FFQ mid-game

The game runs to 13 November; about seven weeks remain after the 25 September round.

1. **Before the 2 October round**, run
   `python screener.py --holdings holdings.csv --deploy <share of the account you want invested>`
   with your actual positions in `holdings.csv` (`holdings_template.csv` shows the format).
2. **Sell** holdings the screen marks SELL: they now fail the gate.
   - Record each sale as "fundamental gate: <reason>". It is a pre-stated rule, not a discretionary override.
3. **Keep** holdings marked "hold": they pass the gate and still rank in the top 24.
4. **Buy** the BUY lines with the round-3 money and the sale proceeds.
5. Each following Friday, re-run the screen. Weekly member trades move the book towards the targets.
   - New results that break the gate trigger a sale.
6. **Record the change** in the journal: the rulebook was replaced on [date] by FFQ, for the reasons in section 1. Rules are in `ffq/config.py`, fixed on 24 Sep 2026 before any backtest was seen.

---

## 8. The backtest (how it is run and how to read it)

`python run_backtest.py --sec` downloads everything from Yahoo Finance and, for US names, SEC EDGAR. It then writes `reports/BACKTEST_REPORT.md`.

**Design (same as the complete record, section 17)**
- A 9-week game starts every 5 trading days. It buys at the next open and holds 45 trading days.
- Commission is 0.25% each way (minimum USD 25). The −25% per-lot stop applies, and cash earns 0%.
- Results are compared on **identical games**. Differences are tested with a paired moving-block bootstrap (blocks of 9 starts).

**Variants**

| Variant | What it isolates |
| --- | --- |
| FFQ | The full strategy |
| FFQ picks, equal weight | Value of the covariance optimiser and capital allocation line |
| Risk engine only | FFQ without any fundamentals: value of the gate and the quality/value score |
| Legacy rulebook | The current strategy, rebuilt on the same data |
| Legacy + fundamental gate | What vetoing bad accounts does to the old screen (the "option C" question) |
| Legacy on the liquid universe | The universe change alone |
| US only, SEC accounts (with `--sec`) | The gate over 2015–2026, through 2018, 2020 and 2022 |

**Robustness (`--sensitivity`).** Each of 12 variants changes one rule: drop gate G2, G3, G4 or G6;
momentum-only or equal-thirds score; correlation limit 0.55 or 0.75; no volatility target; no stop;
market risk premium 4.5% or 6.5%. Each is re-run on the same games and tested against the base with the
bootstrap. The table shows which rules carry weight. It must not be used to pick a new "best" rule set.

**Three honest limits**
1. **Yahoo serves only four years of annual accounts**, so the full two-country strategy can be tested only from about 2024–2025. That is roughly 50–100 overlapping games but only 6–12 independent ones.
   - The `--sec` run adds a US-only test from 2015 using every 10-K and 10-Q as first filed. This is the only free, point-in-time test of the gate across market regimes.
   - Nothing free gives point-in-time Singapore accounts before 2023.
2. **Survivorship.** The universe is today's listed companies, so absolute returns are overstated for every variant. Compare variants with each other.
3. **Pre-registration.** The rules above were fixed before the backtest was run. If they are changed after seeing the results, say so, because the results then become in-sample. That was the lesson of errors E6, E10 and E11.

**How to read the result against the priorities**
- **Priority 1:** the "what the strategies owned" table: share of holdings failing the gate, with falling profits, or earning ROE below their cost of equity.
- **Priority 2:** worst game, 5th percentile, CVaR, maximum drawdown, beta, and realised versus predicted volatility.
- **Priority 3:** Sharpe, Sortino, and the bootstrap confidence interval of the Sharpe difference.
- **Priority 4:** mean and median 9-week return.

---

## 9. Using this in the report

| Report section (brief) | Material |
| --- | --- |
| Objective | The four priorities, in order, and why (brief: no marks for luck; justification is worth 20 of 25 marks) |
| Investment strategy | Sections 2–3: SML, FF3, APT, Markowitz, CAL; the gate; the score; the optimiser |
| Fundamental factors of industries | Screen: sector of each holding, the APT macro exposures (oil, rates, USD, gold) and the sector caps |
| Fundamental factors of stocks | The screen's section 2: ROE vs SML cost of equity, growth, F-score, leverage, P/E, and the one-paragraph case per stock |
| Technical factors of each buy and sell | Residual momentum and FF3 loadings at entry; the −25% stop; gate-failure sales with the failing test named |

---

## 10. Change log

| Date | Change | Seen backtest results first? |
| --- | --- | --- |
| 24 Sep 2026 | Rules written and fixed (this document, `ffq/config.py`) | No |

---

## References

Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). Value and momentum everywhere. *The Journal of Finance, 68*(3), 929–985.

Blitz, D., Huij, J., & Martens, M. (2011). Residual momentum. *Journal of Empirical Finance, 18*(3), 506–521.

Blume, M. E. (1971). On the assessment of risk. *The Journal of Finance, 26*(1), 1–10.

Bodie, Z., Kane, A., & Marcus, A. J. (2021). *Investments* (12th ed.). McGraw-Hill.

Chen, N.-F., Roll, R., & Ross, S. A. (1986). Economic forces and the stock market. *The Journal of Business, 59*(3), 383–403.

Daniel, K., & Moskowitz, T. J. (2016). Momentum crashes. *Journal of Financial Economics, 122*(2), 221–247.

DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal versus naive diversification: How inefficient is the 1/N portfolio strategy? *The Review of Financial Studies, 22*(5), 1915–1953.

Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics, 33*(1), 3–56.

Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics, 116*(1), 1–22.

Frazzini, A., & Pedersen, L. H. (2014). Betting against beta. *Journal of Financial Economics, 111*(1), 1–25.

Grinold, R. C., & Kahn, R. N. (2000). *Active portfolio management* (2nd ed.). McGraw-Hill.

Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers: Implications for stock market efficiency. *The Journal of Finance, 48*(1), 65–91.

Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis, 88*(2), 365–411.

Markowitz, H. (1952). Portfolio selection. *The Journal of Finance, 7*(1), 77–91.

Moreira, A., & Muir, T. (2017). Volatility-managed portfolios. *The Journal of Finance, 72*(4), 1611–1644.

Novy-Marx, R. (2013). The other side of value: The gross profitability premium. *Journal of Financial Economics, 108*(1), 1–28.

Piotroski, J. D. (2000). Value investing: The use of historical financial statement information to separate winners from losers. *Journal of Accounting Research, 38*(Supplement), 1–41.

Ross, S. A. (1976). The arbitrage theory of capital asset pricing. *Journal of Economic Theory, 13*(3), 341–360.

Sharpe, W. F. (1964). Capital asset prices: A theory of market equilibrium under conditions of risk. *The Journal of Finance, 19*(3), 425–442.

Sloan, R. G. (1996). Do stock prices fully reflect information in accruals and cash flows about future earnings? *The Accounting Review, 71*(3), 289–315.

Treynor, J. L., & Black, F. (1973). How to use security analysis to improve portfolio selection. *The Journal of Business, 46*(1), 66–86.
