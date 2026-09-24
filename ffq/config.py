"""
Every parameter of the FFQ strategy lives here.

All values were fixed BEFORE any backtest was run (24 Sep 2026) from textbook
conventions or from the group's earlier tested rulebook. They are not tuned.
If you change one, record it in STRATEGY.md section 9 (change log).
"""

# --------------------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------------------
# Singapore: the 46 names of the original rulebook (tested and broadened in the
# complete record, section 6). Yahoo tickers end in ".SI".
SG = ["D05.SI", "O39.SI", "U11.SI", "Z74.SI", "S68.SI", "C38U.SI", "A17U.SI", "N2IU.SI",
      "AJBU.SI", "BN4.SI", "U96.SI", "S63.SI", "C6L.SI", "F34.SI", "Y92.SI", "OV8.SI",
      "C07.SI", "C09.SI", "U14.SI", "H78.SI", "G13.SI", "BS6.SI", "H02.SI", "V03.SI",
      "C52.SI", "S58.SI", "CC3.SI", "ME8U.SI", "M44U.SI", "K71U.SI", "T82U.SI", "J69U.SI",
      "BUOU.SI", "C2PU.SI", "HMN.SI", "TQ5.SI", "F17.SI", "H13.SI", "558.SI", "F99.SI",
      "A7RU.SI", "NS8U.SI", "S08.SI", "E5H.SI", "S59.SI", "VC2.SI"]

# The 28 hand-picked US names of the original rulebook (kept only so the legacy
# strategy can be reproduced; round-2 record section 5 found them hindsight-tilted).
US_LEGACY = ["AAPL", "MSFT", "NVDA", "AMD", "MU", "AVGO", "JPM", "V", "WMT", "COST", "UNH",
             "HD", "MCD", "CAT", "GE", "LLY", "XOM", "KO", "PG", "NFLX", "AMZN", "GOOGL",
             "META", "CRM", "ORCL", "DIS", "BA", "QCOM"]

GOLD = ["NEM", "B"]  # Barrick is "B" on Yahoo; "GOLD" is a different company.

# US pool for the new strategy: S&P 100 members (approximate 2026 list) plus the
# legacy 28, the liquid semiconductor names the round-2 record found missing, and
# the two gold miners. At every date the strategy keeps only the US_TOP_N most
# liquid of these (252-day median dollar volume), a rule rather than a hand-pick.
US_POOL = sorted(set(US_LEGACY + GOLD + [
    "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMGN", "AMT", "AXP", "BAC", "BK", "BKNG", "BLK",
    "BMY", "BRK-B", "C", "CHTR", "CL", "CMCSA", "COF", "COP", "CSCO", "CVS", "CVX", "DE",
    "DHR", "DUK", "EMR", "F", "FDX", "GD", "GILD", "GM", "GS", "HON", "IBM", "INTC", "INTU",
    "ISRG", "JNJ", "LIN", "LMT", "LOW", "MA", "MDLZ", "MDT", "MET", "MMM", "MO", "MRK", "MS",
    "NEE", "NKE", "NOW", "PEP", "PFE", "PLTR", "PM", "PYPL", "RTX", "SBUX", "SCHW", "SO",
    "SPG", "T", "TGT", "TMO", "TMUS", "TSLA", "TXN", "UBER", "UNP", "UPS", "USB", "VZ", "WFC",
    "AMAT", "LRCX", "KLAC", "MRVL", "ADI", "WDC", "STX",
]))
US_TOP_N = 40

# Sector map (GICS-style, static). Used for sector caps and for choosing which
# accounting tests apply. "semi" and "gold" are sub-flags used for name caps.
SECTOR = {
    # Singapore
    "D05.SI": "Financials", "O39.SI": "Financials", "U11.SI": "Financials",
    "Z74.SI": "Communication", "S68.SI": "Financials", "C38U.SI": "Real Estate",
    "A17U.SI": "Real Estate", "N2IU.SI": "Real Estate", "AJBU.SI": "Real Estate",
    "BN4.SI": "Industrials", "U96.SI": "Utilities", "S63.SI": "Industrials",
    "C6L.SI": "Industrials", "F34.SI": "Staples", "Y92.SI": "Staples", "OV8.SI": "Staples",
    "C07.SI": "Discretionary", "C09.SI": "Real Estate", "U14.SI": "Real Estate",
    "H78.SI": "Real Estate", "G13.SI": "Discretionary", "BS6.SI": "Industrials",
    "H02.SI": "Health Care", "V03.SI": "Technology", "C52.SI": "Industrials",
    "S58.SI": "Industrials", "CC3.SI": "Communication", "ME8U.SI": "Real Estate",
    "M44U.SI": "Real Estate", "K71U.SI": "Real Estate", "T82U.SI": "Real Estate",
    "J69U.SI": "Real Estate", "BUOU.SI": "Real Estate", "C2PU.SI": "Real Estate",
    "HMN.SI": "Real Estate", "TQ5.SI": "Real Estate", "F17.SI": "Real Estate",
    "H13.SI": "Real Estate", "558.SI": "Technology", "F99.SI": "Staples",
    "A7RU.SI": "Utilities", "NS8U.SI": "Industrials", "S08.SI": "Industrials",
    "E5H.SI": "Staples", "S59.SI": "Utilities", "VC2.SI": "Staples",
    # United States
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "AMD": "Technology",
    "MU": "Technology", "AVGO": "Technology", "QCOM": "Technology", "INTC": "Technology",
    "TXN": "Technology", "AMAT": "Technology", "LRCX": "Technology", "KLAC": "Technology",
    "MRVL": "Technology", "ADI": "Technology", "WDC": "Technology", "STX": "Technology",
    "CRM": "Technology", "ORCL": "Technology", "ACN": "Technology", "ADBE": "Technology",
    "CSCO": "Technology", "IBM": "Technology", "INTU": "Technology", "NOW": "Technology",
    "PLTR": "Technology",
    "JPM": "Financials", "V": "Financials", "MA": "Financials", "AIG": "Financials",
    "AXP": "Financials", "BAC": "Financials", "BK": "Financials", "BLK": "Financials",
    "BRK-B": "Financials", "C": "Financials", "COF": "Financials", "GS": "Financials",
    "MET": "Financials", "MS": "Financials", "PYPL": "Financials", "SCHW": "Financials",
    "USB": "Financials", "WFC": "Financials",
    "WMT": "Staples", "COST": "Staples", "KO": "Staples", "PG": "Staples", "CL": "Staples",
    "MDLZ": "Staples", "MO": "Staples", "PEP": "Staples", "PM": "Staples", "TGT": "Staples",
    "UNH": "Health Care", "LLY": "Health Care", "ABBV": "Health Care", "ABT": "Health Care",
    "AMGN": "Health Care", "BMY": "Health Care", "CVS": "Health Care", "DHR": "Health Care",
    "GILD": "Health Care", "ISRG": "Health Care", "JNJ": "Health Care", "MDT": "Health Care",
    "MRK": "Health Care", "PFE": "Health Care", "TMO": "Health Care",
    "HD": "Discretionary", "MCD": "Discretionary", "AMZN": "Discretionary",
    "BKNG": "Discretionary", "F": "Discretionary", "GM": "Discretionary",
    "LOW": "Discretionary", "NKE": "Discretionary", "SBUX": "Discretionary",
    "TSLA": "Discretionary",
    "CAT": "Industrials", "GE": "Industrials", "BA": "Industrials", "DE": "Industrials",
    "EMR": "Industrials", "FDX": "Industrials", "GD": "Industrials", "HON": "Industrials",
    "LMT": "Industrials", "MMM": "Industrials", "RTX": "Industrials", "UBER": "Industrials",
    "UNP": "Industrials", "UPS": "Industrials",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "NFLX": "Communication", "GOOGL": "Communication", "META": "Communication",
    "DIS": "Communication", "CHTR": "Communication", "CMCSA": "Communication",
    "T": "Communication", "TMUS": "Communication", "VZ": "Communication",
    "LIN": "Materials", "NEM": "Materials", "B": "Materials",
    "DUK": "Utilities", "NEE": "Utilities", "SO": "Utilities",
    "AMT": "Real Estate", "SPG": "Real Estate",
}

# Semiconductors and memory/storage (the "memory complex" correlations in the
# round-2 record are 0.65-0.87). At most SEMI_MAX_NAMES of these in the book.
SEMI = {"NVDA", "AMD", "MU", "AVGO", "QCOM", "INTC", "TXN", "AMAT", "LRCX", "KLAC", "MRVL",
        "ADI", "WDC", "STX", "558.SI"}

# Deposit-taking banks, brokers and insurers: debt/equity is meaningless for them,
# so the leverage test uses equity/assets and the F-score drops the two signals
# that need current assets and gross margin.
BANK_LIKE = {"D05.SI", "O39.SI", "U11.SI", "JPM", "BAC", "C", "WFC", "USB", "COF", "BK",
             "GS", "MS", "SCHW", "AIG", "MET", "BRK-B", "AXP"}

# REITs and business trusts: leverage test is gearing (debt / total assets).
REIT_LIKE = {"C38U.SI", "A17U.SI", "N2IU.SI", "AJBU.SI", "ME8U.SI", "M44U.SI", "K71U.SI",
             "T82U.SI", "J69U.SI", "BUOU.SI", "C2PU.SI", "HMN.SI", "A7RU.SI", "NS8U.SI",
             "AMT", "SPG"}

# SGX names whose Yahoo price is quoted in USD, not SGD. The legacy screener
# multiplied these by SGDUSD as well, a small error that this version fixes.
SG_USD_QUOTED = {"H78.SI", "NS8U.SI"}

# Reporting currency of the accounts when it differs from the quote currency.
# Used only if Yahoo's `financialCurrency` field is unavailable.
FIN_CCY_FALLBACK = {"BS6.SI": "CNY", "Y92.SI": "THB", "H78.SI": "USD", "NS8U.SI": "HKD",
                    "F34.SI": "USD", "E5H.SI": "USD", "S59.SI": "CNY", "C07.SI": "USD"}

# --------------------------------------------------------------------------------------
# Market data series (all from Yahoo Finance)
# --------------------------------------------------------------------------------------
FX_TICKERS = {"SGD": "SGDUSD=X", "CNY": "CNYUSD=X", "THB": "THBUSD=X", "HKD": "HKDUSD=X",
              "EUR": "EURUSD=X", "GBP": "GBPUSD=X", "JPY": "JPYUSD=X", "MYR": "MYRUSD=X",
              "IDR": "IDRUSD=X", "AUD": "AUDUSD=X"}
RF_TICKER = "^IRX"  # 13-week US Treasury bill yield, percent a year

# Tradable proxies for the factor models. Each factor is a return or a
# long-short return difference of two ETFs, so the whole model is Yahoo data.
FACTOR_ETFS = ["SPY", "IWM", "IWB", "IWD", "IWF", "EWS", "EFA", "EFV", "EFG", "SCZ",
               "QUAL", "MTUM", "IEF", "GLD", "USO", "UUP"]
INDEX_TICKERS = ["^GSPC", "^STI"]

# Fama-French three-factor model (FF3), by listing country:
#   US stocks:  MKT = SPY - rf,  SMB = IWM - IWB,  HML = IWD - IWF
#   SG stocks:  MKT = EWS - rf,  SMB = SCZ - EFA,  HML = EFV - EFG  (international proxies)
# APT macro-factor risk model (used for the covariance matrix):
APT_FACTORS = ["MKT_US", "MKT_SG", "SMB_US", "HML_US", "RATES", "OIL", "GOLD_F", "USD"]

# --------------------------------------------------------------------------------------
# SML / CAPM
# --------------------------------------------------------------------------------------
MRP = 0.055                 # market risk premium, a year (textbook range 5-6%)
BLUME = (0.67, 0.33)        # adjusted beta = 0.67 * raw beta + 0.33 (Blume 1971; Bloomberg)
BETA_WEEKS = 104            # two years of weekly returns for CAPM beta
BETA_MIN_WEEKS = 78

# --------------------------------------------------------------------------------------
# Fundamental gate (all must pass). Statements are used only after they are public.
# --------------------------------------------------------------------------------------
FUND_LAG_DAYS = 75          # annual accounts usable 75 days after fiscal year end
QTR_LAG_DAYS = 45           # quarterly / half-year accounts usable 45 days after period end
NI_GROWTH_MIN = -0.10       # profit may not have fallen more than 10% in the last year
FSCORE_MIN = 5              # Piotroski F-score (scaled to 9) must be at least this
FSCORE_MIN_SIGNALS = 6      # need at least 6 of the 9 signals to be computable
DE_MAX = 2.0                # non-financials: total debt / equity
REIT_GEARING_MAX = 0.50     # REITs and trusts: total debt / total assets (MAS limit 50%)
BANK_EQ_ASSETS_MIN = 0.05   # banks and insurers: equity / assets
PE_MAX = 60.0               # valuation sanity: trailing P/E must be positive and at most 60

# --------------------------------------------------------------------------------------
# Ranking score (computed only among stocks that pass the gate)
# --------------------------------------------------------------------------------------
W_MOM, W_QUAL, W_VALUE = 0.50, 0.25, 0.25
FF_WEEKS = 156              # FF3 regression window (3 years of weekly returns)
FF_MIN_WEEKS = 104
SKIP_DAYS = 21              # formation window ends 21 trading days ago (skip last month)
FORM_DAYS = 252             # and starts 252 trading days ago (12-1, as in the rulebook)
IC = 0.05                   # information coefficient for alpha = IC * resid vol * score
SCORE_MIN = 0.0             # buy only stocks at or above the average score of those passing the gate

# --------------------------------------------------------------------------------------
# Portfolio construction and risk management
# --------------------------------------------------------------------------------------
CANDIDATES_PER_COUNTRY = 12
N_MAX, N_MIN = 12, 8
CORR_MAX_PAIR = 0.65        # no two holdings may correlate above this (max, not average)
W_MIN, W_MAX = 0.04, 0.15   # weight of each holding in the risky book
SECTOR_W_MAX, SECTOR_N_MAX = 0.30, 3
SEMI_W_MAX, SEMI_N_MAX = 0.20, 2
GOLD_N_MAX = 1
COUNTRY_W_MIN = 0.30        # at least 30% in each of Singapore and the US
BETA_SPY_MAX = 1.00         # book beta to the S&P 500 no higher than the index
RC_MAX = 0.20               # no holding may contribute more than 20% of portfolio variance
SIGMA_TARGET = 0.15         # capital allocation line: invest y = min(1, 15% / book vol)
EXPOSURE_FLOOR = 0.50
COV_WEEKS = 156
COV_MIN_WEEKS = 104
COV_BLEND = 0.5             # covariance = 0.5 * APT factor model + 0.5 * Ledoit-Wolf sample
EWMA_HALFLIFE_WEEKS = 52

# --------------------------------------------------------------------------------------
# Trading costs, stops, backtest
# --------------------------------------------------------------------------------------
COMMISSION = 0.0025
COMMISSION_MIN_USD = 25.0
STOP = 0.75                 # sell a lot that closes at or below 75% of its purchase price
HOLD_DAYS = 45              # one 9-week game
STEP_DAYS = 5               # a new game starts every 5 trading days
ACCOUNT_USD = 1_000_000
CASH_FLOOR_USD = 15_000
