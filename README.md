
# Regime-Adaptive Long/Short Equity ETF  
**Systematic ETF Strategy using HMMs and Technical Analysis**  
*MSDS Term Project – Summer 2025*  
**Author:** Genki Hirayama • **Email:** genkihirayama2024@u.northwestern.edu

---

## 1) Project Overview

This project implements a **rules-based, regime-adaptive long/short equity strategy** designed as an actively managed, systematic ETF. The strategy:

- Detects **market regimes** (Bull / Neutral / Bear) with a **Hidden Markov Model (HMM)** and a **200-day SMA override** (upgrade/degrade safety) to reduce regime lag.
- Builds **cross-sectional long/short portfolios** over a **fixed 50-stock S&P 500 subset**, blending **momentum (12-1 + 6m)** and **mean-reversion (RSI + Bollinger Z)** signals.
- Applies **tilt-aware beta targeting** (preserves L1 exposure), **weekly rebalancing** with **weight smoothing**, **EWMA volatility targeting**, and realistic **ETF-like costs**:
  - Trading costs = **turnover × bps**
  - Optional **management fee** (bps/year → daily)
  - **Short borrow costs** on short notional
- Benchmarks vs **SPY (net of expense ratio)** and saves plots, weights, logs, and summary metrics.

**Motivation.** Passive ETFs can struggle in volatile, regime-shifting markets. An adaptive approach aims to improve risk-adjusted returns and drawdown control.

---

## 2) Universe

A fixed list of 50 liquid S&P 500 names with long histories and no special-character tickers (e.g., no BRK.B). Both Alphabet share classes are included.

```
AAPL, MSFT, AMZN, GOOGL, GOOG, META, NVDA, TSLA, JPM, BAC,
WFC, C, GS, MS, V, MA, HD, LOW, COST, WMT,
TGT, UNH, PFE, MRK, JNJ, ABT, TMO, XOM, CVX, COP,
SLB, NKE, SBUX, MCD, PEP, KO, DIS, NFLX, INTU, ADBE,
ORCL, CRM, AMD, INTC, AVGO, CSCO, LLY, ABBV, AMGN, BMY
```

Function in code: `default_universe_50()`.

---

## 3) Repo Structure (suggested)

```
.
├─ README.md
├─ requirements.txt
├─ src/
│  ├─ regime_adaptive_v4_4.py           # main script (HMM + SMA override + macro safety valve)
│  ├─ regime_adaptive_50_sp500.py       # earlier CLI version (optional)
│  └─ notebook_style_backtest.py        # step-by-step version (optional)
├─ artifacts/                           # outputs (set OUT_DIR="artifacts" in scripts)
│  ├─ results_tradelog.csv
│  ├─ results_weights.csv
│  ├─ results_summary.csv
│  ├─ results_equity.png
│  ├─ results_drawdowns.png
│  └─ results_equity_vs_spy.png
└─ reports/
   ├─ Research_Report_2025-08-03.docx
   └─ Research_Report_2025-08-17.docx   # latest update with figures & metrics
```

> If your script currently writes to an absolute path, set `OUT_DIR = "artifacts"` so outputs land inside the repo.

---

## 4) Installation

Python ≥ 3.10 recommended.

```bash
# create & activate a virtual env (macOS/Linux)
python -m venv .venv
source .venv/bin/activate

# Windows PowerShell
# python -m venv .venv
# .\.venv\Scripts\Activate.ps1

# install dependencies
pip install -r requirements.txt
```

**requirements.txt (suggested)**

```
numpy
pandas
yfinance
matplotlib
scikit-learn
hmmlearn
ta
python-docx   # only needed if you update the DOCX report via script
```

---

## 5) How to Run

### Option A — Main script (recommended)

Edit user settings at the top of `src/regime_adaptive_v4_4.py`:

- Dates: `START_DATE`, `END_DATE`
- Outputs: `OUT_DIR = "artifacts"`
- Fees: `TRADE_BPS`, `MGMT_ER_BPS`, `BORROW_BPS`, `BENCH_ER_BPS`
- Risk: `TARGET_VOL`, `VOL_WIN`
- Regime knobs: `REBAL_FREQ`, `SMOOTH`, `NET_TILT_BASE`, `TARGET_BETA_BASE`, selection quantiles, SMA upgrade/degrade thresholds, macro safety valve

Run:

```bash
python src/regime_adaptive_v4_4.py
```

### Option B — Notebook-style (step-by-step)

Use `src/notebook_style_backtest.py` and run cell-by-cell (e.g., VS Code / Jupyter). Ensure `OUT_DIR = "artifacts"`.

---

## 6) Outputs

After a run you should see (in `artifacts/`):

- **`results_tradelog.csv`** — daily regime, gross return, fees, net return, equity; benchmark net return/equity  
- **`results_weights.csv`** — last business day of each month cross-section weights (post-scale)  
- **`results_summary.csv`** — summary metrics for Strategy (net) and SPY (net ER)  
- **`results_equity.png`** — strategy equity (net)  
- **`results_drawdowns.png`** — rolling drawdowns  
- **`results_equity_vs_spy.png`** — strategy vs SPY equity (net of ER)

Add latest figures/metrics to your report at `reports/Research_Report_2025-08-17.docx`.

---

## 7) Methodology (brief)

- **Regimes:** Walk-forward **Gaussian HMM** with 3 states (features: 1-day return, 5-day return, 20-day vol).  
  **SMA-200 override** upgrades/degrades one level when SPY is sufficiently above/below trend to reduce regime mislabeling.
- **Signals:**
  - **Momentum:** 12-1 and 6-month (lagged by one month to avoid look-ahead), cross-sectional rank.
  - **Mean-reversion:** RSI-based oversold + Bollinger Z composite, cross-sectional rank.
- **Selection by regime:**
  - **Bull:** long momentum winners; shorts disabled by default.
  - **Neutral:** blended long sleeve; shorts on overbought/weak names.
  - **Bear:** short momentum losers (trend-following) with small contrarian long sleeve.
- **Positioning:** Weekly rebalance on W-WED → forward-filled → smoothed (hysteresis).  
  **Tilt-aware beta targeting** preserves L1 exposure and aims at regime-specific beta (e.g., +0.5 in bull).  
  **EWMA volatility targeting** scales to `TARGET_VOL`.
- **Costs:** Trading (turnover×bps), optional management fee (bps/yr), and borrow on short notional.  
  Benchmark SPY net of its ER.

---

## 8) Results Snapshot

> Replace links with your most recent run:

- **Summary:** [`artifacts/results_summary.csv`](artifacts/results_summary.csv)  
- **Equity vs SPY:** ![Equity vs SPY](artifacts/results_equity_vs_spy.png)  
- **Drawdowns:** ![Drawdowns](artifacts/results_drawdowns.png)

**Notes:**  
If performance sags in strong recoveries, tune the **SMA upgrade threshold** and **macro safety valve** (raises net/beta floors and reduces/turns off shorts on bullish macro days).  
While debugging, set `MGMT_ER_BPS = 0.0`; then restore realistic fees.

---

## 9) Reproducibility & Testing

- **Randomness**: HMM uses `SEED = 42`.  
- **Sanity checks:**
  1. Run with `MGMT_ER_BPS = 0.0` and confirm equity isn’t drifting down purely from fees.  
  2. Confirm non-zero weights after resample/smoothing (mean |weights| (pre-scale) > 0).  
  3. Review diagnostics printed by the script:
     - Avg daily turnover (L1), mean post-scale gross exposure  
     - Mean realized beta overall and on **bullish macro days** (should be ≥ 0 when macro is bullish)
- **Smoke test (optional):** verify CSVs exist and `equity` is well-formed; check that `results_weights.csv` has monthly snapshots.

---

## 10) Deliverables Checklist

- Public GitHub repository (share **cloneable URL** ending with `.git`)  
- Research report (`reports/Research_Report_2025-08-17.docx` and/or PDF) with latest figures/metrics  
- Source code under `src/` and generated outputs under `artifacts/`  
- This **README.md** with usage instructions and explanation  
- Data from `yfinance` (no proprietary data included)

---

## 11) Troubleshooting

- **No plots/CSVs generated:** Ensure `OUT_DIR = "artifacts"` and the script creates/uses that folder.  
- **Too bearish in recoveries (e.g., 2023):** Increase SMA **UPGRADE** sensitivity; raise macro safety-valve floors (`MACRO_BULL_BETAFLOOR`, `MACRO_BULL_LONGFLOOR`); optionally disable shorts when `dist_to_sma > MACRO_BULL_DIST_HARD`.  
- **Equity slopes down slowly:** Check that NaNs are filled before smoothing; start with `MGMT_ER_BPS = 0` to isolate costs.

---

## 12) Use of AI Assistance

AI assistance (ChatGPT/GPT) was used to **troubleshoot Python script errors**, improve robustness (NaN handling, weekly resample & smoothing), design regime logic refinements (SMA upgrade/degrade, macro safety valve), and draft this README. All modeling choices and final code were reviewed and validated by the author.

---

### Quick Commands

```bash
# run the backtest (outputs to ./artifacts)
python src/regime_adaptive_v4_4.py

