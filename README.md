# Regime-Adaptive 50-Stock S&P 500 Strategy (v4.7 + Monte Carlo)

This project implements a **regime-adaptive equity strategy** combining Hidden Markov Models (HMMs), SMA-based upgrades/downgrades, residual momentum, mean reversion, and a macro safety valve. It includes **backtesting, portfolio construction, risk management, and Monte Carlo simulations**.

---

## 📂 Project Layout

```
Financial-Engineering-Term-Project/
├─ results_drawdowns.png
├─ results_equity.png
├─ results_equity_vs_spy.png
├─ results_summary.xlsx
├─ results_tradelog.xlsx
├─ results_weights.xlsx
├─ README.md
├─ requirements.txt
├─ Prospectus_Regime_Adaptive_Strategy.docx
├─ results_mc_maxdd_hist.png
├─ results_mc_sharpe_hist.png
├─ results_mc_cagr_hist.png
├─ results_mc_forward_fan.png
├─ results_mc_forward_percentiles.xlsx
├─ results_mc_metric_sims.xlsx
├─ results_mc_sample_paths.xlsx
└─ data_cache/        # cached px.csv, spy.csv
```

**Notes:**
- All **backtest outputs** are saved under the root with the prefix `results_*`.
- Monte Carlo forward simulations and histograms are included in both **Excel (`.xlsx`)** and **PNG plots**.
- The `Prospectus_Regime_Adaptive_Strategy.docx` contains the written prospectus for submission.
- `data_cache/` holds cached market data (`px.csv`, `spy.csv`) to ensure reproducibility.

---

## ⚙️ Running Instructions

1. **Setup environment**

```bash
python -m venv .venv
.\.venv\Scriptsctivate
pip install -r requirements.txt
```

2. **Run backtest**

From the `code/` directory:

```bash
python main_backtest.py
```

This produces:
- `results_tradelog.xlsx`
- `results_weights.xlsx`
- `results_summary.xlsx`
- Equity/drawdown plots (`results_equity.png`, `results_drawdowns.png`, etc.)

3. **Run Monte Carlo simulation**

```bash
python main_mc.py
```

This produces:
- Forward fan chart (`results_mc_forward_fan.png`)
- Metric histograms (`results_mc_cagr_hist.png`, `results_mc_sharpe_hist.png`, `results_mc_maxdd_hist.png`)
- Excel outputs (`results_mc_forward_percentiles.xlsx`, `results_mc_metric_sims.xlsx`, `results_mc_sample_paths.xlsx`)

---

## 📊 Outputs

- **Backtest:**
  - `results_tradelog.xlsx`: Daily returns, costs, equity series
  - `results_weights.xlsx`: Monthly portfolio weights
  - `results_summary.xlsx`: Key performance metrics (CAGR, Vol, Sharpe, MaxDD, TotalReturn)
  - `results_equity.png`, `results_equity_vs_spy.png`, `results_drawdowns.png`: Visualizations

- **Monte Carlo:**
  - `results_mc_forward_percentiles.xlsx`: Percentile fan bands
  - `results_mc_metric_sims.xlsx`: Resampled metric distributions
  - `results_mc_sample_paths.xlsx`: Sample forward equity paths
  - `results_mc_forward_fan.png`: Forward simulation fan chart
  - `results_mc_cagr_hist.png`, `results_mc_sharpe_hist.png`, `results_mc_maxdd_hist.png`: Metric histograms

---

## 🤖 LLM Use Report

As part of this project, I used an AI assistant for:
- Debugging trading scripts
- Model tuning for performance improvement
- Writing this **README.md** for clarity and reproducibility.

Final decisions (signals used, parameters, conclusions on underperformance vs SPY) were mine, with AI used for **debugging, documentation, and structuring**.

