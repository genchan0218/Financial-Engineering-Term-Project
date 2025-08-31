# Regime-Adaptive S&P 500 Strategy

This project implements a **regime-adaptive trading strategy** across a broad S&P 500 equity universe (as many stocks as could be pulled with sufficient history). It combines **Hidden Markov Models (HMMs)**, SMA-based regime overrides, momentum, mean reversion, and volatility targeting. The project includes **backtesting, portfolio construction, risk management, and Monte Carlo simulations**.

---

## 📂 Project Layout

```
Financial-Engineering-Term-Project/
├─ regime_adaptive_50_sp500_notebook_v5.py   # Main Python script
├─ Prospectus_Regime_Adaptive_Strategy.docx  # Written prospectus
├─ PP.pptx                                   # Presentation slides
├─ results_equity.png
├─ results_drawdowns.png
├─ results_equity_vs_spy.png
├─ results_summary.xlsx
├─ results_tradelog.xlsx
├─ results_weights.xlsx
├─ results_mc_cagr_hist.png
├─ results_mc_sharpe_hist.png
├─ results_mc_maxdd_hist.png
├─ results_mc_forward_fan.png
├─ results_mc_forward_percentiles.xlsx
├─ results_mc_metric_sims.xlsx
├─ results_mc_sample_paths.xlsx
├─ README.md
├─ requirements.txt
└─ data_cache/
   ├─ px.csv   # Cached equity universe (majority of S&P 500 tickers with 10+ years of history)
   └─ spy.csv  # Cached SPY index benchmark
```

**Notes:**
- `px.csv` and `spy.csv` in `data_cache/` ensure reproducibility by avoiding repeated API calls.  
- The equity universe in `px.csv` includes most S&P 500 names with sufficient historical coverage.  
- All **results** files are produced automatically when running backtests or Monte Carlo simulations.  
- The **prospectus** and **presentation** document the methodology, results, and interpretation.  

---

## ⚙️ Running Instructions

1. **Setup environment**

```bash
python -m venv .venv
.\.venv\Scriptsctivate
pip install -r requirements.txt
```

2. **Run the backtest**

From the project root:

```bash
python regime_adaptive_50_sp500_notebook_v5.py
```

This will:
- Load cached data from `data_cache/px.csv` (stock prices) and `data_cache/spy.csv` (SPY benchmark).  
- Generate portfolio weights, trade logs, summary stats, and equity/drawdown charts.  

Outputs include:
- `results_tradelog.xlsx`
- `results_weights.xlsx`
- `results_summary.xlsx`
- Equity and drawdown plots (`results_equity.png`, `results_drawdowns.png`, `results_equity_vs_spy.png`)

3. **Run Monte Carlo simulation**

Monte Carlo analysis is integrated into the script. Running the notebook will also generate:
- `results_mc_forward_fan.png`
- Histograms (`results_mc_cagr_hist.png`, `results_mc_sharpe_hist.png`, `results_mc_maxdd_hist.png`)
- Forward-simulation Excel files (`results_mc_forward_percentiles.xlsx`, `results_mc_metric_sims.xlsx`, `results_mc_sample_paths.xlsx`)

---

## 📊 Outputs

- **Backtest Results**
  - `results_tradelog.xlsx`: Daily portfolio returns, costs, equity curve
  - `results_weights.xlsx`: Monthly portfolio weights
  - `results_summary.xlsx`: Key metrics (CAGR, Volatility, Sharpe, Max Drawdown, Total Return)
  - Plots: equity vs SPY, rolling drawdowns, standalone equity  

- **Monte Carlo Results**
  - `results_mc_forward_fan.png`: Forward equity distribution fan chart
  - `results_mc_cagr_hist.png`, `results_mc_sharpe_hist.png`, `results_mc_maxdd_hist.png`: Distribution histograms
  - `results_mc_forward_percentiles.xlsx`, `results_mc_metric_sims.xlsx`, `results_mc_sample_paths.xlsx`: Quantitative outputs  

---

## 🤖 LLM Use Report

For this project, I used an AI assistant to:
- Debug Python code (e.g., relative imports, regime logic, file handling)  
- Improve documentation (README structure, running instructions)   
- Draft the written prospectus and presentation narratives.  

Final choices of signals, parameter settings, and conclusions on **strategy underperformance vs. SPY buy-and-hold** were made independently, with AI used mainly for **debugging, documentation, and communication clarity**.  
