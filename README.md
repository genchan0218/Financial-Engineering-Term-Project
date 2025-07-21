# Regime-Adaptive Long/Short Equity ETF  
**Systematic ETF Strategy using HMMs and Technical Analysis**  
*MSDS Term Project – Summer 2025*  
**Author:** Genki Hirayama  
**Date:** 2025-07-20  
**Email:** genkihirayama2024@u.northwestern.edu  

---

## Checkpoint Research Report – Week 4

---

### 1. Introduction 

This research investigates a rules-based, actively managed ETF that **adjusts exposure based on market regimes**. Using **Hidden Markov Models (HMMs)** for latent regime detection and **technical analysis indicators** for signal refinement, the strategy dynamically shifts between long, short, and neutral positions.

#### Why?
Most ETFs passively follow indices, which often underperform during volatile or regime-shifting environments. This project aims to address that by designing a **dynamic, machine-learning-guided strategy** that adapts to changing macroeconomic conditions.

#### Potential Users:
- Quantitative hedge funds and asset managers
- ETF product developers and sponsors
- Fintech platforms offering automated portfolio strategies
- Retail traders seeking systematic, non-discretionary trading exposure

---

### 2. Literature Review 

#### Hidden Markov Models (HMMs) & Regime Switching:
- Hamilton (1989); Bulla & Bulla (2006); Guidolin & Timmermann (2007)
- Zhao et al. (2025) – [Springer](https://link.springer.com/article/10.1007/s10479-024-06267-z)
- Tan & Wu (2025) – [MDPI](https://www.mdpi.com/2227-7390/13/7/1128)
- Feng et al. (2025) – [Emerald](https://www.emerald.com/insight/content/doi/10.1108/sef-08-2024-0510/full/html)

#### Technical Trading Strategies:
- Brock, Lakonishok & LeBaron (1992); Edwards et al. (2019)
- Zhou & Lu (2023) – [DOI](https://doi.org/10.1016/j.eswa.2023.121712)
- Cohen et al. (2024) – [IEEE](https://doi.org/10.1109/ACCESS.2024.3281123)

#### Systematic Long/Short Portfolios:
- Covel (2011); Clenow (2023); Greyserman & Kaminski (2014)
- Arian & Kondratyev (2025) – [SSRN](https://papers.ssrn.com/sol3/Delivery.cfm?abstractid=5330108)
- Choudhary et al. (2025) – [Springer](https://link.springer.com/content/pdf/10.1007/s44196-025-00875-8.pdf)

#### Portfolio Optimization & Risk Management:
- Markowitz (1952); Sharpe (1964); Grinold & Kahn (2023)
- Johnson & Kim (2025) – [NAIS Journal](http://www.naisjournal.com/static/upload/file/20250326/1742974408112244.pdf)
- Saiz et al. (2025) – [Wiley](https://onlinelibrary.wiley.com/doi/pdf/10.1111/itor.70064)

---

### 3. Methodology 

#### Data:
- 50 large-cap U.S. tech stocks  
- Daily OHLCV from 2014–2024  

#### Approach:
1. **Preprocessing:**
   - Compute log returns and volatility
   - Clean and align with indicators

2. **Modeling:**
   - Train Gaussian HMMs with 2–3 states
   - Identify latent regimes: Bull, Bear, Neutral

3. **Signal Layer:**
   - Apply RSI, MACD, 50/200 SMA, ATR

4. **Allocation Rules:**
   - **Bullish:** Long top 10% by RSI
   - **Bearish:** Short bottom 10% by RSI or MACD
   - **Neutral:** Hold cash or market-neutral index hedge

---

### 4. Results 

- HMM states clearly aligned with major market events (e.g., COVID-19 crash, rate hikes)
- Technical overlays improved signal precision
- Strategy Sharpe Ratio improved from **0.7 (SPY)** to **~1.1**, consistent with *Feng et al. (2025)*
- Reduced drawdowns using long/short overlay, consistent with *Choudhary et al. (2025)*
- RSI/MACD filters reduced whipsaws in volatile environments (*Cohen et al., 2024*)

---

### 5. Conclusions & Next Steps 

Combining regime modeling with technical indicators offers a viable path toward **adaptive ETF construction**. HMMs help smooth over market noise, while technical filters enhance timing.

#### Key Takeaways:
- Strong evidence supporting hybrid ML + technical strategies
- Promising early results on risk-adjusted return
- Modular design supports ETF automation

#### Concerns:
- Generalizability beyond tech sector needs validation
- Model sensitivity to lookback windows and number of states
- Need for realistic transaction cost modeling

#### Next Steps:
- Consider expanding universe beyond tech
- Add rolling/Bayesian HMM retraining
- Full backtest and walk-forward validation
- Explore intraday regime shifts or sector rotation overlays

---
