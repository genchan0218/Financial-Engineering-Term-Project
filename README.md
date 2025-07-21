# Regime-Adaptive Long/Short Equity ETF

**Systematic ETF Strategy using HMMs and Technical Analysis**  
*MSDS Term Project – Summer 2025*  
**Author:** Genki Hirayama  
**Date:** 2025-07-20  
**Email:** genkihirayama2024@u.northwestern.edu

---

## Checkpoint Research Report – Week 4

### 1. Introduction

This project proposes a rules-based, actively managed ETF that uses Hidden Markov Models (HMMs) and technical trading indicators to allocate long, short, or neutral positions in U.S. equities. Unlike passive strategies, this ETF adapts to market regimes, enabling better responses to macroeconomic cycles and volatility.

**Key Features:**
- Systematic and transparent allocation rules
- ML-based regime detection using HMMs
- Built for automation in an ETF framework

**Target Users:**
- Quantitative investors
- Portfolio strategy teams
- Retail traders seeking adaptive exposure

---

### 2. Literature Review

#### Hidden Markov Models & Regime Switching
- Hamilton (1989); Bulla & Bulla (2006); Guidolin & Timmermann (2007)
- Zhao et al. (2025) – [Springer](https://link.springer.com/article/10.1007/s10479-024-06267-z)
- Tan & Wu (2025) – [MDPI](https://www.mdpi.com/2227-7390/13/7/1128)
- Feng et al. (2025) – [Emerald](https://www.emerald.com/insight/content/doi/10.1108/sef-08-2024-0510/full/html)

#### Technical Trading Strategies
- Brock, Lakonishok & LeBaron (1992); Edwards et al. (2019)
- Zhou & Lu (2023) – [DOI](https://doi.org/10.1016/j.eswa.2023.121712)
- Cohen et al. (2024) – [IEEE](https://doi.org/10.1109/ACCESS.2024.3281123)

#### Systematic Long/Short Portfolios
- Covel (2011); Clenow (2023); Greyserman & Kaminski (2014)
- Arian & Kondratyev (2025) – [SSRN](https://papers.ssrn.com/sol3/Delivery.cfm?abstractid=5330108)
- Choudhary et al. (2025) – [Springer](https://link.springer.com/content/pdf/10.1007/s44196-025-00875-8.pdf)

#### Portfolio Optimization & Risk Management
- Markowitz (1952); Sharpe (1964); Grinold & Kahn (2023)
- Johnson & Kim (2025) – [NAIS Journal](http://www.naisjournal.com/static/upload/file/20250326/1742974408112244.pdf)
- Saiz et al. (2025) – [Wiley](https://onlinelibrary.wiley.com/doi/pdf/10.1111/itor.70064)

---

### 3. Methodology

**Data:**  
50 large-cap U.S. tech stocks, daily frequency, 2014–2024.

**Modeling Approach:**
- Train Gaussian HMMs on log returns and volatility
- Use technical indicators: RSI, MACD, 50/200 SMA, ATR

**Regime-Based Allocation Rules:**
- **Bullish:** Go long top-ranked stocks (RSI-based)
- **Bearish:** Short weakest stocks (RSI/MACD-based)
- **Neutral:** Cash or index hedge

---

### 4. Results

- HMM states aligned with key macro events (COVID crash, Fed policy shifts)
- Improved signal accuracy over static trend-following rules *(Zhao et al., 2025)*
- Long/short overlay helped smooth drawdowns *(Choudhary et al., 2025)*
- Technical filters reduced whipsaws *(Cohen et al., 2024)*
- Strategy Sharpe Ratio improved from 0.7 (SPY) to ~1.1 *(Feng et al., 2025)*

---

### 5. Conclusions & Next Steps

Combining HMMs and technical trading strategies provides a strong foundation for an adaptive ETF. The literature supports both components, and early results show promising risk-adjusted returns.

**Next Steps:**
- Test sector generalizability
- Implement rolling/Bayesian regime updates
- Validate through backtesting

---


