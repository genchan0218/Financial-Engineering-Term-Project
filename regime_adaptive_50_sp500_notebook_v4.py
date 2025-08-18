# %%
# Regime-Adaptive 50-Stock S&P 500 Strategy — v4.4 (macro safety valve)
# - Regime-specific long/short selection
# - HMM regime with SMA200 degrade/upgrade safety
# - Macro safety valve: lighten/disable shorts when SPY is trending up
# - Weekly rebalance + smoothing; tilt-aware beta target (preserves L1)
# - EWMA vol targeting; trading/mgmt/borrow costs
# - SPY (net ER) comparison + diagnostics
#
# Requirements:
#   pip install numpy pandas yfinance matplotlib scikit-learn hmmlearn ta

import math, warnings, datetime as dt, os
from typing import List, Dict
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ---- Optional libs ----
HMM_OK = True
try:
    from hmmlearn.hmm import GaussianHMM
except Exception:
    HMM_OK = False
try:
    from ta.momentum import RSIIndicator
except Exception:
    RSIIndicator = None

# ------------------- USER SETTINGS -------------------
START_DATE     = "2010-01-01"
END_DATE       = dt.datetime.today().strftime("%Y-%m-%d")
OUT_PREFIX     = "results"
OUT_DIR        = r"C:\Users\Genki\Documents\Homework\Financial Machine Learning\Term Project\Financial-Engineering-Term-Project"  # or "."

TARGET_VOL     = 0.15      # annualized target portfolio vol
VOL_WIN        = 20        # EWMA span (days)
TRADE_BPS      = 2.0       # one-way bps per unit L1 turnover
MGMT_ER_BPS    = 0.0       # set 0 while debugging; e.g., 60 for 0.60%/yr
BENCH_ER_BPS   = 9.0       # SPY ER (annual bps)
BORROW_BPS     = 25.0      # short borrow (annual bps)

MIN_YEARS      = 10.0
SEED           = 42

# Regime gross exposure and short permission
REGIME_PARAMS = {
    "bull":    {"gross": 1.20, "allow_shorts": False},
    "neutral": {"gross": 1.00, "allow_shorts": True},
    "bear":    {"gross": 0.80, "allow_shorts": True},
}

# Desired net exposure and beta target by regime (kept unless macro valve overrides)
NET_TILT_BASE    = {"bull": +0.50, "neutral": 0.00, "bear": -0.20}
TARGET_BETA_BASE = {"bull": +0.50, "neutral": 0.00, "bear": -0.20}

REBAL_FREQ  = "W-WED"   # weekly rebalance on Wed
SMOOTH      = 0.70      # 0=no smoothing; 0.7=sticky weights

# Regime selection cutoffs (quantiles)
CUTS_BASE = {
    "bull":    {"long_q": 0.80, "short_q": 0.00},  # long winners; no shorts
    "neutral": {"long_q": 0.80, "short_q": 0.20},  # balanced
    "bear":    {"long_q": 0.90, "short_q": 0.30},  # few contrarian longs; many shorts
}

# SMA override settings
USE_SMA_OVERRIDE = True
SMA_WINDOW       = 200
UPGRADE_THRESH   = +0.005  # +0.5% above SMA → upgrade regime one notch
DEGRADE_THRESH   = -0.005  # -0.5% below SMA → degrade regime one notch

# Macro safety valve (when market is trending up)
MACRO_BULL_LONGFLOOR   = +0.20  # min net tilt when macro is bullish
MACRO_BULL_BETAFLOOR   = +0.20  # min beta target when macro is bullish
MACRO_BULL_SHORT_SHIFT = -0.10  # reduce short quantile by 10% (often to 0)
MACRO_BULL_DIST_HARD   = 0.02   # if above SMA by >2%, turn shorts fully off

# Macro bear intensification (optional, gentle)
MACRO_BEAR_SHORT_BUMP  = +0.10  # add 10% to short_q when regime is bear and macro bearish
MACRO_BEAR_TILT_ADD    = -0.05  # push tilt slightly more negative in macro bearish

# ------------------- UNIVERSE -------------------
def default_universe_50() -> List[str]:
    return [
        "AAPL","MSFT","AMZN","GOOGL","GOOG","META","NVDA","TSLA","JPM","BAC",
        "WFC","C","GS","MS","V","MA","HD","LOW","COST","WMT",
        "TGT","UNH","PFE","MRK","JNJ","ABT","TMO","XOM","CVX","COP",
        "SLB","NKE","SBUX","MCD","PEP","KO","DIS","NFLX","INTU","ADBE",
        "ORCL","CRM","AMD","INTC","AVGO","CSCO","LLY","ABBV","AMGN","BMY"
    ]

# ------------------- HELPERS -------------------
def download_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series): df = df.to_frame()
    df = df.dropna(how="all"); df.columns = [str(c).upper() for c in df.columns]
    return df

def years_between(a: pd.Timestamp, b: pd.Timestamp) -> float:
    return (b - a).days / 365.25

def filter_min_history(px: pd.DataFrame, min_years: float) -> pd.DataFrame:
    end = px.index.max(); keep = []
    for c in px.columns:
        s = px[c].dropna()
        if not s.empty and years_between(s.index.min(), end) >= min_years:
            keep.append(c)
    return px[keep]

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    if RSIIndicator is not None:
        return RSIIndicator(series, window=window).rsi()
    d = series.diff(); up = d.clip(lower=0).rolling(window).mean(); dn = -d.clip(upper=0).rolling(window).mean()
    rs = up / dn.replace(0, np.nan); return 100.0 - (100.0 / (1.0 + rs))

def bollinger_z(series: pd.Series, window: int = 20) -> pd.Series:
    ma = series.rolling(window).mean(); sd = series.rolling(window).std(ddof=0)
    return (series - ma) / sd.replace(0, np.nan)

def dd_series(e: pd.Series) -> pd.Series:
    return e / e.cummax() - 1.0

def apply_net_tilt_preserve_L1(w_row: pd.Series, target_net: float, eps: float = 1e-8) -> pd.Series:
    L = w_row.clip(lower=0); S = (-w_row.clip(upper=0))
    Lsum, Ssum = float(L.sum()), float(S.sum()); L1 = Lsum + Ssum
    if L1 < eps: return w_row
    Ldes = L1 * (1 + target_net) / 2.0; Sdes = L1 * (1 - target_net) / 2.0
    if Ssum < eps and Lsum > eps: return (L / (Lsum + eps)) * L1
    if Lsum < eps and Ssum > eps: return - (S / (Ssum + eps)) * L1
    return L * (Ldes / (Lsum + eps)) - S * (Sdes / (Ssum + eps))

def beta_neutralize_row(w_row: pd.Series, betas_row: pd.Series, target_beta: float, eps: float = 1e-8) -> pd.Series:
    b = betas_row.reindex_like(w_row).fillna(0.0)
    num = float((w_row * b).sum() - target_beta)
    den = float((b * b).sum() + eps)
    w_adj = w_row - (num / den) * b
    l1_target = float(w_row.abs().sum()); l1 = float(w_adj.abs().sum())
    if l1 > eps: w_adj = w_adj * (l1_target / l1)  # preserve L1
    return w_adj

# ------------------- STEP 1: DATA -------------------
print("Downloading data...")
UNIVERSE = default_universe_50()
px_all = download_prices(UNIVERSE, START_DATE, END_DATE)
px = filter_min_history(px_all, MIN_YEARS)
spy = download_prices(["SPY"], START_DATE, END_DATE).iloc[:, 0]

common = px.index.intersection(spy.index)
px, spy = px.reindex(common).dropna(how="all"), spy.reindex(common)

rets, spy_ret = px.pct_change(), spy.pct_change()
betas = rets.rolling(252).cov(spy_ret).div(spy_ret.rolling(252).var())

print(f"Universe: {px.shape[1]} stocks; period: {px.index.min().date()} → {px.index.max().date()}")

# ------------------- STEP 2: SIGNALS -------------------
print("Computing signals...")
# Momentum: 12-1 + 6m (lagged one month)
mom6    = px.shift(1).pct_change(126)
mom12_1 = px.shift(21).pct_change(252 - 21)
mom     = 0.35 * mom6 + 0.65 * mom12_1

# Mean-reversion: oversold composite (higher = more oversold)
rsi14   = px.apply(rsi, window=14)
bbz20   = px.apply(bollinger_z, window=20)
mr_scr  = (100 - rsi14) / 100.0 + (-bbz20)
mr_scr  = mr_scr.replace([np.inf, -np.inf], np.nan)

# Cross-sectional ranks in [0,1]
mom_rank = mom.rank(axis=1, pct=True)
mr_rank  = mr_scr.rank(axis=1, pct=True)

ma200 = px.rolling(SMA_WINDOW).mean()
short_gate = (px < ma200)   # only short below SMA

# ------------------- STEP 3: Regimes -------------------
print("Detecting regimes...")
def regimes_walkforward_hmm(spy_px: pd.Series, seed=42) -> pd.Series:
    r1 = spy_px.pct_change(); r5 = spy_px.pct_change(5); vol20 = r1.rolling(20).std(ddof=0)*np.sqrt(252)
    X = pd.concat([r1, r5, vol20], axis=1).dropna()
    if X.empty: raise ValueError("Insufficient data for HMM")
    mends = X.resample("M").last().index; out = pd.Series(index=X.index, dtype=int)
    if not HMM_OK:
        sma200 = spy_px.rolling(200).mean(); dist = (spy_px/sma200)-1.0
        raw = pd.Series(np.where(dist>0.02,2,np.where(dist<-0.02,0,1)), index=spy_px.index)
        return raw.reindex(X.index).ffill().astype(int)
    for i in range(12, len(mends)-1):
        tr_end, te_end = mends[i], mends[i+1]
        Xtr = X.loc[:tr_end].dropna()
        if len(Xtr) < 200: continue
        hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=200, random_state=seed)
        hmm.fit(Xtr.values)
        s_tr = pd.Series(hmm.predict(Xtr.values), index=Xtr.index)
        mean_r = r1.reindex(Xtr.index).groupby(s_tr).mean().sort_values()
        order  = list(mean_r.index); map_ = {order[0]:0, order[1]:1, order[2]:2}
        Xte = X.loc[tr_end:te_end].iloc[1:]
        if len(Xte)==0: continue
        s_te = pd.Series(hmm.predict(Xte.values), index=Xte.index)
        out.loc[Xte.index] = s_te.map(map_)
    return out.ffill().bfill().astype(int)

raw_states   = regimes_walkforward_hmm(spy, seed=SEED)
base_regimes = raw_states.map({0:"bear",1:"neutral",2:"bull"}).reindex(px.index).ffill().bfill()

# SMA upgrade/degrade
if USE_SMA_OVERRIDE:
    spy_sma = spy.rolling(SMA_WINDOW).mean()
    dist = (spy / spy_sma) - 1.0
    override = pd.Series(index=px.index, dtype=object)
    for t in px.index:
        r = base_regimes.loc[t]
        d = dist.loc[t]
        if pd.notna(d):
            if d < DEGRADE_THRESH:   # degrade one level
                r = "neutral" if r == "bull" else ("bear" if r == "neutral" else "bear")
            elif d > UPGRADE_THRESH: # upgrade one level
                r = "bull" if r == "neutral" else ("neutral" if r == "bear" else "bull")
        override.loc[t] = r
    regimes = override
else:
    regimes = base_regimes

print("Regime counts:\n", regimes.value_counts())

# Additional macro context (for safety valve)
spy_sma200 = spy.rolling(SMA_WINDOW).mean()
dist_to_sma = (spy / spy_sma200) - 1.0
spy_mom_1m  = spy.pct_change(21)

# ------------------- STEP 4: Weights (regime-specific selection with macro valve) -------------------
print("Building weights...")
weights_daily = pd.DataFrame(0.0, index=px.index, columns=px.columns)

for t in px.index:
    reg = regimes.loc[t]
    pars = REGIME_PARAMS[reg]
    gross, allow_shorts = pars["gross"], pars["allow_shorts"]

    # --- dynamic parameters with macro valve ---
    net_tilt  = NET_TILT_BASE[reg]
    beta_tgt  = TARGET_BETA_BASE[reg]
    cuts      = CUTS_BASE[reg].copy()
    d = dist_to_sma.loc[t]
    m1 = spy_mom_1m.loc[t]
    macro_bull = (pd.notna(d) and pd.notna(m1) and (d > 0) and (m1 > 0))
    macro_bear = (pd.notna(d) and pd.notna(m1) and (d < 0) and (m1 < 0))

    if macro_bull:
        # lighten shorts materially; raise net/beta floors
        cuts["short_q"] = max(0.0, cuts["short_q"] + MACRO_BULL_SHORT_SHIFT)
        if d > MACRO_BULL_DIST_HARD:
            allow_shorts = False
            cuts["short_q"] = 0.0
        net_tilt = max(net_tilt, MACRO_BULL_LONGFLOOR)
        beta_tgt = max(beta_tgt, MACRO_BULL_BETAFLOOR)
    elif macro_bear and reg == "bear":
        # more shorts when macro lines up with bear regime
        cuts["short_q"] = min(0.9, cuts["short_q"] + MACRO_BEAR_SHORT_BUMP)
        net_tilt = net_tilt + MACRO_BEAR_TILT_ADD

    # --- regime-specific ranking for long/short sleeves ---
    if reg == "bull":
        long_score  = mom_rank.loc[t]                      # long winners
        short_score = pd.Series(0.0, index=px.columns)     # none
    elif reg == "neutral":
        long_score  = 0.6*mom_rank.loc[t] + 0.4*mr_rank.loc[t]
        short_score = 0.6*(1.0 - mr_rank.loc[t]) + 0.4*(1.0 - mom_rank.loc[t])
    else:  # bear
        long_score  = mr_rank.loc[t]                       # small contrarian longs
        short_score = (1.0 - mom_rank.loc[t])              # short losers (weak momentum)

    # Convert to masks
    qL, qS = cuts["long_q"], cuts["short_q"]
    long_mask  = long_score >= long_score.quantile(qL)
    short_mask = (short_score >= short_score.quantile(qS)) & short_gate.loc[t].fillna(False) if allow_shorts and qS>0 else pd.Series(False, index=px.columns)

    # Equal-weight within sleeves → L1 normalize
    nL, nS = int(long_mask.sum()), int(short_mask.sum())
    wL = (long_mask.astype(float)/nL) if nL>0 else pd.Series(0.0, index=px.columns)
    wS = (short_mask.astype(float)/nS) if nS>0 else pd.Series(0.0, index=px.columns)
    w  = wL - wS
    L1 = float(w.abs().sum());  w = w/L1 if L1>0 else w

    # Target beta then enforce net tilt (both preserve L1)
    w = beta_neutralize_row(w, betas.loc[t], target_beta=beta_tgt)
    w = apply_net_tilt_preserve_L1(w, target_net=net_tilt)

    weights_daily.loc[t] = w * gross

# Weekly targets → align to all days → back/forward fill → smooth
weights = weights_daily.resample(REBAL_FREQ).last()
weights = weights.reindex(px.index).bfill().ffill().fillna(0.0)

weights_sm = weights.copy()
for i in range(1, len(weights.index)):
    weights_sm.iloc[i] = SMOOTH*weights_sm.iloc[i-1] + (1-SMOOTH)*weights.iloc[i]
weights = weights_sm

print("Mean |weights| (pre-scale):", float(weights.abs().sum(axis=1).mean()))
print("Mean raw beta (pre-scale):", float((weights*betas).sum(axis=1).mean()))

# ------------------- STEP 5: Vol targeting + costs + returns -------------------
gross_port_ret = (weights.shift(1) * rets).sum(axis=1)
realized = gross_port_ret.ewm(span=VOL_WIN, adjust=False).std() * np.sqrt(252)
scale = (TARGET_VOL / realized).clip(upper=3.0).fillna(1.0)
weights_scaled = weights.mul(scale, axis=0)

turnover      = weights_scaled.diff().abs().sum(axis=1).fillna(0.0)
trading_costs = turnover * (TRADE_BPS/10000.0)
mgmt_fee      = pd.Series((MGMT_ER_BPS/10000.0)/252.0, index=px.index)
borrow_costs  = ((BORROW_BPS/10000.0)/252.0) * weights_scaled.clip(upper=0).abs().sum(axis=1)

port_ret = (weights_scaled.shift(1) * rets).sum(axis=1)
net_ret  = port_ret - trading_costs - mgmt_fee - borrow_costs
equity   = (1.0 + net_ret.fillna(0.0)).cumprod()

bench_er_daily = (BENCH_ER_BPS/10000.0)/252.0
bench_net_ret  = spy_ret - bench_er_daily
bench_equity   = (1.0 + bench_net_ret.fillna(0.0)).cumprod()

# ------------------- STEP 6: Logs & Summary -------------------
os.makedirs(OUT_DIR, exist_ok=True)
tradelog = pd.DataFrame({
    "regime": regimes,
    "gross_ret": port_ret,
    "trading_costs": -trading_costs,
    "mgmt_fee": -mgmt_fee,
    "borrow_costs": -borrow_costs,
    "net_ret": net_ret,
    "equity": equity,
    "bench_net_ret": bench_net_ret,
    "bench_equity": bench_equity
}, index=px.index)
monthly_weights = weights_scaled.resample("M").last().dropna(how="all")
tradelog.to_csv(os.path.join(OUT_DIR, f"{OUT_PREFIX}_tradelog.csv"), float_format="%.8f")
monthly_weights.to_csv(os.path.join(OUT_DIR, f"{OUT_PREFIX}_weights.csv"), float_format="%.6f")

def summary_metrics(net: pd.Series, eq: pd.Series) -> Dict[str, float]:
    net = net.dropna()
    if net.empty or len(eq)<2: return {"CAGR":0.0,"AnnVol":0.0,"Sharpe":0.0,"MaxDD":0.0,"TotalReturn":0.0}
    total = float((1+net).prod()); years = (net.index[-1]-net.index[0]).days/365.25
    cagr  = total**(1/years) - 1 if years>0 else 0.0
    annv  = float(net.std(ddof=0)*math.sqrt(252))
    shar  = float((net.mean()*252)/annv) if annv>0 else 0.0
    maxdd = float((eq/eq.cummax()-1).min())
    tot   = float(eq.iloc[-1]/eq.iloc[0]-1)
    return {"CAGR":cagr,"AnnVol":annv,"Sharpe":shar,"MaxDD":maxdd,"TotalReturn":tot}

summary_df = pd.DataFrame([
    {"Series":"Strategy (net)", **summary_metrics(net_ret, equity)},
    {"Series":"SPY (net ER)",   **summary_metrics(bench_net_ret, bench_equity)},
])
summary_df.to_csv(os.path.join(OUT_DIR, f"{OUT_PREFIX}_summary.csv"), index=False, float_format="%.6f")
print(summary_df)

# Diagnostics
port_beta_series = (weights_scaled * betas).sum(axis=1)
bullish_days = (dist_to_sma > 0) & (spy_mom_1m > 0)
print("Avg daily turnover (L1):", float(turnover.mean()))
print("Mean post-scale gross:", float(weights_scaled.abs().sum(axis=1).mean()))
print("Mean realized beta (post-scale):", float(port_beta_series.mean()))
print("Mean realized beta on bullish macro days:", float(port_beta_series[bullish_days].mean()))
print("Share of days with beta < -0.05 while macro bullish:", float((port_beta_series[bullish_days] < -0.05).mean()))
print("Corr(net_ret, SPY):", float(pd.Series(net_ret).corr(spy_ret)))

# ------------------- STEP 7: Plots -------------------
plt.figure()
equity.plot(label="Strategy (net fees)")
bench_equity.plot(label="SPY (net ER)")
plt.legend(); plt.title("Equity Curve: Strategy vs SPY")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f"{OUT_PREFIX}_equity_vs_spy.png"), dpi=150)

plt.figure()
(equity/equity.cummax()-1).plot(label="Strategy DD")
(bench_equity/bench_equity.cummax()-1).plot(label="SPY DD")
plt.legend(); plt.title("Rolling Drawdowns")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f"{OUT_PREFIX}_drawdowns.png"), dpi=150)

plt.figure()
equity.plot(); plt.title("Regime-Adaptive 50-Stock Strategy Equity (net fees)")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f"{OUT_PREFIX}_equity.png"), dpi=150)
# plt.show()
