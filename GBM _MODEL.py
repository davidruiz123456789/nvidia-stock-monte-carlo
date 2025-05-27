import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# === STEP 1: Load and Prepare Data ===
file_path = "/Users/davidruiz/Desktop/data_stocks/NVIDIA_Stock_Price_Data.csv"
data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
data = data.sort_index()

print("📊 Last 5 rows of data:")
print(data.tail())
print(f"\nMost recent date: {data.index[-1]}")
print(f"Most recent price: ${data['Price'].iloc[-1]:.2f}")

data['Returns'] = data['Price'].pct_change()
data.dropna(inplace=True)

# === STEP 2: Enhanced Parameter Estimation ===

# 2.1 Dynamic Drift Calculation
returns = data['Returns'].dropna()
historical_drift = returns.mean() * 252  # Annualized historical return
risk_free_rate = 0.045  # Current 10-year Treasury rate (approximate)
market_premium = 0.06   # Historical equity risk premium
capm_drift = risk_free_rate + market_premium

# Use average of historical and CAPM-based drift
mu = min((historical_drift + capm_drift) / 2, 0.12)  # Cap drift to 18% max

print(f"\n📈 Enhanced Drift Calculation:")
print(f"Historical Annual Return: {historical_drift:.4f} ({historical_drift*100:.2f}%)")
print(f"CAPM-based Expected Return: {capm_drift:.4f} ({capm_drift*100:.2f}%)")
print(f"Final Drift (μ): {mu:.4f} ({mu*100:.2f}%)")

# 2.2 Multi-timeframe Volatility
vol_30d = returns.tail(30).std() * np.sqrt(252)    # Short-term
vol_90d = returns.tail(90).std() * np.sqrt(252)    # Medium-term
vol_252d = returns.tail(252).std() * np.sqrt(252)  # Long-term

# Weighted average volatility (more weight on recent)
weights = [0.5, 0.3, 0.2]  # 50% short, 30% medium, 20% long
sigma = weights[0] * vol_30d + weights[1] * vol_90d + weights[2] * vol_252d

print(f"\n📊 Multi-timeframe Volatility:")
print(f"30-day Volatility: {vol_30d:.4f} ({vol_30d*100:.2f}%)")
print(f"90-day Volatility: {vol_90d:.4f} ({vol_90d*100:.2f}%)")
print(f"252-day Volatility: {vol_252d:.4f} ({vol_252d*100:.2f}%)")
print(f"Weighted Volatility (σ): {sigma:.4f} ({sigma*100:.2f}%)")

# 2.3 Regime Detection (Simple Bull/Bear Market Indicator)
ma_50 = data['Price'].rolling(50).mean()
ma_200 = data['Price'].rolling(200).mean()
current_regime = "Bull" if ma_50.iloc[-1] > ma_200.iloc[-1] else "Bear"

# Adjust parameters based on regime
if current_regime == "Bull":
    regime_drift_adj = 1.1  # 10% boost in bull market
    regime_vol_adj = 0.9    # 10% lower volatility in bull market
else:
    regime_drift_adj = 0.8  # 20% reduction in bear market
    regime_vol_adj = 1.2    # 20% higher volatility in bear market

mu_adjusted = mu * regime_drift_adj
sigma_adjusted = sigma * regime_vol_adj

print(f"\n🎯 Market Regime Analysis:")
print(f"Current Regime: {current_regime} Market")
print(f"Regime-Adjusted Drift: {mu_adjusted:.4f} ({mu_adjusted*100:.2f}%)")
print(f"Regime-Adjusted Volatility: {sigma_adjusted:.4f} ({sigma_adjusted*100:.2f}%)")

# === STEP 3: Jump Diffusion Model Enhancement ===
# Detect historical jumps (returns > 2 standard deviations)
jump_threshold = 2 * returns.std()
jumps = returns[abs(returns) > jump_threshold]
jump_frequency = len(jumps) / len(returns) * 252  # Jumps per year
jump_magnitude = jumps.std() if len(jumps) > 0 else 0

print(f"\n⚡ Jump Analysis:")
print(f"Jump Frequency: {jump_frequency:.2f} jumps per year")
print(f"Average Jump Magnitude: {jump_magnitude:.4f}")

# === STEP 4: Enhanced Monte Carlo Simulation ===
S0 = data['Price'].iloc[-1]
actual_last_date = data.index[-1]
start_date = actual_last_date + pd.Timedelta(days=1)
T = 5
dt = 1 / 252
N = int(T / dt)
M = 1000

print(f"\n🚀 Simulation Parameters:")
print(f"Starting Price: ${S0:.2f}")
print(f"Starting Date: {start_date.strftime('%Y-%m-%d')}")
print(f"Simulation Period: {T} years ({N} days)")
print(f"Number of Paths: {M}")

# Generate future dates
future_dates = pd.bdate_range(start=start_date, periods=N, freq='B')
all_dates = [actual_last_date] + list(future_dates)

# Enhanced Monte Carlo with multiple improvements
paths = np.zeros((N + 1, M))
paths[0] = S0

# Time-varying volatility decay (volatility mean-reverts over time)
vol_decay_rate = 0.1  # How fast volatility reverts to long-term mean
long_term_vol = 0.25  # Long-term average volatility

for t in range(1, N + 1):
    # Time-varying volatility
    time_factor = t / N
    current_vol = sigma_adjusted * np.exp(-vol_decay_rate * time_factor) + \
                  long_term_vol * (1 - np.exp(-vol_decay_rate * time_factor))
    
    # Random components
    Z = np.random.standard_normal(M)
    
    # Jump component (Poisson process)
    jump_occur = np.random.poisson(jump_frequency * dt, M)
    jump_sizes = np.random.normal(0, jump_magnitude, M) * jump_occur
    
    # GBM with jumps and time-varying volatility
    diffusion = (mu_adjusted - 0.5 * current_vol**2) * dt + current_vol * np.sqrt(dt) * Z
    paths[t] = paths[t - 1] * np.exp(diffusion + jump_sizes)

# === STEP 5: Mean Reversion Component ===
# Add subtle mean reversion (Ornstein-Uhlenbeck component)
mean_reversion_strength = 0.05
long_term_price = S0 * np.exp(mu_adjusted * T)  # Expected long-term price

for t in range(1, N + 1):
    reversion_factor = mean_reversion_strength * (long_term_price - paths[t]) / paths[t] * dt
    paths[t] *= (1 + reversion_factor)

# === STEP 6: Expected Paths Calculation ===
# Drift-only path
expected_path = [S0]
for i in range(N):
    expected_path.append(expected_path[-1] * np.exp(mu_adjusted * dt))

# === STEP 7: Enhanced Visualization ===
mean_path = np.mean(paths, axis=1)
median_path = np.median(paths, axis=1)
upper_95 = np.percentile(paths, 95, axis=1)
lower_5 = np.percentile(paths, 5, axis=1)
upper_75 = np.percentile(paths, 75, axis=1)
lower_25 = np.percentile(paths, 25, axis=1)

plt.figure(figsize=(15, 10))

# Plot sample paths with different colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
for i in range(10):
    plt.plot(all_dates, paths[:, i], lw=1.2, alpha=0.6, color=colors[i])

# Enhanced confidence bands
plt.fill_between(all_dates, lower_5, upper_95, color='lightgray', alpha=0.3, label='90% Confidence')
plt.fill_between(all_dates, lower_25, upper_75, color='gray', alpha=0.3, label='50% Confidence')

# Key paths
plt.plot(all_dates, mean_path, color='black', lw=3, label='Mean Path')
plt.plot(all_dates, median_path, color='red', lw=2, linestyle='--', label='Median Path')
plt.plot(all_dates, expected_path, color='orange', lw=2, linestyle=':', label='Expected (Drift) Path')

# Current price reference
plt.axhline(y=S0, color='blue', linestyle=':', alpha=0.7, label=f'Current Price (${S0:.2f})')

plt.title(f"📈 Enhanced Stock Forecast - Advanced Monte Carlo Model")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === STEP 8: Enhanced Forecast Analysis ===
horizons = {
    "3 months": 63,
    "6 months": 126,
    "9 months": 189,
    "1 year": 252,
    "2 years": 504,
    "3 years": 756,
    "4 years": 1008,
    "5 years": 1260
}

print(f"\n📊 Enhanced Price Forecasts (Starting: ${S0:.2f} on {actual_last_date.strftime('%Y-%m-%d')}):")
print(f"{'Horizon':<10} | {'Target Date':<12} | {'Expected':>10} | {'Mean':>10} | {'Median':>10} | {'5%':>8} | {'95%':>9}")
print("-" * 85)

for label, step in horizons.items():
    if step <= N:
        target_date = future_dates[step-1]
        expected_val = expected_path[step]
        mean_val = mean_path[step]
        median_val = median_path[step]
        low_5 = np.percentile(paths[step], 5)
        high_95 = np.percentile(paths[step], 95)
        
        print(f"{label:<10} | {target_date.strftime('%Y-%m-%d'):<12} | ${expected_val:9.2f} | ${mean_val:9.2f} | ${median_val:9.2f} | ${low_5:7.2f} | ${high_95:8.2f}")

# === STEP 9: Risk Metrics ===
final_prices = paths[-1]
print(f"\n📈 5-Year Risk Analysis:")
print(f"Value at Risk (5%): ${np.percentile(final_prices, 5):.2f}")
print(f"Expected Shortfall (5%): ${np.mean(final_prices[final_prices <= np.percentile(final_prices, 5)]):.2f}")
print(f"Probability of Loss: {np.mean(final_prices < S0) * 100:.1f}%")
print(f"Probability of 50% Gain: {np.mean(final_prices >= 1.5 * S0) * 100:.1f}%")
print(f"Probability of Doubling: {np.mean(final_prices >= 2 * S0) * 100:.1f}%")
print(f"Maximum Drawdown Risk: {((S0 - np.min(paths)) / S0 * 100):.1f}%")

# === STEP 10: Model Validation Metrics ===
print(f"\n🔍 Model Validation:")
print(f"Sharpe Ratio (Expected): {(mu_adjusted - risk_free_rate) / sigma_adjusted:.2f}")
print(f"Volatility of Returns: {sigma_adjusted * 100:.1f}% annually")
print(f"Skewness of Final Prices: {stats.skew(final_prices):.2f}")
print(f"Kurtosis of Final Prices: {stats.kurtosis(final_prices):.2f}")

# === STEP 11: Bullish vs Bearish Forecast Sentiment ===
bullish_pct = np.mean(final_prices > S0) * 100
bearish_pct = np.mean(final_prices < S0) * 100
neutral_pct = 100 - bullish_pct - bearish_pct  # if any flat paths

print(f"\n📊 Monte Carlo Sentiment Analysis (at T = {T} years):")
print(f"🔼 Bullish Paths (price > S₀): {bullish_pct:.2f}%")
print(f"🔽 Bearish Paths (price < S₀): {bearish_pct:.2f}%")
print(f"⏸️ Neutral Paths (price ≈ S₀): {neutral_pct:.2f}%")
