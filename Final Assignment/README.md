
# PPO-Based Portfolio Rebalancer

This project implements a reinforcement learning (RL) agent using Proximal Policy Optimization (PPO) to learn portfolio rebalancing strategies. The agent dynamically allocates capital across a basket of large-cap technology stocks and aims to outperform baseline strategies such as equal-weighted and buy-and-hold portfolios.

---

## Overview

- **RL Algorithm**: Proximal Policy Optimization (PPO)
- **Library**: Stable-Baselines3
- **Environment**: Custom Gym-style portfolio environment
- **Assets Used**: AAPL, MSFT, GOOGL, AMZN, META
- **Baselines**: Equal Weight, Buy-and-Hold
- **Key Features**:
  - Vectorized observations and reward normalization using `VecNormalize`
  - Incorporates technical indicators (RSI, volatility, moving averages)
  - Accounts for transaction costs
  - Evaluation using risk-adjusted performance metrics

---

## Environment Description

A Gym-compatible environment (`PortfolioEnv`) simulates multi-asset trading by feeding technical indicators as state features and taking portfolio weights as continuous actions. The agent earns rewards based on log-returns net of transaction fees.

### State Features:
- Price Relatives
- Moving Averages (5-day and 20-day)
- Rolling Volatility (20-day)
- Relative Strength Index (RSI, 5-day)

### Action Space:
- Continuous vector of portfolio weights summing to 1

### Reward Function:
- Log return adjusted for transaction costs

---

## Data Source

Historical price data is fetched using Yahoo Finance via the `yfinance` library. The dataset includes the adjusted closing prices of selected tickers from January 2020 to January 2025.

---

## Training

The PPO agent is trained on price data from 2020 to 2023 using Stable-Baselines3. A normalized vectorized environment (`VecNormalize`) is used to stabilize training. Evaluation is conducted on the 2024 period.

from stable_baselines3 import PPO

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=500_000, callback=eval_callback)
model.save("ppo_portfolio_rebalancer")

---

## Evaluation

Performance is evaluated against the following baseline strategies:

- **Equal Weight**: Capital equally distributed across assets
- **Buy-and-Hold**: 100% allocation to a single asset throughout

### Evaluation Metrics:
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Annualized Volatility
- Calmar Ratio
- Final Portfolio Value

---

## Future Work

- Incorporating additional technical and macroeconomic features
- Using recurrent policies (e.g., LSTM-based PPO)
- Expanding the asset universe to include ETFs, bonds, or commodities
- Modeling slippage and leverage
- Adding position constraints and regulatory limits

---
