# PPO-Based Portfolio Rebalancer

This project implements a reinforcement learning (RL) agent using Proximal Policy Optimization (PPO) to learn dynamic portfolio rebalancing strategies. The agent adjusts capital allocation across a diversified set of large-cap U.S. stocks and aims to outperform baseline strategies such as equal-weighted and buy-and-hold portfolios.

---

## Overview

- **RL Algorithm**: Proximal Policy Optimization (PPO)
- **Library**: Stable-Baselines3
- **Environment**: Custom Gym-style portfolio environment
- **Assets Used**: AAPL, MSFT, NVDA, GOOGL, META, AMZN, TSLA, KO, JPM, MA, UNH, JNJ, XOM, BA, NEE
- **Baselines**: Equal Weight, Buy-and-Hold
- **Key Features**:
  - Custom reward combining return and risk (Sortino-based)
  - Transaction cost penalty
  - Normalized vectorized training with `VecNormalize`
  - Evaluation using risk-adjusted metrics

---

## Environment Description

A custom OpenAI Gym-compatible environment (`PortfolioEnv`) simulates multi-asset portfolio trading. The environment provides log returns as state features and accepts continuous portfolio weights as actions. Rewards are computed from net log returns adjusted for transaction costs and shaped using the Sortino ratio.

### Observations:
- Windowed log returns of all assets
- Previous portfolio weights

### Action Space:
- Continuous vector representing asset weights (softmax-normalized)


### Reward Function

At each timestep, the reward $R_t$ is defined as a weighted combination of **net log return** and the **Sortino ratio**:

```
```math
R_t = \alpha \cdot r_t + (1 - \alpha) \cdot S_t
```
```


**Where:**

- $r_t = \log\left(\frac{P_t}{P_{t-1}}\right) - c \cdot \text{turnover}_t$  
  → net log return after accounting for transaction cost

- $S_t = \left(\frac{\mu_w}{\sigma_d + \varepsilon}\right) \cdot \sqrt{252}$  
  → annualized Sortino ratio over the recent return window


**Definitions:**

| Symbol        | Description                                      |
|---------------|--------------------------------------------------|
| $\mu_w$      | Mean return over a rolling reward window         |
| $\sigma_d$   | Standard deviation of downside returns           |
| $\varepsilon$| Small constant to avoid division-by-zero         |
| $\alpha$     | Blending weight between net return and Sortino   |
| $c$           | Transaction cost coefficient                     |

---

## Data Source

Price data is fetched from Yahoo Finance using the `yfinance` library. The dataset consists of adjusted close prices from January 2020 to January 2025 for 15 large-cap stocks across sectors such as technology, finance, energy, healthcare, and consumer staples.

---

## Training

The PPO agent is trained on data from 2020 to 2023 using vectorized and normalized environments. An evaluation environment is held out for 2024 data to assess generalization. The best model is selected based on performance during periodic evaluation using a callback mechanism.

---

## Evaluation

Performance is compared against two baseline strategies:

- **Equal Weight**: Capital equally split across all assets at each timestep
- **Buy-and-Hold**: Full capital allocated to the first asset and held throughout

### Evaluation Metrics:
- Final Portfolio Value
- Cumulative Return
- Sharpe Ratio
- Maximum Drawdown
- Annualized Volatility

---

## Future Work

- Expand asset universe to include ETFs or macroeconomic instruments
- Incorporate additional signals or indicators
- Introduce market frictions like slippage or leverage
- Explore recurrent architectures for regime awareness

---
