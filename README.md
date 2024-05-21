# Reinforcement Learning for Stock Market Trading: A Case Study on Nvidia Stock

This project applies a Q-learning agent to develop a trading strategy that maximizes profit through stock trading. The environment is based on historical stock prices of Nvidia over the past two years, containing 504 entries from 02/01/2021 to 01/31/2023.

### Key Points

- **Objective:** Use Q-learning to learn stock price trends and execute profitable trades.
- **Actions:** Buy, sell, or hold the stock.
- **Initial Capital:** $100,000.
- **Performance Metric:** Percentage return on investment (ROI).
- **Dataset Features:** Includes open price, intraday high and low, close price, adjusted close price, and trading volume.
- **Output:** Save the Q-table as a pickle file and attach it to your assignment submission.

### Steps

1. **Implement Q-learning:** Adapt your Q-learning agent to the stock trading environment.
2. **Train the Agent:** Use the historical stock price data to train the agent.
3. **Execute Trades:** The agent will decide whether to buy, sell, or hold based on the learned strategy.
4. **Evaluate Performance:** Measure the agent's performance in terms of ROI.
5. **Save Q-table:** Save the trained Q-table as a pickle file for submission.

The goal is to leverage Q-learning to devise a optimize trading strategy that optimizes profit by effectively learning and acting on stock price trends.