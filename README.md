# TradingBot
## Lumibot Trading Strategy with FinBERT Sentiment Analysis

This README provides an overview of a trading strategy implemented using Lumibot, a trading automation framework. The strategy combines financial news sentiment analysis using FinBERT with trading signals generated from the sentiment analysis results.

### Lumibot Components

1. **Alpaca Broker**: The strategy uses Alpaca as the brokerage platform for executing trades in a paper trading environment.

2. **MLTrader Strategy**: The MLTrader strategy, inheriting from Lumibot's `Strategy` class, implements the trading logic. It integrates sentiment analysis of financial news articles with trading decisions.

3. **YahooDataBacktesting**: Lumibot's `YahooDataBacktesting` module is used for backtesting the trading strategy with historical data.

### Usage

1. **Alpaca API Credentials**: Ensure to set up Alpaca API credentials (`API_KEY` and `API_SECRET`) and specify the base URL.

2. **Initialize MLTrader**: Create an instance of the `MLTrader` strategy with specified parameters such as the trading symbol (`symbol`) and the amount of cash at risk (`cash_at_risk`).

3. **Backtesting**: Define the start and end dates for backtesting historical data. Instantiate the Alpaca broker and the MLTrader strategy. Then, perform backtesting using the `backtest` method of the strategy.

### FinBERT Sentiment Analysis

FinBERT sentiment analysis is performed using a pre-trained model and tokenizer. The sentiment labels include positive, negative, and neutral. The `estimate_sentiment` function tokenizes news articles, passes them through the model, and predicts sentiment labels along with corresponding probabilities.

### Example Usage

```python
# Example usage of the estimate_sentiment function
tensor, sentiment = estimate_sentiment(['markets responded positively to the news!', 'traders were pleasantly surprised!!'])

# Print the sentiment probability and label
print(tensor, sentiment)

# Print whether CUDA (GPU) is available
print("CUDA Available:", torch.cuda.is_available())
```

Ensure to have the necessary libraries installed and set up CUDA for GPU acceleration if available.

### Dependencies

- lumibot
- alpaca_trade_api
- finbert_utils
- transformers
- torch

### Setup

1. Install the required dependencies.
2. Set up Alpaca API credentials.
3. Load the pre-trained FinBERT model and tokenizer.
4. Run the Lumibot trading strategy for backtesting or live trading.

### Note

This README provides an overview and usage instructions for the Lumibot trading strategy with FinBERT sentiment analysis. Ensure to review and customize the code according to specific requirements and trading objectives.
