import os
import pandas as pd
import numpy as np
from typing import List, Tuple
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
SYMBOL = "BTCUSDT"
LEVERAGE = 10
INITIAL_INVESTMENT = 10  # USD
STOP_LOSS_PCT = 0.002  # 0.2%
TAKE_PROFIT_PCT = 0.004  # 0.4%
SMA_PERIOD = 200
EMA_PERIOD = 50
MAX_TRADES_PER_CROSSOVER = 5
MAX_DAILY_TRADES = 10
COOLDOWN_PERIOD = 300  # 5 minutes in seconds

# Initialize Binance client
client = Client(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET"))


def load_historical_data(
    symbol: str, interval: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """Fetch historical data from Binance."""
    try:
        klines = client.futures_historical_klines(
            symbol=symbol, interval=interval, start_str=start_date, end_str=end_date
        )

        df = pd.DataFrame(
            klines,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]]
        df = df.astype(float)

        return df

    except BinanceAPIException as e:
        print(f"An error occurred: {e}")
        return None


def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return data.rolling(window=period).mean()


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return data.ewm(span=period, adjust=False).mean()


def simulate_trade(entry_price: float, exit_price: float, side: str) -> float:
    """Simulate a trade and return the profit/loss percentage."""
    if side == "BUY":
        return (exit_price - entry_price) / entry_price
    else:
        return (entry_price - exit_price) / entry_price


def backtest(data: pd.DataFrame) -> Tuple[float, int, int, List[float]]:
    """Backtest the scalping strategy."""
    data["SMA"] = calculate_sma(data["close"], SMA_PERIOD)
    data["EMA"] = calculate_ema(data["close"], EMA_PERIOD)

    trades = []
    trades_since_crossover = 0
    daily_trades = 0
    last_trend = None
    last_trade_time = data.index[0]
    total_profit = 0
    winning_trades = 0
    losing_trades = 0

    for i in range(1, len(data)):
        current_time = data.index[i]

        # Check if we've reached the daily trade limit
        if (current_time - last_trade_time).total_seconds() > 86400:  # 24 hours
            daily_trades = 0

        if daily_trades >= MAX_DAILY_TRADES:
            continue

        # Check if we're in the cooldown period
        if (current_time - last_trade_time).total_seconds() < COOLDOWN_PERIOD:
            continue

        long_condition = (
            data["close"][i] > data["SMA"][i] and data["EMA"][i] > data["SMA"][i]
        )
        short_condition = (
            data["close"][i] < data["SMA"][i] and data["EMA"][i] < data["SMA"][i]
        )

        if long_condition and (
            last_trend != "LONG" or trades_since_crossover < MAX_TRADES_PER_CROSSOVER
        ):
            if last_trend != "LONG":
                trades_since_crossover = 0
                last_trend = "LONG"

            entry_price = data["close"][i]
            take_profit_price = entry_price * (1 + TAKE_PROFIT_PCT)
            stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)

            for j in range(i + 1, len(data)):
                if data["high"][j] >= take_profit_price:
                    profit = simulate_trade(entry_price, take_profit_price, "BUY")
                    total_profit += profit
                    trades.append(profit)
                    winning_trades += 1
                    break
                elif data["low"][j] <= stop_loss_price:
                    loss = simulate_trade(entry_price, stop_loss_price, "BUY")
                    total_profit += loss
                    trades.append(loss)
                    losing_trades += 1
                    break

            trades_since_crossover += 1
            daily_trades += 1
            last_trade_time = current_time

        elif short_condition and (
            last_trend != "SHORT" or trades_since_crossover < MAX_TRADES_PER_CROSSOVER
        ):
            if last_trend != "SHORT":
                trades_since_crossover = 0
                last_trend = "SHORT"

            entry_price = data["close"][i]
            take_profit_price = entry_price * (1 - TAKE_PROFIT_PCT)
            stop_loss_price = entry_price * (1 + STOP_LOSS_PCT)

            for j in range(i + 1, len(data)):
                if data["low"][j] <= take_profit_price:
                    profit = simulate_trade(entry_price, take_profit_price, "SELL")
                    total_profit += profit
                    trades.append(profit)
                    winning_trades += 1
                    break
                elif data["high"][j] >= stop_loss_price:
                    loss = simulate_trade(entry_price, stop_loss_price, "SELL")
                    total_profit += loss
                    trades.append(loss)
                    losing_trades += 1
                    break

            trades_since_crossover += 1
            daily_trades += 1
            last_trade_time = current_time

    return total_profit, winning_trades, losing_trades, trades


def run_backtest():
    """Run the backtest and print results."""
    print(f"Starting backtest for {SYMBOL}")

    # Load historical data using Binance API
    data = load_historical_data(SYMBOL, "5m", "2023-01-01", "2023-04-30")

    if data is None:
        print("Failed to load historical data. Exiting.")
        return

    total_profit, winning_trades, losing_trades, trades = backtest(data)

    print(f"Total profit: {total_profit:.2%}")
    print(f"Number of trades: {len(trades)}")
    print(f"Winning trades: {winning_trades}")
    print(f"Losing trades: {losing_trades}")
    print(f"Win rate: {winning_trades / len(trades):.2%}")
    print(f"Average profit per trade: {np.mean(trades):.2%}")
    print(f"Sharpe ratio: {np.mean(trades) / np.std(trades):.2f}")

    # Calculate daily returns
    daily_returns = [
        sum(trades[i : i + MAX_DAILY_TRADES])
        for i in range(0, len(trades), MAX_DAILY_TRADES)
    ]
    print(f"Average daily return: {np.mean(daily_returns):.2%}")
    print(f"Maximum daily return: {max(daily_returns):.2%}")
    print(f"Minimum daily return: {min(daily_returns):.2%}")

    # Calculate total return
    total_return = (1 + total_profit) * INITIAL_INVESTMENT - INITIAL_INVESTMENT
    print(f"Total return on ${INITIAL_INVESTMENT} investment: ${total_return:.2f}")


if __name__ == "__main__":
    run_backtest()
