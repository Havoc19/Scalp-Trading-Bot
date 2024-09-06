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
LEVERAGE = 5  # Lower leverage to reduce risk
INITIAL_INVESTMENT = 10  # USD
STOP_LOSS_MULTIPLIER = 2.0  # Slightly wider stop-loss
TAKE_PROFIT_MULTIPLIER = 3.0  # Improved risk-reward ratio
ATR_PERIOD = 14  # Period for ATR calculation
SMA_PERIOD = 200  # Longer period for SMA to determine trend
CANDLESTICK_PATTERNS = [
    "BullishEngulfing",
    "BearishEngulfing",
    "PinBar",
]  # Focus on key reversal patterns

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


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)."""
    high_low = data["high"] - data["low"]
    high_close = np.abs(data["high"] - data["close"].shift())
    low_close = np.abs(data["low"] - data["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(period).mean()
    return atr


def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average (SMA)."""
    return data.rolling(window=period).mean()


def identify_supply_demand_zones(data: pd.DataFrame) -> List[Tuple[str, float]]:
    """Identify supply and demand zones."""
    zones = []
    for i in range(2, len(data) - 2):
        # Check for demand zones
        if (
            data["low"].iloc[i] < data["low"].iloc[i - 1]
            and data["low"].iloc[i] < data["low"].iloc[i + 1]
        ):
            zones.append(("demand", data["low"].iloc[i]))

        # Check for supply zones
        if (
            data["high"].iloc[i] > data["high"].iloc[i - 1]
            and data["high"].iloc[i] > data["high"].iloc[i + 1]
        ):
            zones.append(("supply", data["high"].iloc[i]))

    return zones


def check_candlestick_pattern(data: pd.DataFrame, i: int) -> str:
    """Check for key candlestick patterns."""
    if (
        data["close"].iloc[i] > data["open"].iloc[i]
        and data["close"].iloc[i - 1] < data["open"].iloc[i - 1]
    ):
        if (
            data["close"].iloc[i] > data["open"].iloc[i - 1]
            and data["open"].iloc[i] < data["close"].iloc[i - 1]
        ):
            return "BullishEngulfing"
    if (
        data["close"].iloc[i] < data["open"].iloc[i]
        and data["close"].iloc[i - 1] > data["open"].iloc[i - 1]
    ):
        if (
            data["close"].iloc[i] < data["open"].iloc[i - 1]
            and data["open"].iloc[i] > data["close"].iloc[i - 1]
        ):
            return "BearishEngulfing"
    if (
        abs(data["close"].iloc[i] - data["open"].iloc[i])
        < (data["high"].iloc[i] - data["low"].iloc[i]) * 0.2
    ):
        return "PinBar"
    return ""


def simulate_trade(entry_price: float, exit_price: float, side: str) -> float:
    """Simulate a trade and return the profit/loss percentage."""
    if side == "BUY":
        return (exit_price - entry_price) / entry_price
    else:
        return (entry_price - exit_price) / entry_price


def backtest(data: pd.DataFrame) -> Tuple[float, int, int, List[float]]:
    """Backtest the simplified strategy with key patterns."""
    data["ATR"] = calculate_atr(data, ATR_PERIOD)
    data["SMA"] = calculate_sma(data["close"], SMA_PERIOD)

    supply_demand_zones = identify_supply_demand_zones(data)

    trades = []
    total_profit = 0
    winning_trades = 0
    losing_trades = 0

    for zone_type, price_level in supply_demand_zones:
        for i in range(len(data)):
            pattern = check_candlestick_pattern(data, i)
            if pattern in CANDLESTICK_PATTERNS:
                if zone_type == "demand" and data["low"].iloc[i] <= price_level:
                    if pattern == "BullishEngulfing" or pattern == "PinBar":
                        entry_price = data["close"].iloc[i]
                        stop_loss = (
                            entry_price - STOP_LOSS_MULTIPLIER * data["ATR"].iloc[i]
                        )
                        take_profit = (
                            entry_price + TAKE_PROFIT_MULTIPLIER * data["ATR"].iloc[i]
                        )

                        for j in range(i + 1, len(data)):
                            if data["high"].iloc[j] >= take_profit:
                                profit = simulate_trade(entry_price, take_profit, "BUY")
                                total_profit += profit
                                trades.append(profit)
                                winning_trades += 1
                                break
                            elif data["low"].iloc[j] <= stop_loss:
                                loss = simulate_trade(entry_price, stop_loss, "BUY")
                                total_profit += loss
                                trades.append(loss)
                                losing_trades += 1
                                break

                elif zone_type == "supply" and data["high"].iloc[i] >= price_level:
                    if pattern == "BearishEngulfing" or pattern == "PinBar":
                        entry_price = data["close"].iloc[i]
                        stop_loss = (
                            entry_price + STOP_LOSS_MULTIPLIER * data["ATR"].iloc[i]
                        )
                        take_profit = (
                            entry_price - TAKE_PROFIT_MULTIPLIER * data["ATR"].iloc[i]
                        )

                        for j in range(i + 1, len(data)):
                            if data["low"].iloc[j] <= take_profit:
                                profit = simulate_trade(
                                    entry_price, take_profit, "SELL"
                                )
                                total_profit += profit
                                trades.append(profit)
                                winning_trades += 1
                                break
                            elif data["high"].iloc[j] >= stop_loss:
                                loss = simulate_trade(entry_price, stop_loss, "SELL")
                                total_profit += loss
                                trades.append(loss)
                                losing_trades += 1
                                break

    return total_profit, winning_trades, losing_trades, trades


def run_backtest(start_year: int = 2023, end_year: int = 2024):
    """Run the backtest and print results."""
    print(f"Starting backtest for {SYMBOL} from {start_year} to {end_year}")

    # Define the start and end dates based on the years provided
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-01-01"

    # Load historical data using Binance API
    data = load_historical_data(SYMBOL, "1d", start_date, end_date)

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

    # Calculate total return
    total_return = (1 + total_profit) * INITIAL_INVESTMENT - INITIAL_INVESTMENT
    print(f"Total return on ${INITIAL_INVESTMENT} investment: ${total_return:.2f}")


if __name__ == "__main__":
    # You can specify the start and end years when running the backtest
    run_backtest(start_year=2022, end_year=2023)
