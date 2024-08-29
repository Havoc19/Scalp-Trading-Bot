# scalping_bot.py
import os
import time
from typing import List, Tuple
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Configuration
SYMBOL = "BTCUSDT"
LEVERAGE = 10  # Reduced from 50 to 10 for lower risk
INITIAL_INVESTMENT = 10  # USD
STOP_LOSS_PCT = 0.002  # 0.2%
TAKE_PROFIT_PCT = 0.004  # 0.4%
SMA_PERIOD = 200
EMA_PERIOD = 50
MAX_TRADES_PER_CROSSOVER = 5
MAX_DAILY_TRADES = 10
COOLDOWN_PERIOD = 300  # 5 minutes in seconds

# Load API keys from environment variables
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

client = Client(API_KEY, API_SECRET)


def get_historical_klines(symbol: str, interval: str, limit: int = 500) -> List[List]:
    """Fetch recent candlestick data."""
    return client.futures_klines(symbol=symbol, interval=interval, limit=limit)


def calculate_sma(data: List[List], period: int) -> float:
    """Calculate Simple Moving Average."""
    closing_prices = [float(candle[4]) for candle in data[-period:]]
    return np.mean(closing_prices)


def calculate_ema(data: List[List], period: int) -> float:
    """Calculate Exponential Moving Average."""
    closing_prices = np.array([float(candle[4]) for candle in data[-period:]])
    weights = np.exp(np.linspace(-1.0, 0.0, period))
    weights /= weights.sum()
    return np.dot(closing_prices, weights)


def calculate_position_size(symbol: str, investment: float, leverage: int) -> float:
    """Determine position size based on investment and leverage."""
    current_price = float(client.futures_symbol_ticker(symbol=symbol)["price"])
    position_value = investment * leverage
    quantity = position_value / current_price
    return round(quantity, 3)


def check_entry_conditions(data: List[List]) -> Tuple[bool, bool]:
    """Check entry conditions for long and short positions."""
    current_price = float(data[-1][4])
    sma_value = calculate_sma(data, SMA_PERIOD)
    ema_value = calculate_ema(data, EMA_PERIOD)

    long_condition = current_price > sma_value and ema_value > sma_value
    short_condition = current_price < sma_value and ema_value < sma_value

    return long_condition, short_condition


def place_order(symbol: str, side: str, quantity: float) -> dict:
    """Place a market order."""
    try:
        order = client.futures_create_order(
            symbol=symbol, side=side, type="MARKET", quantity=quantity
        )
        return order
    except BinanceAPIException as e:
        print(f"An error occurred: {e}")
        return None


def monitor_position(symbol: str, side: str, entry_price: float, quantity: float):
    """Monitor an open position and close it when conditions are met."""
    if side == "BUY":
        take_profit_price = entry_price * (1 + TAKE_PROFIT_PCT)
        stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
        exit_side = "SELL"
    else:
        take_profit_price = entry_price * (1 - TAKE_PROFIT_PCT)
        stop_loss_price = entry_price * (1 + STOP_LOSS_PCT)
        exit_side = "BUY"

    while True:
        current_price = float(client.futures_symbol_ticker(symbol=symbol)["price"])

        if (side == "BUY" and current_price >= take_profit_price) or (
            side == "SELL" and current_price <= take_profit_price
        ):
            print(f"Closing position at take profit: {current_price}")
            place_order(symbol, exit_side, quantity)
            return "PROFIT"

        if (side == "BUY" and current_price <= stop_loss_price) or (
            side == "SELL" and current_price >= stop_loss_price
        ):
            print(f"Closing position at stop loss: {current_price}")
            place_order(symbol, exit_side, quantity)
            return "LOSS"

        time.sleep(1)


def run_scalping_bot():
    """Main function to run the scalping bot."""
    print(f"Starting scalping bot for {SYMBOL}")

    trades_since_crossover = 0
    daily_trades = 0
    last_trend = None
    last_trade_time = 0

    while True:
        try:
            current_time = time.time()

            # Check if we've reached the daily trade limit
            if current_time - last_trade_time > 86400:  # 24 hours
                daily_trades = 0

            if daily_trades >= MAX_DAILY_TRADES:
                print("Daily trade limit reached. Waiting for next day.")
                time.sleep(3600)  # Wait an hour before checking again
                continue

            # Check if we're in the cooldown period
            if current_time - last_trade_time < COOLDOWN_PERIOD:
                time.sleep(10)
                continue

            klines = get_historical_klines(SYMBOL, "5m")
            long_condition, short_condition = check_entry_conditions(klines)

            if long_condition and (
                last_trend != "LONG"
                or trades_since_crossover < MAX_TRADES_PER_CROSSOVER
            ):
                if last_trend != "LONG":
                    trades_since_crossover = 0
                    last_trend = "LONG"

                quantity = calculate_position_size(SYMBOL, INITIAL_INVESTMENT, LEVERAGE)
                print(f"Opening LONG position: {quantity} {SYMBOL}")
                order = place_order(SYMBOL, "BUY", quantity)

                if order:
                    entry_price = float(order["avgPrice"])
                    result = monitor_position(SYMBOL, "BUY", entry_price, quantity)
                    print(f"Position closed with {result}")
                    trades_since_crossover += 1
                    daily_trades += 1
                    last_trade_time = current_time

            elif short_condition and (
                last_trend != "SHORT"
                or trades_since_crossover < MAX_TRADES_PER_CROSSOVER
            ):
                if last_trend != "SHORT":
                    trades_since_crossover = 0
                    last_trend = "SHORT"

                quantity = calculate_position_size(SYMBOL, INITIAL_INVESTMENT, LEVERAGE)
                print(f"Opening SHORT position: {quantity} {SYMBOL}")
                order = place_order(SYMBOL, "SELL", quantity)

                if order:
                    entry_price = float(order["avgPrice"])
                    result = monitor_position(SYMBOL, "SELL", entry_price, quantity)
                    print(f"Position closed with {result}")
                    trades_since_crossover += 1
                    daily_trades += 1
                    last_trade_time = current_time

            time.sleep(10)  # Wait before checking conditions again

        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(60)  # Wait a minute before retrying


if __name__ == "__main__":
    run_scalping_bot()