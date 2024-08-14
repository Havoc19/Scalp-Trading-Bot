from binance.client import Client
import time
import numpy as np
import os

# Load API keys from environment variables
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
client = Client(api_key, api_secret)

# Trading parameters
initial_investment = 100  # USD
symbol = "BTCUSDT"  # Change to "ETHUSDT" if you want to trade Ethereum
leverage = 5
stop_loss_pct = 0.004  # 0.4%
take_profit_pct = 0.008  # 0.8%
sma_period = 200
ema_period = 50


# Fetch recent candlestick data
def get_ohlc(symbol, interval, limit=500):
    return client.get_klines(symbol=symbol, interval=interval, limit=limit)


# Calculate simple moving average
def sma(data, period):
    prices = [float(x[4]) for x in data[-period:]]  # Use closing prices
    return np.mean(prices)


# Calculate exponential moving average (simplified)
def ema(data, period):
    prices = [float(x[4]) for x in data[-period:]]  # Use closing prices
    return np.mean(prices)  # This is simplified; consider using actual EMA calculation


# Determine position size based on initial investment
def calculate_position_size(symbol, initial_investment, leverage):
    current_price = float(client.get_symbol_ticker(symbol=symbol)["price"])
    position_value = initial_investment * leverage
    quantity = position_value / current_price
    return round(quantity, 6)  # Rounded to 6 decimal places for precision


# Entry Conditions
def long_entry(data):
    price = float(data[-1][4])  # Closing price
    sma_value = sma(data, sma_period)
    ema_value = ema(data, ema_period)
    return price > sma_value and ema_value > sma_value


def short_entry(data):
    price = float(data[-1][4])  # Closing price
    sma_value = sma(data, sma_period)
    ema_value = ema(data, ema_period)
    return price < sma_value and ema_value < sma_value


# Place market order
def place_order(symbol, side, quantity):
    return client.futures_create_order(
        symbol=symbol, side=side, type="MARKET", quantity=quantity
    )


# Scalp trading loop
while True:
    data = get_ohlc(symbol, "1m")
    quantity = calculate_position_size(symbol, initial_investment, leverage)

    if long_entry(data):
        print(f"Buying {quantity} {symbol}")
        order = place_order(symbol, "BUY", quantity)

        buy_price = float(order["fills"][0]["price"])
        target_price = buy_price * (1 + take_profit_pct)
        stop_price = buy_price * (1 - stop_loss_pct)

        # Monitoring position
        while True:
            current_price = float(client.get_symbol_ticker(symbol=symbol)["price"])

            if current_price >= target_price:
                print(f"Selling {quantity} {symbol} for profit")
                place_order(symbol, "SELL", quantity)
                break

            if current_price <= stop_price:
                print(f"Selling {quantity} {symbol} for stop loss")
                place_order(symbol, "SELL", quantity)
                break

            time.sleep(1)

    elif short_entry(data):
        print(f"Selling {quantity} {symbol}")
        order = place_order(symbol, "SELL", quantity)

        sell_price = float(order["fills"][0]["price"])
        target_price = sell_price * (1 - take_profit_pct)
        stop_price = sell_price * (1 + stop_loss_pct)

        # Monitoring position
        while True:
            current_price = float(client.get_symbol_ticker(symbol=symbol)["price"])

            if current_price <= target_price:
                print(f"Buying {quantity} {symbol} to cover short position")
                place_order(symbol, "BUY", quantity)
                break

            if current_price >= stop_price:
                print(f"Buying {quantity} {symbol} for stop loss")
                place_order(symbol, "BUY", quantity)
                break

            time.sleep(1)

    time.sleep(1)