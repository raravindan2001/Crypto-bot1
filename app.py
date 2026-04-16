# ============================================================
# CoinDCX Trading Bot — Flask Backend Server
# INSTALL: pip install flask flask-cors requests pandas numpy
# RUN: python app.py
# ============================================================

from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import hashlib
import hmac
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Allow Lovable frontend to call this server

# ============================================================
# SECTION 1: YOUR API CREDENTIALS
# Get these from: https://coindcx.com/settings/api
# NEVER share these with anyone
# ============================================================
API_KEY    = "1da1a3fbfc0e8fdf85a2a311ac83e1acae3119d98a7c7fb9"
API_SECRET = "581c98739c4288b93df8b846ef6263950e23c91f1a47a38d20a27b860d8a4414"

# ============================================================
# SECTION 2: BOT STRATEGY SETTINGS (Easy to modify)
# ============================================================
BOT_CONFIG = {
    "pair":            "BTCINR",   # Trading pair — change to ETHINR, XRPINR, etc.
    "quantity":        0.001,      # Amount to trade per order
    "rsi_period":      14,         # RSI look-back period (standard is 14)
    "rsi_buy":         35,         # Buy when RSI drops below this value
    "rsi_sell":        65,         # Sell when RSI rises above this value
    "ma_fast":         9,          # Fast moving average period
    "ma_slow":         21,         # Slow moving average period
    "scalp_buy_pct":  -0.8,        # Scalp: buy if price drops this % from reference
    "scalp_sell_pct":  0.8,        # Scalp: sell if price rises this % from reference
    "strategy":        "rsi",      # Active strategy: "rsi", "ma_crossover", or "scalp"
    "check_interval":  60,         # How often bot checks market (seconds)
    "enabled":         False,      # Master on/off switch — set True to go live
}

# In-memory bot state
bot_state = {
    "running":        False,
    "reference_price": None,
    "last_signal":    "none",
    "last_check":     None,
    "trade_count":    0,
    "log":            [],
}

# ============================================================
# SECTION 3: HELPER FUNCTIONS
# ============================================================
def sign(body: dict) -> str:
    """Generate HMAC-SHA256 signature required by CoinDCX"""
    body_str = json.dumps(body, separators=(',', ':'))
    return hmac.new(
        API_SECRET.encode('utf-8'),
        body_str.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def dcx_headers(body: dict) -> dict:
    """Build authenticated request headers"""
    return {
        "Content-Type": "application/json",
        "X-AUTH-APIKEY": API_KEY,
        "X-AUTH-SIGNATURE": sign(body)
    }

def log_event(msg: str, level: str = "info"):
    """Add event to in-memory log (keeps last 100 entries)"""
    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "msg": msg,
        "level": level
    }
    bot_state["log"].insert(0, entry)
    if len(bot_state["log"]) > 100:
        bot_state["log"].pop()
    print(f"[{entry['time']}] {msg}")

# ============================================================
# SECTION 4: MARKET DATA FUNCTIONS
# ============================================================
def fetch_candles(pair: str, limit: int = 100) -> list:
    """Fetch recent OHLCV candles for a trading pair"""
    try:
        url = "https://public.coindcx.com/market_data/candles"
        params = {
            "pair":       f"I-{pair}",
            "resolution": "5",         # 5-minute candles — change to 1, 15, 60, etc.
            "limit":      limit
        }
        res = requests.get(url, params=params, timeout=10)
        return res.json()
    except Exception as e:
        log_event(f"Candle fetch error: {e}", "error")
        return []

def get_live_price(pair: str) -> float | None:
    """Get the latest trade price for a pair"""
    try:
        url = "https://public.coindcx.com/market_data/trade_history"
        res = requests.get(url, params={"pair": f"I-{pair}", "limit": 1}, timeout=10)
        data = res.json()
        return float(data[0]['p']) if data else None
    except Exception as e:
        log_event(f"Price fetch error: {e}", "error")
        return None

# ============================================================
# SECTION 5: INDICATOR CALCULATIONS
# ============================================================
def calculate_rsi(closes: list, period: int = 14) -> float:
    """
    Calculate RSI (Relative Strength Index)
    RSI > 70 = overbought (consider selling)
    RSI < 30 = oversold (consider buying)
    """
    if len(closes) < period + 1:
        return 50.0  # Neutral if not enough data
    closes = np.array(closes, dtype=float)
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

def calculate_ma(closes: list, period: int) -> float | None:
    """Calculate simple moving average"""
    if len(closes) < period:
        return None
    return round(np.mean(closes[-period:]), 2)

# ============================================================
# SECTION 6: TRADING STRATEGY ENGINE
# ============================================================
def run_strategy(pair: str) -> dict:
    """
    Evaluate the active strategy and return a signal.
    Returns: { signal: "buy"/"sell"/"hold", reason: str, rsi: float, ... }
    """
    candles = fetch_candles(pair, limit=100)
    if not candles:
        return {"signal": "hold", "reason": "No candle data available"}

    closes = [float(c['close']) for c in candles if 'close' in c]
    current_price = closes[-1] if closes else None

    if not current_price:
        return {"signal": "hold", "reason": "Cannot read price"}

    rsi       = calculate_rsi(closes, BOT_CONFIG["rsi_period"])
    ma_fast   = calculate_ma(closes, BOT_CONFIG["ma_fast"])
    ma_slow   = calculate_ma(closes, BOT_CONFIG["ma_slow"])
    strategy  = BOT_CONFIG["strategy"]
    result    = {
        "signal":        "hold",
        "reason":        "",
        "price":         current_price,
        "rsi":           rsi,
        "ma_fast":       ma_fast,
        "ma_slow":       ma_slow,
        "strategy_used": strategy,
    }

    # --- RSI Strategy ---
    if strategy == "rsi":
        if rsi < BOT_CONFIG["rsi_buy"]:
            result["signal"] = "buy"
            result["reason"] = f"RSI {rsi} is below buy threshold {BOT_CONFIG['rsi_buy']}"
        elif rsi > BOT_CONFIG["rsi_sell"]:
            result["signal"] = "sell"
            result["reason"] = f"RSI {rsi} is above sell threshold {BOT_CONFIG['rsi_sell']}"
        else:
            result["reason"] = f"RSI {rsi} is neutral (between {BOT_CONFIG['rsi_buy']} and {BOT_CONFIG['rsi_sell']})"

    # --- Moving Average Crossover Strategy ---
    elif strategy == "ma_crossover":
        if ma_fast and ma_slow:
            if ma_fast > ma_slow:
                result["signal"] = "buy"
                result["reason"] = f"Fast MA ({ma_fast}) crossed above Slow MA ({ma_slow})"
            elif ma_fast < ma_slow:
                result["signal"] = "sell"
                result["reason"] = f"Fast MA ({ma_fast}) crossed below Slow MA ({ma_slow})"
            else:
                result["reason"] = "MAs are equal — no crossover signal"
        else:
            result["reason"] = "Not enough data for MA crossover"

    # --- Scalping Strategy ---
    elif strategy == "scalp":
        ref = bot_state.get("reference_price")
        if ref is None:
            bot_state["reference_price"] = current_price
            result["reason"] = f"Reference price set to {current_price}"
        else:
            change_pct = ((current_price - ref) / ref) * 100
            result["change_pct"] = round(change_pct, 3)
            if change_pct <= BOT_CONFIG["scalp_buy_pct"]:
                result["signal"] = "buy"
                result["reason"] = f"Price dropped {change_pct:.2f}% from reference"
                bot_state["reference_price"] = current_price
            elif change_pct >= BOT_CONFIG["scalp_sell_pct"]:
                result["signal"] = "sell"
                result["reason"] = f"Price rose {change_pct:.2f}% from reference"
                bot_state["reference_price"] = current_price
            else:
                result["reason"] = f"Price change {change_pct:.2f}% — within threshold, holding"

    return result

# ============================================================
# SECTION 7: ORDER EXECUTION
# ============================================================
def execute_order(side: str, price: float, quantity: float, pair: str) -> dict:
    """Place a real limit order on CoinDCX"""
    url  = "https://api.coindcx.com/exchange/v1/orders/create"
    body = {
        "side":           side,
        "order_type":     "limit_order",
        "market":         pair,
        "price_per_unit": price,
        "total_quantity": quantity,
        "timestamp":      int(time.time() * 1000)
    }
    try:
        res  = requests.post(url, json=body, headers=dcx_headers(body), timeout=10)
        data = res.json()
        log_event(f"Order placed: {side} {quantity} {pair} @ {price}", "trade")
        bot_state["trade_count"] += 1
        return data
    except Exception as e:
        log_event(f"Order error: {e}", "error")
        return {"error": str(e)}

# ============================================================
# SECTION 8: API ENDPOINTS (Called by Lovable Frontend)
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    """Check if backend is running"""
    return jsonify({"status": "ok", "time": datetime.now().isoformat()})


@app.route("/balance", methods=["GET"])
def get_balance():
    """Fetch user's account balances"""
    body = {"timestamp": int(time.time() * 1000)}
    try:
        res  = requests.post(
            "https://api.coindcx.com/exchange/v1/users/balances",
            json=body, headers=dcx_headers(body), timeout=10
        )
        data = res.json()
        # Return only non-zero balances
        if isinstance(data, list):
            filtered = [b for b in data if float(b.get("balance", 0)) > 0]
            return jsonify({"balances": filtered})
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/price", methods=["GET"])
def get_price():
    """Get current market price for a pair"""
    pair  = request.args.get("pair", BOT_CONFIG["pair"])
    price = get_live_price(pair)
    return jsonify({"pair": pair, "price": price, "time": datetime.now().isoformat()})


@app.route("/candles", methods=["GET"])
def get_candles():
    """Get OHLCV candle data for charting"""
    pair   = request.args.get("pair",  BOT_CONFIG["pair"])
    limit  = int(request.args.get("limit", 50))
    candles = fetch_candles(pair, limit)
    return jsonify({"pair": pair, "candles": candles})


@app.route("/indicators", methods=["GET"])
def get_indicators():
    """Calculate and return current indicator values"""
    pair    = request.args.get("pair", BOT_CONFIG["pair"])
    candles = fetch_candles(pair, 100)
    closes  = [float(c['close']) for c in candles if 'close' in c]
    return jsonify({
        "pair":    pair,
        "rsi":     calculate_rsi(closes, BOT_CONFIG["rsi_period"]),
        "ma_fast": calculate_ma(closes, BOT_CONFIG["ma_fast"]),
        "ma_slow": calculate_ma(closes, BOT_CONFIG["ma_slow"]),
        "price":   closes[-1] if closes else None,
    })


@app.route("/signal", methods=["GET"])
def get_signal():
    """Run strategy and return current signal (no order placed)"""
    pair   = request.args.get("pair", BOT_CONFIG["pair"])
    result = run_strategy(pair)
    return jsonify(result)


@app.route("/order", methods=["POST"])
def place_order():
    """
    Place an order manually from the dashboard.
    Body: { "side": "buy"/"sell", "pair": "BTCINR", "qty": 0.001, "price": 5000000 }
    """
    if not BOT_CONFIG["enabled"]:
        return jsonify({"error": "Bot is disabled. Enable it in config to trade."}), 403
    data   = request.json
    result = execute_order(
        side=data.get("side", "buy"),
        price=data.get("price"),
        quantity=data.get("qty", BOT_CONFIG["quantity"]),
        pair=data.get("pair", BOT_CONFIG["pair"])
    )
    return jsonify(result)


@app.route("/orders", methods=["GET"])
def get_open_orders():
    """Get all open/active orders"""
    pair = request.args.get("pair", BOT_CONFIG["pair"])
    body = {"market": pair, "timestamp": int(time.time() * 1000)}
    try:
        res  = requests.post(
            "https://api.coindcx.com/exchange/v1/orders/active_orders",
            json=body, headers=dcx_headers(body), timeout=10
        )
        return jsonify({"orders": res.json()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/orders/history", methods=["GET"])
def get_order_history():
    """Get recent order history"""
    pair  = request.args.get("pair", BOT_CONFIG["pair"])
    limit = int(request.args.get("limit", 20))
    body  = {
        "market":    pair,
        "limit":     limit,
        "timestamp": int(time.time() * 1000)
    }
    try:
        res  = requests.post(
            "https://api.coindcx.com/exchange/v1/orders/trade_history",
            json=body, headers=dcx_headers(body), timeout=10
        )
        return jsonify({"history": res.json()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/bot/status", methods=["GET"])
def bot_status():
    """Return current bot state and recent log"""
    return jsonify({
        "running":      bot_state["running"],
        "config":       BOT_CONFIG,
        "last_signal":  bot_state["last_signal"],
        "last_check":   bot_state["last_check"],
        "trade_count":  bot_state["trade_count"],
        "log":          bot_state["log"][:20],  # Last 20 log entries
    })


@app.route("/bot/config", methods=["POST"])
def update_config():
    """
    Update bot configuration from the dashboard.
    Body: any fields from BOT_CONFIG
    """
    data = request.json
    allowed = ["pair", "quantity", "rsi_buy", "rsi_sell", "ma_fast", "ma_slow",
               "scalp_buy_pct", "scalp_sell_pct", "strategy", "check_interval", "enabled"]
    for key in allowed:
        if key in data:
            BOT_CONFIG[key] = data[key]
    log_event(f"Config updated: {data}")
    return jsonify({"success": True, "config": BOT_CONFIG})


@app.route("/bot/run_once", methods=["POST"])
def run_once():
    """
    Manually trigger one strategy check.
    If bot is enabled AND signal is buy/sell, it places a real order.
    """
    pair   = BOT_CONFIG["pair"]
    result = run_strategy(pair)
    signal = result.get("signal", "hold")
    bot_state["last_signal"] = signal
    bot_state["last_check"]  = datetime.now().isoformat()
    log_event(f"Strategy check: {signal} — {result.get('reason', '')}")

    order_result = None
    if BOT_CONFIG["enabled"] and signal in ["buy", "sell"]:
        price = result["price"]
        # Add small buffer: buy slightly below, sell slightly above
        limit_price = price * 0.999 if signal == "buy" else price * 1.001
        order_result = execute_order(signal, round(limit_price, 2), BOT_CONFIG["quantity"], pair)

    return jsonify({"signal_result": result, "order_result": order_result})


# ============================================================
# SECTION 9: START SERVER
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  CoinDCX Trading Bot Backend")
    print("  Running at: http://localhost:5000")
    print("  Bot enabled:", BOT_CONFIG["enabled"])
    print("=" * 50)
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, port=port, host="0.0.0.0")
