import os
import base64
import json
import yfinance as yf
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression

# =========================
# CONFIG
# =========================
STOCK_SYMBOL = "AAPL"          # 🔹 CHANGE ONLY THIS
IMAGE_FOLDER = "images"       # folder with chart images

# =========================
# SETUP
# =========================
load_dotenv()
client = OpenAI()

# =========================
# IMAGE ENCODER
# =========================
def encode_image(image_path):
    with open(image_path, "rb") as img:
        b64 = base64.b64encode(img.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

# =========================
# QUANT STOCK ANALYSIS
# =========================
def analyze_stock_trend(symbol, period="6mo"):
    df = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if len(df) < 50:
        return None

    price = df["Close"].iloc[-1].item()

    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["EMA200"] = df["Close"].ewm(span=200).mean()

    recent = df.tail(30)
    X = np.arange(len(recent)).reshape(-1, 1)
    y = recent["Close"].values.reshape(-1, 1)

    model = LinearRegression().fit(X, y)
    slope = float(model.coef_[0][0])

    momentum = recent["Close"].pct_change().mean().item()

    score = 0
    if price > df["EMA20"].iloc[-1]: score += 1
    if df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1]: score += 1
    if df["EMA50"].iloc[-1] > df["EMA200"].iloc[-1]: score += 1
    if slope > 0: score += 1
    if momentum > 0: score += 1

    confidence = (score / 5) * 100

    return {
        "price": round(price, 2),
        "quant_trend": "BULLISH" if score >= 4 else "BEARISH" if score <= 2 else "NEUTRAL",
        "quant_confidence": round(confidence, 2)
    }

# =========================
# GPT IMAGE ANALYSIS
def analyze_chart_image(image_path):
    prompt = """
    Analyze this stock chart image.
    Return ONLY valid JSON:
    {
      "image_trend": "bullish | bearish | neutral",
      "image_confidence": 0-100,
      "pattern": "pattern or none",
      "reason": "short explanation"
    }
    """

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": encode_image(image_path)}
                ]
            }
        ]
    )

    raw = response.output_text or ""

    # ✅ SAFE JSON PARSING
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except Exception:
        # fallback if GPT output is malformed
        return {
            "image_trend": "neutral",
            "image_confidence": 50,
            "pattern": "none",
            "reason": "Unable to parse model output"
        }

# =========================
# SIGNAL FUSION
# =========================
def fuse_signals(quant, image):
    qt = quant["quant_trend"].lower()
    it = image["image_trend"].lower()

    base_conf = 0.6 * quant["quant_confidence"] + 0.4 * image["image_confidence"]

    if qt != it:
        return "NEUTRAL", round(base_conf * 0.5, 2)

    return qt.upper(), round(base_conf, 2)

# =========================
# BACKTESTING ENGINE
# =========================
def backtest_strategy(symbol, period="2y"):
    df = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if len(df) < 200:
        return None

    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["EMA200"] = df["Close"].ewm(span=200).mean()

    equity = 1.0
    peak = 1.0
    position = 0
    entry = 0
    wins = 0
    trades = 0
    max_dd = 0

    for i in range(200, len(df)):
        recent = df.iloc[i-15:i]
        X = np.arange(len(recent)).reshape(-1, 1)
        y = recent["Close"].values.reshape(-1, 1)
        slope = LinearRegression().fit(X, y).coef_[0][0]

        buy = (
            df["EMA20"].iloc[i] > df["EMA50"].iloc[i]
            and slope > -0.0005
        )

        price = df["Close"].iloc[i].item()

        if buy and position == 0:
            entry = price
            position = 1
            trades += 1
            stop_loss = entry * 0.95 
            
        elif position == 1 and (not buy or price <= stop_loss):
            ret = price / entry
            equity *= ret
            if ret > 1.0:
                wins += 1
            position = 0



        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)

    buy_hold = (
        df["Close"].iloc[-1].item() /
        df["Close"].iloc[200].item()
    )


    return {
        "strategy_return_%": float(round((equity - 1) * 100, 2)),
        "buy_hold_return_%": float(round((buy_hold - 1) * 100, 2)),
        "win_rate_%": float(round((wins / trades) * 100, 2)) if trades else 0.0,
        "trades": int(trades),
        "max_drawdown_%": float(round(max_dd * 100, 2))
    }


def run_full_analysis(stock_symbol):
    quant = analyze_stock_trend(stock_symbol)
    if not quant:
        return {"error": "Not enough data"}

    backtest = backtest_strategy(stock_symbol)

    image_results = []
    for img in os.listdir(IMAGE_FOLDER):
        if img.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(IMAGE_FOLDER, img)
            image_signal = analyze_chart_image(img_path)
            trend, conf = fuse_signals(quant, image_signal)

            image_results.append({
                "image": img,
                "pattern": image_signal["pattern"],
                "final_trend": trend,
                "confidence": conf
            })

    return {
        "symbol": stock_symbol,
        "price": quant["price"],
        "quant_trend": quant["quant_trend"],
        "quant_confidence": quant["quant_confidence"],
        "backtest": backtest,
        "images": image_results
    }
