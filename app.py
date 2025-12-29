import streamlit as st
from backend import run_full_analysis

st.set_page_config(page_title="AI Stock Analyzer", layout="wide")

st.title("📊 AI-Powered Stock Market Analyzer")
st.caption("Quant + Chart Pattern AI + Backtesting")

# =========================
# INPUT
# =========================
stock = st.text_input("Enter Stock Symbol (e.g. AAPL, MSFT, TSLA)", value="AAPL")

analyze_btn = st.button("🚀 Analyze Stock")

# =========================
# OUTPUT
# =========================
if analyze_btn:
    with st.spinner("Analyzing stock, charts & strategy..."):
        result = run_full_analysis(stock.upper())

    if "error" in result:
        st.error(result["error"])
    else:
        col1, col2, col3 = st.columns(3)

        col1.metric("💰 Price", f"${result['price']}")
        col2.metric("📈 Quant Trend", result["quant_trend"])
        col3.metric("✅ Quant Confidence", f"{result['quant_confidence']}%")

        st.divider()

        st.subheader("📉 Backtest Results")
        bt = result["backtest"]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Strategy Return", f"{bt['strategy_return_%']}%")
        c2.metric("Buy & Hold", f"{bt['buy_hold_return_%']}%")
        c3.metric("Win Rate", f"{bt['win_rate_%']}%")
        c4.metric("Trades", bt["trades"])
        c5.metric("Max Drawdown", f"{bt['max_drawdown_%']}%")

        st.divider()

        st.subheader("🖼 Chart Pattern AI Analysis")

        for img in result["images"]:
            with st.expander(f"📊 {img['image']}"):
                st.write(f"**Pattern:** {img['pattern']}")
                st.write(f"**Final Trend:** {img['final_trend']}")
                st.progress(img["confidence"] / 100)
                st.caption(f"Confidence: {img['confidence']}%")
