# app/dashboard.py
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="RL Trading Lab", layout="wide")
st.title("RL Trading Lab — SPY")
st.caption("Educational prototype. Not investment advice.")

@st.cache_data
def load():
    res = pd.read_parquet("data/results.parquet")
    met = json.load(open("data/metrics.json"))
    return res, met

res, met = load()

# 1. metric cards
cols = st.columns(3)
for col, key, label in zip(cols,
        ["total_return", "sharpe", "max_drawdown"],
        ["Total return", "Sharpe", "Max drawdown"]):
    a, b = met["agent"][key], met["buy_hold"][key]
    col.metric(label, f"{a:.2f}", f"{a - b:+.2f} vs Buy & Hold")

# 2. equity curves
fig = go.Figure()
for name in ["agent", "buy_hold", "ma_cross"]:
    fig.add_scatter(x=res.index, y=res[name], name=name)
fig.update_layout(title="Equity curve (start = 1.0)", height=400)
st.plotly_chart(fig, use_container_width=True)

# 3. price + decisions
chg = res["position"].diff()
buys, sells = res[chg == 1], res[chg == -1]
fig2 = go.Figure()
fig2.add_scatter(x=res.index, y=res["close"], name="SPY")
fig2.add_scatter(x=buys.index, y=buys["close"], mode="markers",
                 marker=dict(symbol="triangle-up", size=10, color="green"),
                 name="Buy")
fig2.add_scatter(x=sells.index, y=sells["close"], mode="markers",
                 marker=dict(symbol="triangle-down", size=10, color="red"),
                 name="Sell")
fig2.update_layout(title="Agent decisions", height=400)
st.plotly_chart(fig2, use_container_width=True)

# 4. latest signal
last = res.iloc[-1]
st.subheader(f"Latest signal ({res.index[-1].date()})")
st.info("HOLD SPY" if last["position"] == 1 else "STAY IN CASH")