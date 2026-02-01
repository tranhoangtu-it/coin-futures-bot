import streamlit as st
import pandas as pd
import json
import time
from pathlib import Path

# Config
st.set_page_config(
    page_title="PET Trading Center",
    page_icon="🦅",
    layout="wide",
)

# Constants
STATE_FILE = Path("data") / "live_state.json"
GOAL = 1_000_000
START_CAPITAL = 100

def load_data():
    if not STATE_FILE.exists():
        return None
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except:
        return None

# Custom CSS for "Screen" feel
st.markdown("""
<style>
    .stMetric {
        background-color: #0e1117;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #303030;
    }
    .log-box {
        font-family: monospace;
        font-size: 12px;
        color: #00ff00;
        background-color: black;
        padding: 5px;
        border-radius: 4px;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Auto-refresh loop
placeholder = st.empty()

while True:
    data = load_data()
    
    with placeholder.container():
        if not data:
            st.warning("📡 Waiting for Scanner Data...")
            time.sleep(1)
            continue
            
        # Metrics
        bal = data.get("balance", 0)
        start_bal = data.get("start_balance", START_CAPITAL)
        pnl = bal - start_bal
        pnl_pct = (pnl / start_bal * 100) if start_bal > 0 else 0
        positions = data.get("positions", {})
        count_pos = len(positions)
        active = data.get("active", False)
        
        # Header Section
        st.title("🦅 PET Command Center")
        
        # 1. Goal Tracker
        col_g1, col_g2 = st.columns([3, 1])
        with col_g1:
            progress = min(bal / GOAL, 1.0)
            st.progress(progress)
            st.caption(f"Goal Progress: ${bal:.2f} / ${GOAL:,.0f}")
        with col_g2:
            status_icon = "🟢 ONLINE" if active else "🔴 OFFLINE"
            st.markdown(f"### {status_icon}")

        # 2. Key Metrics
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Equity", f"${bal:,.2f}", f"{pnl:+.2f} ({pnl_pct:+.2f}%)")
        k2.metric("Active Positions", count_pos, delta_color="normal")
        k3.metric("Scan Speed", "High (Dual-Layer)")
        k4.metric("Strategy", "Micro-Scalp + Swing")

        st.divider()

        # 3. "Multi-Screen" Monitor Grid
        if not positions:
            st.info("🔭 Scanning Markets... No active trades yet.")
        else:
            st.subheader(f"📺 Active Monitors ({count_pos})")
            
            # Grid Layout (3 columns)
            cols = st.columns(3)
            
            # Sort positions by PnL (simulated or real) or just symbol
            pos_list = list(positions.items())
            
            for i, (sym, pos) in enumerate(pos_list):
                col = cols[i % 3]
                with col:
                    # Parse data
                    side = pos["side"]
                    entry = pos["entry"]
                    qty = pos["qty"]
                    lev = pos["leverage"]
                    
                    # Try to get latest log for this coin
                    logs = data.get("logs", {})
                    last_log = logs.get(sym, "No recent data")
                    
                    # Create card
                    with st.container():
                        st.markdown(f"**{sym}** · {side} {lev}x")
                        st.text(f"Entry: {entry}")
                        st.markdown(f"<div class='log-box'>{last_log}</div>", unsafe_allow_html=True)
                        st.divider()

        # 4. Recent Trades Log
        with st.expander("📜 Transaction History", expanded=False):
            trades = data.get("recent_trades", [])
            if trades:
                df_trades = pd.DataFrame(trades)
                df_trades["time"] = pd.to_datetime(df_trades["timestamp"], unit="s").dt.strftime('%H:%M:%S')
                st.dataframe(
                    df_trades[["time", "symbol", "side", "price", "leverage"]].sort_values("time", ascending=False),
                    hide_index=True,
                    use_container_width=True
                )

    time.sleep(1) # Fast UI refresh
