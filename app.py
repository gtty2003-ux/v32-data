import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import yfinance as yf
from github import Github 
import time
from FinMind.data import DataLoader
# 移除 twstock，改用 yfinance 確保連線穩定
import matplotlib.colors as mcolors
import io
import requests

# --- 設定頁面資訊 ---
st.set_page_config(page_title="V32 戰情室 (Risk Gradient Edition)", layout="wide", page_icon="⚔️")

# --- 全域變數 ---
DATA_REPO = "gtty2003-ux/v32-auto-updater" 
DATA_FILE = "v32_dataset.csv"
HOLDING_REPO = "gtty2003-ux/v32-data"
HOLDINGS_FILE = "holdings.csv"

# --- 樣式與色階 ---
st.markdown("""<style>.stDataFrame thead tr th {background-color: #ffebee !important; color: #b71c1c !important; font-weight: bold;}</style>""", unsafe_allow_html=True)
cmap_risk = mcolors.LinearSegmentedColormap.from_list("risk", ["#e8f5e9", "#fff9c4", "#ffcdd2", "#b71c1c"])

# --- [修正] 報價獲取函式：捨棄 twstock 改用 yfinance ---
@st.cache_data(ttl=60)
def get_safe_realtime_quotes(code_list):
    if not code_list: return {}
    # 統一轉換為 Yahoo 格式 (e.g., 2330.TW)
    yf_codes = [f"{c}.TW" for c in code_list]
    realtime_data = {}
    try:
        data = yf.download(yf_codes, period="1d", interval="1m", progress=False)
        for code in code_list:
            ticker = f"{code}.TW"
            if ticker in data.columns.levels[1]:
                latest_price = data['Close'][ticker].iloc[-1]
                realtime_data[code] = latest_price
    except Exception as e:
        st.warning(f"報價服務暫時無法取得: {e}")
    return realtime_data

# --- [核心] 籌碼 + 地雷坡度分析 (依據您的風險模型) ---
def get_advanced_analysis(symbol_list):
    results = []
    dl = DataLoader()
    p_bar = st.progress(0)
    for i, symbol in enumerate(symbol_list):
        p_bar.progress((i + 1) / len(symbol_list))
        risk_score = 0
        try:
            # 坡度計算邏輯整合
            ticker = yf.Ticker(f"{symbol}.TW")
            qf = ticker.quarterly_financials
            qc = ticker.quarterly_cashflow
            qb = ticker.quarterly_balance_sheet
            if not qf.empty and not qc.empty:
                # 1. 現金流坡度 (30分)
                ni, ocf = qf.loc['Net Income'].iloc[0], qc.loc['Operating Cash Flow'].iloc[0]
                if ni > 0: risk_score += min(30, ((ni - ocf) / ni) * 15)
                # 2. 償債壓力坡度 (20分)
                cr = qb.loc['Current Assets'].iloc[0] / qb.loc['Current Liabilities'].iloc[0]
                if cr < 1.5: risk_score += min(20, (1.5 - cr) * 20)
        except: pass
        results.append({'代號': symbol, '地雷分': round(risk_score, 1), '主力動向': '分析中...'})
    p_bar.empty()
    return pd.DataFrame(results)

# --- 介面渲染 ---
def display_v32_tables(df, price_limit, suffix):
    filtered = df[(df['收盤'] <= price_limit) & (df['攻擊分'] >= 86) & (df['攻擊分'] <= 92)].sort_values('攻擊分', ascending=False)
    if filtered.empty: return st.warning("目前無符合標的")

    top_df = pd.concat([filtered[filtered['攻擊分'] >= 90].head(10), 
                        filtered[(filtered['攻擊分'] >= 88) & (filtered['攻擊分'] < 90)].head(10),
                        filtered[(filtered['攻擊分'] >= 86) & (filtered['攻擊分'] < 88)].head(10)])

    if st.button(f"🚀 籌碼+地雷掃描 (Top {len(top_df)})", key=f"btn_{suffix}"):
        adv_res = get_advanced_analysis(top_df['代號'].tolist())
        top_df = pd.merge(top_df, adv_res, on='代號', how='left')

    # [修正點] 呼叫 Yahoo 報價避免 SSL 錯誤
    quotes = get_safe_realtime_quotes(top_df['代號'].tolist())
    top_df['即時價'] = top_df['代號'].map(lambda x: quotes.get(x, np.nan)).fillna(top_df['收盤'])

    for title, mask in [("👑 S 級主力區 (90-92分)", top_df['攻擊分'] >= 90), 
                        ("🚀 A 級蓄勢區 (88-90分)", (top_df['攻擊分'] >= 88) & (top_df['攻擊分'] < 90)),
                        ("👀 B 級觀察區 (86-88分)", (top_df['攻擊分'] >= 86) & (top_df['攻擊分'] < 88))]:
        st.subheader(title)
        sub = top_df[mask]
        if not sub.empty:
            style = sub[['代號','名稱','即時價','技術分','量能分','攻擊分'] + (['地雷分'] if '地雷分' in sub.columns else [])].style.format('{:.1f}')
            if '地雷分' in sub.columns: style = style.background_gradient(subset=['地雷分'], cmap=cmap_risk, vmin=0, vmax=60)
            st.dataframe(style, hide_index=True, use_container_width=True)
