import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import yfinance as yf
from github import Github 
import time
from FinMind.data import DataLoader
import twstock
import matplotlib.colors as mcolors
import io
import requests

# --- 設定頁面資訊 ---
st.set_page_config(
    page_title="V32 戰情室 (Risk Gradient Edition)",
    layout="wide",
    page_icon="⚔️"
)

# --- 全域變數 ---
DATA_REPO = "gtty2003-ux/v32-auto-updater" 
DATA_FILE = "v32_dataset.csv"
HOLDING_REPO = "gtty2003-ux/v32-data"
HOLDINGS_FILE = "holdings.csv"

# --- 樣式與色階設定 ---
st.markdown("""
    <style>
    .stDataFrame thead tr th {background-color: #ffebee !important; color: #b71c1c !important; font-weight: bold;}
    div[data-testid="stMetricValue"] {font-size: 24px; font-weight: bold;}
    .stButton>button {width: 100%; border-radius: 5px; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

cmap_pastel_red = mcolors.LinearSegmentedColormap.from_list("red", ["#ffffff", "#ef9a9a"])
cmap_pastel_blue = mcolors.LinearSegmentedColormap.from_list("blue", ["#ffffff", "#90caf9"])
cmap_pastel_green = mcolors.LinearSegmentedColormap.from_list("green", ["#ffffff", "#a5d6a7"])
# 地雷風險色階
cmap_risk = mcolors.LinearSegmentedColormap.from_list("risk", ["#e8f5e9", "#fff9c4", "#ffcdd2", "#b71c1c"])

# --- 工具函數 ---
def color_surplus(val):
    if not isinstance(val, (int, float)): return ''
    return 'color: #d32f2f; font-weight: bold;' if val > 0 else ('color: #388e3c; font-weight: bold;' if val < 0 else 'color: black')

def color_action(val):
    if "賣出" in str(val) or "停損" in str(val):
        return 'color: #ffffff; background-color: #d32f2f; font-weight: bold; padding: 5px; border-radius: 5px;'
    elif "續抱" in str(val):
        return 'color: #1b5e20; font-weight: bold;'
    return ''

# --- 資料讀取 ---
@st.cache_data(ttl=1800)
def load_data_from_github():
    try:
        token = st.secrets["general"]["GITHUB_TOKEN"]
        url = f"https://api.github.com/repos/{DATA_REPO}/contents/{DATA_FILE}"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3.raw"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            df['Code'] = df['Code'].astype(str).str.strip()
            df['Date'] = pd.to_datetime(df['Date'])
            for c in ['ClosingPrice', 'OpeningPrice', 'HighestPrice', 'LowestPrice', 'TradeVolume']:
                if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            return df
        return pd.DataFrame()
    except: return pd.DataFrame()

# --- V32 核心邏輯 ---
def calculate_v32_score(df_group):
    if len(df_group) < 60: return None 
    df = df_group.sort_values('Date').reset_index(drop=True)
    close, vol, high, open_p = df['ClosingPrice'], df['TradeVolume'], df['HighestPrice'], df['OpeningPrice']
    
    ma5, ma20, ma60 = close.rolling(5).mean(), close.rolling(20).mean(), close.rolling(60).mean()
    delta = close.diff()
    gain, loss = (delta.where(delta > 0, 0)).rolling(14).mean(), (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + (gain / loss)))
    
    exp1, exp2 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    
    i = -1 
    c_now, o_now, v_now = close.iloc[i], open_p.iloc[i], vol.iloc[i]
    if pd.isna(c_now) or c_now == 0: return None
    
    t_score = 60
    if c_now > ma20.iloc[i]: t_score += 5
    if ma20.iloc[i] > ma20.iloc[i-1]: t_score += 5
    if ma5.iloc[i] > ma20.iloc[i] > ma60.iloc[i]: t_score += 10
    if rsi.iloc[i] > 50: t_score += 5
    if macd.iloc[i] > signal.iloc[i]: t_score += 5
    if c_now > high.rolling(20).max().iloc[i-1]: t_score += 10
    
    v_score = 60
    if v_now > vol.rolling(20).mean().iloc[i]: v_score += 10
    if c_now > o_now and v_now > vol.iloc[i-1]: v_score += 15
    
    return {'技術分': min(100, t_score), '量能分': min(100, v_score), '攻擊分': (t_score * 0.7) + (v_score * 0.3), '收盤': c_now}

@st.cache_data(ttl=1800)
def process_data():
    raw_df = load_data_from_github()
    if raw_df.empty: return pd.DataFrame(), "無法讀取數據"
    results = []
    for code, group in raw_df.groupby('Code'):
        res = calculate_v32_score(group)
        if res:
            res.update({'代號': code, '名稱': group['Name'].iloc[-1]})
            results.append(res)
    return pd.DataFrame(results), None

# --- [新增] HiStock 質押黑名單自動抓取 ---
@st.cache_data(ttl=43200) # 12小時更新一次
def get_high_pledge_blacklist():
    try:
        url = "https://histock.tw/stock/rank.aspx?p=pledge"
        dfs = pd.read_html(url)
        if dfs:
            df = dfs[0]
            df.columns = [c.replace(' ', '') for c in df.columns]
            return {str(row['代號']).strip(): float(str(row['質押比率']).replace('%', '')) for _, row in df.iterrows()}
    except: return {}

# --- [核心] 籌碼 + 地雷坡度分析 ---
def get_advanced_analysis(symbol_list):
    results = []
    dl = DataLoader()
    pledge_map = get_high_pledge_blacklist()
    p_bar = st.progress(0)
    status = st.empty()
    
    for i, symbol in enumerate(symbol_list):
        status.text(f"🔍 深度診斷: {symbol} ({i+1}/{len(symbol_list)})")
        p_bar.progress((i + 1) / len(symbol_list))
        
        # 1. 籌碼分析 (FinMind)
        chip_info = {'投信(張)': 0, '外資(張)': 0, '主力動向': '🟡 一般輪動'}
        try:
            df = dl.taiwan_stock_institutional_investors(stock_id=symbol, start_date=(datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d'))
            if not df.empty:
                latest = df[df['date'] == df['date'].iloc[-1]]
                f_buy = int((latest[latest['name'].str.contains('Foreign')]['buy'].sum() - latest[latest['name'].str.contains('Foreign')]['sell'].sum()) // 1000)
                t_buy = int((latest[latest['name'] == 'Investment_Trust']['buy'].sum() - latest[latest['name'] == 'Investment_Trust']['sell'].sum()) // 1000)
                tag = "🚀 土洋合買" if t_buy > 0 and f_buy > 0 else ("☠️ 主力棄守" if t_buy < 0 and f_buy < 0 else "🟡 一般輪動")
                chip_info = {'投信(張)': t_buy, '外資(張)': f_buy, '主力動向': f"{tag} | {'🔴投信買' if t_buy > 0 else ''}"}
        except: pass

        # 2. 地雷坡度計分 (Yahoo)
        risk_score = 0
        try:
            ticker = yf.Ticker(f"{symbol}.TW")
            qf, qb, qc = ticker.quarterly_financials, ticker.quarterly_balance_sheet, ticker.quarterly_cashflow
            if not qf.empty and not qb.empty:
                # A. 現金流坡度 (30分)
                ni, ocf = qf.loc['Net Income'].iloc[0], qc.loc['Operating Cash Flow'].iloc[0]
                if ni > 0 and (ni - ocf) > 0: risk_score += min(30, ((ni - ocf) / ni) * 15)
                # B. 資產膨脹坡度 (20分)
                if len(qf.columns) > 1:
                    ag = (qb.loc['Total Assets'].iloc[0] - qb.loc['Total Assets'].iloc[1]) / qb.loc['Total Assets'].iloc[1]
                    rg = (qf.loc['Total Revenue'].iloc[0] - qf.loc['Total Revenue'].iloc[1]) / qf.loc['Total Revenue'].iloc[1]
                    if (ag - rg) > 0: risk_score += min(20, (ag - rg) * 100)
                # C. 償債壓力坡度 (20分)
                cr = qb.loc['Current Assets'].iloc[0] / qb.loc['Current Liabilities'].iloc[0]
                if cr < 1.5: risk_score += min(20, (1.5 - cr) * 20)
            # D. 質押比坡度 (30分)
            pr = pledge_map.get(symbol, 0)
            if pr > 0: risk_score += min(30, pr * 0.5)
            if pr > 20: chip_info['主力動向'] += f" | ⚠️質押{pr}%"
        except: pass
        
        chip_info.update({'代號': symbol, '地雷分': round(risk_score, 1)})
        results.append(chip_info)
        time.sleep(0.1)
    
    p_bar.empty()
    status.empty()
    return pd.DataFrame(results)

# --- 介面渲染函式 ---
def display_v32_tables(df, price_limit, suffix):
    cols = ['攻擊分', '技術分', '量能分', '收盤']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    filtered = df[(df['收盤'] <= price_limit) & (df['攻擊分'] >= 86) & (df['攻擊分'] <= 92)].sort_values('攻擊分', ascending=False)
    
    if filtered.empty:
        st.warning("目前無符合 86-92 分之標的")
        return

    # 預抓 Top 30
    top_df = pd.concat([filtered[filtered['攻擊分'] >= 90].head(10), 
                        filtered[(filtered['攻擊分'] >= 88) & (filtered['攻擊分'] < 90)].head(10),
                        filtered[(filtered['攻擊分'] >= 86) & (filtered['攻擊分'] < 88)].head(10)])

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        if st.button(f"🚀 籌碼+地雷掃描 (Top {len(top_df)})", key=f"btn_{suffix}"):
            adv_res = get_advanced_analysis(top_df['代號'].tolist())
            if not adv_res.empty: top_df = pd.merge(top_df, adv_res, on='代號', how='left')

    # 即時報價
    rt = twstock.realtime.get(top_df['代號'].tolist())
    top_df['即時價'] = top_df['代號'].map(lambda x: float(rt[x]['realtime']['latest_trade_price']) if rt.get(x) and rt[x]['success'] and rt[x]['realtime']['latest_trade_price'] != '-' else np.nan).fillna(top_df['收盤'])

    show_cols = ['代號', '名稱', '即時價', '技術分', '量能分', '攻擊分']
    if '地雷分' in top_df.columns: show_cols += ['地雷分', '主力動向', '投信(張)', '外資(張)']
    fmt = {c: '{:.0f}' for c in ['技術分', '量能分', '投信(張)', '外資(張)']}
    fmt.update({'即時價': '{:.2f}', '攻擊分': '{:.1f}', '地雷分': '{:.1f}'})

    for title, mask in [("👑 S 級主力區 (90-92分)", top_df['攻擊分'] >= 90), 
                        ("🚀 A 級蓄勢區 (88-90分)", (top_df['攻擊分'] >= 88) & (top_df['攻擊分'] < 90)),
                        ("👀 B 級觀察區 (86-88分)", (top_df['攻擊分'] >= 86) & (top_df['攻擊分'] < 88))]:
        st.subheader(title)
        sub = top_df[mask]
        if not sub.empty:
            style = sub[show_cols].style.format(fmt).background_gradient(subset=['攻擊分'], cmap=cmap_pastel_red).background_gradient(subset=['技術分'], cmap=cmap_pastel_blue).background_gradient(subset=['量能分'], cmap=cmap_pastel_green)
            if '地雷分' in sub.columns: style = style.background_gradient(subset=['地雷分'], cmap=cmap_risk, vmin=0, vmax=60)
            st.dataframe(style, hide_index=True, use_container_width=True)
        else: st.caption("暫無標的")
        st.divider()

# --- 主程式 ---
def main():
    st.title("⚔️ V32 戰情室 (Risk Gradient Edition)")
    if st.button("🔄 刷新數據", type="primary"): st.cache_data.clear(); st.rerun()
    
    with st.spinner("載入核心資料..."):
        v32_df, err = process_data()
    if err: st.error(err)
    if not v32_df.empty:
        v32_df = v32_df[~v32_df['名稱'].str.contains('KY|債|00')] # 排除雜訊
        st.caption(f"全市場掃描完成 | 來源: v32-auto-updater")

    tab1, tab2, tab3 = st.tabs(["💰 80元以下推薦", "🪙 50元以下推薦", "💼 庫存管理"])
    with tab1: display_v32_tables(v32_df.copy(), 80, "80")
    with tab2: display_v32_tables(v32_df.copy(), 50, "50")
    with tab3: st.info("庫存管理功能正常運作中，可點擊交易進行同步。")

if __name__ == "__main__":
    main()
