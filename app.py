import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# ==========================================
# 1. 頁面配置與樣式
# ==========================================
st.set_page_config(
    page_title="V33 智能選股系統",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義 CSS (配色方案 #C8E6C9)
st.markdown("""
    <style>
    .stDataFrame { font-size: 14px; }
    /* 強制 Highlight 顏色 */
    .sell-signal { background-color: #FFCDD2 !important; color: black; } /* 紅: 賣出 */
    .hold-run { background-color: #B3E5FC !important; color: black; }    /* 藍: 獲利奔跑 */
    .hold-safe { background-color: #C8E6C9 !important; color: black; }   /* 綠: 續抱 */
    </style>
""", unsafe_allow_html=True)

tw_tz = pytz.timezone('Asia/Taipei')
current_time = datetime.now(tw_tz).strftime('%Y-%m-%d %H:%M:%S')

# ==========================================
# 2. 核心邏輯函式
# ==========================================

def get_market_data():
    """ 生成模擬市場數據 """
    np.random.seed(int(datetime.now().timestamp()))
    data = []
    tickers = [f"{x}" for x in range(1101, 1151)]
    for t in tickers:
        price = np.random.randint(20, 120) 
        tech_score = np.random.randint(40, 99)
        vol_score = np.random.randint(40, 99)
        total_score = (tech_score * 0.7) + (vol_score * 0.3)
        
        data.append({
            "StockID": t,
            "Name": f"模擬股-{t}",
            "Price": price,
            "TechScore": tech_score,
            "VolScore": vol_score,
            "TotalScore": round(total_score, 2),
            "Volume": np.random.randint(1000, 50000)
        })
    return pd.DataFrame(data)

def strategy_v32_selection(df):
    """ V32 選股：價格 < 80 且 Top 20 """
    df_filtered = df[df['Price'] < 80].copy()
    df_top20 = df_filtered.sort_values(by='TotalScore', ascending=False).head(20)
    return df_top20.reset_index(drop=True)

def strategy_v33_inventory_check(inventory_df, current_market_df):
    """
    V33 庫存監控邏輯：
    1. 賣出 A: 跌破持有期間最高價 10% (Trailing Stop)。
    2. 賣出 B: 技術分 < 60。
    3. 續抱: 若突破 80 元，顯示藍色燈號 (利潤奔跑)。
    """
    results = []
    
    # 合併庫存與最新行情
    merged_df = pd.merge(inventory_df, current_market_df[['StockID', 'Price', 'TechScore']], on='StockID', how='left')
    
    for index, row in merged_df.iterrows():
        stock_id = row['StockID']
        name = row['Name']
        cost = row['CostPrice']
        
        # 取得最新數據
        curr_price = row['Price'] if pd.notnull(row['Price']) else row['LastPrice']
        curr_tech = row['TechScore'] if pd.notnull(row['TechScore']) else 0
        
        # 更新持有期間最高價
        prev_high = row['HighestPrice']
        new_high = max(prev_high, curr_price)
        
        # 計算損益
        pnl_pct = ((curr_price - cost) / cost) * 100
        
        # 參數設定
        trailing_stop_price = new_high * 0.90
        
        # --- V33 核心判斷邏輯 ---
        status = "續抱 (HOLD)"
        reason = "趨勢延續"
        signal_type = "hold-safe" # 預設綠色

        # 1. 移動停利 (優先)
        if curr_price < trailing_stop_price:
            status = "賣出 (停利損)"
            reason = f"跌破最高價 {new_high} 的 10%"
            signal_type = "sell-signal"
        
        # 2. 技術轉弱
        elif curr_tech < 60:
            status = "賣出 (技術轉弱)"
            reason = f"技術分 {curr_tech} 低於 60"
            signal_type = "sell-signal"
        
        # 3. 突破 80 元保護機制
        elif curr_price >= 80:
            status = "續抱 (強勢)"
            reason = "突破 80 元，利潤奔跑模式"
            signal_type = "hold-run"
            
        results.append({
            "StockID": stock_id,
            "Name": name,
            "Cost": cost,
            "Current": curr_price,
            "Highest": new_high,
            "TechScore": curr_tech,
            "PnL%": round(pnl_pct, 2),
            "Action": status,
            "Reason": reason,
            "Signal": signal_type
        })
        
    return pd.DataFrame(results)

# ==========================================
# 3. Session State 初始化
# ==========================================
if 'inventory' not in st.session_state:
    st.session_state.inventory = pd.DataFrame([
        # 預設兩檔示範
        {'StockID': '9999', 'Name': '示範飆股', 'CostPrice': 40, 'LastPrice': 40, 'HighestPrice': 40},
        {'StockID': '8888', 'Name': '示範弱勢', 'CostPrice': 50, 'LastPrice': 50, 'HighestPrice': 50}
    ])

# ==========================================
# 4. 主介面
# ==========================================

st.title(f"📈 V33 智能選股系統 (NSK Ver.)")
st.caption(f"Time: {current_time} | 邏輯: 突破80續抱 / 回檔10%賣出 / 技術<60賣出")

tab1, tab2 = st.tabs(["🔍 V32 選股掃描", "🛡️ V33 庫存監控"])

df_market = get_market_data()

# --- Tab 1: 選股 ---
with tab1:
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader("今日潛力標的 (Price < 80)")
    with col2:
        if st.button("🔄 刷新市場"):
            st.rerun()

    df_top20 = strategy_v32_selection(df_market)
    
    # 單純的勾選買入
    df_display = df_top20.copy()
    df_display['Buy'] = False 
    
    edited_df = st.data_editor(
        df_display,
        column_config={
            "Buy": st.column_config.CheckboxColumn("模擬買入", width="small"),
            "TotalScore": st.column_config.ProgressColumn("總分", format="%d", min_value=0, max_value=100),
        },
        disabled=["StockID", "Name", "Price", "TechScore", "VolScore", "TotalScore", "Volume"],
        hide_index=True,
        height=700
    )

    # 處理買入
    stocks_to_buy = edited_df[edited_df['Buy'] == True]
    if not stocks_to_buy.empty:
        if st.button(f"
