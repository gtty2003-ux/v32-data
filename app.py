import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# ==========================================
# 1. 頁面配置與樣式
# ==========================================
st.set_page_config(
    page_title="V33.1 智能選股系統",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義 CSS
st.markdown("""
    <style>
    .stDataFrame { font-size: 14px; }
    /* 強制 Highlight 顏色 */
    .sell-stop { background-color: #FFCDD2 !important; color: black; } /* 紅: 停損 */
    .sell-profit { background-color: #FFE0B2 !important; color: black; } /* 橘: 獲利了結/技術轉弱 */
    .hold-run { background-color: #B3E5FC !important; color: black; } /* 藍: 獲利奔跑 */
    .hold-safe { background-color: #C8E6C9 !important; color: black; } /* 綠: 續抱 */
    </style>
""", unsafe_allow_html=True)

tw_tz = pytz.timezone('Asia/Taipei')
current_time = datetime.now(tw_tz).strftime('%Y-%m-%d %H:%M:%S')

# ==========================================
# 2. 核心邏輯函式
# ==========================================

def get_market_data():
    """ 生成模擬市場數據 (可替換為真實資料源) """
    np.random.seed(int(datetime.now().timestamp()))
    data = []
    # 模擬 50 檔股票
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
    V33.1 庫存邏輯修正版：
    1. 硬性停損: 跌破成本 7%。
    2. 移動停利: 跌破持有期間最高價 10%。
    3. 動態技術門檻: 
       - 獲利 > 30% 時，技術分需 >= 70 (嚴格)
       - 獲利 < 30% 時，技術分需 >= 60 (標準)
    """
    results = []
    
    # 合併庫存與最新行情
    merged_df = pd.merge(inventory_df, current_market_df[['StockID', 'Price', 'TechScore']], on='StockID', how='left')
    
    for index, row in merged_df.iterrows():
        stock_id = row['StockID']
        name = row['Name']
        cost = row['CostPrice']
        qty = row['Quantity'] # 張數
        
        # 取得最新價與技術分 (若無數據則沿用)
        curr_price = row['Price'] if pd.notnull(row['Price']) else row['LastPrice']
        curr_tech = row['TechScore'] if pd.notnull(row['TechScore']) else 0
        
        # 更新最高價
        prev_high = row['HighestPrice']
        new_high = max(prev_high, curr_price)
        
        # 計算損益
        pnl_val = (curr_price - cost) * qty * 1000 # 損益金額
        pnl_pct = ((curr_price - cost) / cost) * 100 # 損益 %
        
        # 參數設定
        hard_stop_price = cost * 0.93 # 硬性停損 (-7%)
        trailing_stop_price = new_high * 0.90 # 移動停利 (-10% from High)
        
        # 動態技術門檻設定
        if pnl_pct > 30:
            tech_threshold = 70 # 高檔區更嚴格
            tech_status_msg = "高檔嚴控 (需 > 70)"
        else:
            tech_threshold = 60 # 一般區
            tech_status_msg = "標準監控 (需 > 60)"

        # --- 判斷邏輯 (優先順序很重要) ---
        status = "續抱 (HOLD)"
        reason = f"趨勢穩健 | {tech_status_msg}"
        signal_type = "hold-safe"

        # 1. 硬性停損 (保命第一)
        if curr_price < hard_stop_price:
            status = "賣出 (硬性停損)"
            reason = f"跌破成本 7% (價位 {hard_stop_price:.2f})"
            signal_type = "sell-stop"
            
        # 2. 移動停利 (保住獲利)
        elif curr_price < trailing_stop_price:
            status = "賣出 (移動停利)"
            reason = f"自高點回檔 > 10% (價位 {trailing_stop_price:.2f})"
            signal_type = "sell-profit"
            
        # 3. 技術面檢測 (動態門檻)
        elif curr_tech < tech_threshold:
            status = "賣出 (技術轉弱)"
            reason = f"技術分 {curr_tech} 低於門檻 {tech_threshold}"
            signal_type = "sell-profit"
            
        # 4. 獲利奔跑模式 (突破 80)
        elif curr_price >= 80:
            status = "續抱 (強勢)"
            reason = "突破 80 元，進入主升段"
            signal_type = "hold-run"

        results.append({
            "StockID": stock_id,
            "Name": name,
            "Cost": cost,
            "Qty": qty,
            "Current": curr_price,
            "Highest": new_high,
            "TechScore": curr_tech,
            "PnL": pnl_val,
            "PnL%": pnl_pct,
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
        {'StockID': '9999', 'Name': '示範飆股', 'CostPrice': 40.0, 'LastPrice': 40.0, 'HighestPrice': 40.0, 'Quantity': 2},
        {'StockID': '8888', 'Name': '示範停損', 'CostPrice': 50.0, 'LastPrice': 50.0, 'HighestPrice': 50.0, 'Quantity': 1}
    ])

# ==========================================
# 4. 主介面
# ==========================================

st.title(f"📈 V33.1 智能選股系統 (NSK Ver.)")
st.caption(f"Time: {current_time} | 邏輯: 硬性停損 7% / 動態技術門檻")

tab1, tab2 = st.tabs(["🔍 V32 選股 (市場)", "🛡️ V33.1 庫存 (資產)"])

df_market = get_market_data()

# --- Tab 1: 選股與買入 ---
with tab1:
    col1, col2 = st.columns([4, 1])
    with col1:
        st.write("今日 Top 20 潛力股 (Price < 80)")
    with col2:
        if st.button("🔄 刷新市場"):
            st.rerun()

    df_top20 = strategy_v32_selection(df_market)
    
    # 準備顯示的資料，增加 '張數' 欄位
    df_display = df_top20.copy()
    df_display['Buy'] = False 
    df_display['張數'] = 1 # 預設 1 張
    
    # 設定可編輯表格
    edited_df = st.data_editor(
        df_display,
        column_config={
            "Buy": st.column_config.CheckboxColumn("買入", width="small"),
            "張數": st.column_config.NumberColumn("張數", min_value=1, max_value=100, step=1, width="small"),
            "TotalScore": st.column_config.ProgressColumn("總分", format="%d", min_value=0, max_value=100),
            "Price": st.column_config.NumberColumn("現價", format="%.2f"),
        },
        disabled=["StockID", "Name", "Price", "TechScore", "VolScore", "TotalScore", "Volume"],
        hide_index=True,
        height=700
    )

    # 處理買入
    stocks_to_buy = edited_df[edited_df['Buy'] == True]
    if not stocks_to_buy.empty:
        if st.button(f"確認買入 {len(stocks_to_buy)} 檔標的"):
            for index, row in stocks_to_buy.iterrows():
                # 簡單去重：若已存在，則不重複新增 (實戰可改為加碼邏輯)
                if row['StockID'] not in st.session_state.inventory['StockID'].values:
                    new_entry = pd.DataFrame([{
                        'StockID': row['StockID'], 
                        'Name': row['Name'], 
                        'CostPrice': float(row['Price']), 
                        'LastPrice': float(row['Price']),
                        'HighestPrice': float(row['Price']),
                        'Quantity': int(row['張數'])
                    }])
                    st.session_state.inventory = pd.concat([st.session_state.inventory, new_entry], ignore_index=True)
            st.success("已新增至庫存！")
            st.rerun()

# --- Tab 2: 庫存管理 ---
with tab2:
    st.write("目前持股狀態與操作建議")
    
    if st.session_state.inventory.empty:
        st.info("目前無庫存。")
    else:
        # --- 模擬數據注入 (為了測試各種情境) ---
        # 1. 示範飆股: 漲到 85 (獲利 > 100%), 技術分給 65 -> 應觸發 "高檔嚴控" 而賣出 (因為 >30% 獲利需 70 分)
        df_market.loc[df_market['StockID'] == '9999', 'Price'] = 85.0
        df_market.loc[df_market['StockID'] == '9999', 'TechScore'] = 65 
        
        # 2. 示範停損: 跌到 46 (成本 50, 46/50 = 0.92, -8%) -> 應觸發 "硬性停損"
        df_market.loc[df_market['StockID'] == '8888', 'Price'] = 46.0
        # ------------------------------------
        
        inventory_analysis = strategy_v33_inventory_check(st.session_state.inventory, df_market)
        
        def highlight_signal(row):
            if row['Signal'] == 'sell-stop': return ['background-color: #FFCDD2; color: black'] * len(row)
            if row['Signal'] == 'sell-profit': return ['background-color: #FFE0B2; color: black'] * len(row)
            if row['Signal'] == 'hold-run': return ['background-color: #B3E5FC; color: black'] * len(row)
            return ['background-color: #C8E6C9; color: black'] * len(row)

        st.dataframe(
            inventory_analysis.style.apply(highlight_signal, axis=1),
            column_config={
                "StockID": "代號",
                "Name": "名稱",
                "Cost": st.column_config.NumberColumn("成本", format="%.2f"),
                "Qty": st.column_config.NumberColumn("張數", format="%d"),
                "Current": st.column_config.NumberColumn("現價", format="%.2f"),
                "Highest": st.column_config.NumberColumn("最高價", format="%.2f"),
                "PnL": st.column_config.NumberColumn("總損益($)", format="%.2f"),
                "PnL%": st.column_config.NumberColumn("損益 %", format="%.2f %%"),
                "TechScore": st.column_config.NumberColumn("技術分", format="%d"),
                "Action": "建議動作",
                "Reason": "判斷依據"
            },
            hide_index=True,
            height=500
        )
        
        if st.button("🗑️ 清空庫存 (重置)"):
            st.session_state.inventory = pd.DataFrame(columns=['StockID', 'Name', 'CostPrice', 'LastPrice', 'HighestPrice', 'Quantity'])
            st.rerun()
