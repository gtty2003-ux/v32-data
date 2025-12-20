import streamlit as st
import pandas as pd
import numpy as np # 僅用於數據處理，不進行模擬
from datetime import datetime
import pytz
import os

# ==========================================
# 1. 頁面配置與樣式 (V32 Standard)
# ==========================================
st.set_page_config(
    page_title="V32 智能選股系統 (Standard)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# V32 指定配色: 表格高亮 #C8E6C9
st.markdown("""
    <style>
    .stDataFrame { font-size: 14px; }
    .highlight-v32 { background-color: #C8E6C9 !important; color: black !important; }
    </style>
""", unsafe_allow_html=True)

# 時間標準: 台北時間 UTC+8
tw_tz = pytz.timezone('Asia/Taipei')
current_time = datetime.now(tw_tz).strftime('%Y-%m-%d %H:%M:%S')

# ==========================================
# 2. 資料讀取與處理 (Data Ingestion)
# ==========================================

def get_market_data():
    """
    [V32 核心] 讀取真實資料檔案。
    不再模擬，只針對現在。
    """
    file_path = 'twse_data.csv' # 請確認檔名一致
    
    if not os.path.exists(file_path):
        st.error(f"❌ 找不到資料檔：{file_path}")
        st.warning("請將您的雲端檔案下載，改名為 'twse_data.csv' 並放在同目錄下。")
        # 回傳空表以防當機
        return pd.DataFrame(columns=['StockID', 'Name', 'Price', 'TechScore', 'VolScore'])
    
    try:
        # 讀取 CSV (假設編碼為 utf-8 或 big5，視您的檔案而定)
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='big5') # 嘗試 Big5 (常見於台股資料)

        # 資料前處理 (確保欄位名稱對應)
        # 假設您的 CSV 欄位名稱可能不同，這裡做個簡單的映射防呆
        # 這裡預設您的 CSV 已經有: StockID, Name, Price, TechScore, VolScore
        
        # 確保數值型態正確
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df['TechScore'] = pd.to_numeric(df['TechScore'], errors='coerce')
        df['VolScore'] = pd.to_numeric(df['VolScore'], errors='coerce')
        
        # V32 評分邏輯: 技術分(A)*0.7 + 量能分(B)*0.3
        df['TotalScore'] = (df['TechScore'] * 0.7) + (df['VolScore'] * 0.3)
        df['TotalScore'] = df['TotalScore'].round(2)
        
        return df
        
    except Exception as e:
        st.error(f"讀取資料發生錯誤: {e}")
        return pd.DataFrame()

# ==========================================
# 3. V32 選股邏輯 (Selection Logic)
# ==========================================

def strategy_v32_selection(df):
    """
    V32 標準選股邏輯：
    1. 低價門檻: Price < 80
    2. 數量保證: Top 20 (依總分排序)
    """
    if df.empty:
        return df
        
    # 1. 濾除資料不全的列
    df = df.dropna(subset=['Price', 'TotalScore'])
    
    # 2. 執行低價篩選
    mask_price = df['Price'] < 80
    df_filtered = df[mask_price].copy()
    
    # 3. 排序並取 Top 20
    df_top20 = df_filtered.sort_values(by='TotalScore', ascending=False).head(20)
    
    # 整理顯示欄位
    return df_top20.reset_index(drop=True)

# ==========================================
# 4. Session State (庫存暫存)
# ==========================================
if 'inventory' not in st.session_state:
    st.session_state.inventory = pd.DataFrame(columns=['StockID', 'Name', 'CostPrice', 'Quantity'])

# ==========================================
# 5. 主介面 (Main Layout)
# ==========================================

st.title(f"📈 V32 智能選股系統 (Standard Ver.)")
st.caption(f"系統時間: {current_time} | 資料來源: twse_data.csv | 核心邏輯: V32 (<80元, Top 20)")

tab1, tab2 = st.tabs(["🔍 V32 選股掃描 (Top 20)", "📊 持股監控 (Inventory)"])

# 讀取資料
df_market = get_market_data()

# --- Tab 1: 選股結果 ---
with tab1:
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader("V32 每日精選 (Top 20)")
    with col2:
        if st.button("🔄 重新讀取資料"):
            st.rerun()

    if df_market.empty:
        st.info("尚無資料，請確認 CSV 檔案是否就緒。")
    else:
        # 執行 V32 選股
        df_top20 = strategy_v32_selection(df_market)
        
        # 顯示互動表格 (包含買入功能)
        # 為了介面乾淨，複製一份來顯示
        df_display = df_top20.copy()
        df_display['Select'] = False # 勾選框
        df_display['Qty'] = 1        # 張數預設
        
        edited_df = st.data_editor(
            df_display,
            column_config={
                "Select": st.column_config.CheckboxColumn("加入庫存", width="small"),
                "Qty": st.column_config.NumberColumn("張數", min_value=1, step=1, width="small"),
                "TotalScore": st.column_config.ProgressColumn("V32 總分", format="%.1f", min_value=0, max_value=100),
                "Price": st.column_config.NumberColumn("收盤價", format="%.2f"),
                "TechScore": st.column_config.NumberColumn("技術分(70%)", format="%d"),
                "VolScore": st.column_config.NumberColumn("量能分(30%)", format="%d"),
            },
            disabled=["StockID", "Name", "Price", "TechScore", "VolScore", "TotalScore"],
            hide_index=True,
            height=735 # V32 指定高度
        )
        
        # 處理買入動作
        to_buy = edited_df[edited_df['Select'] == True]
        if not to_buy.empty:
            st.divider()
            if st.button(f"確認買入選中的 {len(to_buy)} 檔標的"):
                for idx, row in to_buy.iterrows():
                    # 避免重複加入，若已存在則略過 (V32 簡單邏輯)
                    if row['StockID'] not in st.session_state.inventory['StockID'].values:
                        new_row = pd.DataFrame([{
                            'StockID': row['StockID'],
                            'Name': row['Name'],
                            'CostPrice': float(row['Price']),
                            'Quantity': int(row['Qty'])
                        }])
                        st.session_state.inventory = pd.concat([st.session_state.inventory, new_row], ignore_index=True)
                st.success("已更新庫存！")
                st.rerun()

# --- Tab 2: 庫存管理 (基礎版) ---
with tab2:
    st.subheader("我的持股明細")
    
    if st.session_state.inventory.empty:
        st.write("目前無庫存。")
    else:
        # 計算即時損益 (需比對 df_market 中的最新價)
        # V32 不做複雜的賣出訊號，僅顯示損益
        
        inventory_view = st.session_state.inventory.copy()
        
        # 嘗試從 df_market 抓取最新價 (Current Price)
        # 建立一個 mapping dictionary: StockID -> Price
        if not df_market.empty:
            price_map = df_market.set_index('StockID')['Price'].to_dict()
            inventory_view['CurrentPrice'] = inventory_view['StockID'].map(price_map)
        else:
            inventory_view['CurrentPrice'] = inventory_view['CostPrice'] # 若無市價則假設不變
            
        # 計算損益
        # 損益 = (現價 - 成本) * 張數 * 1000
        inventory_view['PnL_Amt'] = (inventory_view['CurrentPrice'] - inventory_view['CostPrice']) * inventory_view['Quantity'] * 1000
        inventory_view['PnL_Pct'] = ((inventory_view['CurrentPrice'] - inventory_view['CostPrice']) / inventory_view['CostPrice']) * 100
        
        # 顯示表格 (使用 V32 指定配色 highlight)
        def color_pnl(val):
            color = 'red' if val < 0 else 'green'
            return f'color: {color}'

        st.dataframe(
            inventory_view,
            column_config={
                "StockID": "代號",
                "Name": "名稱",
                "CostPrice": st.column_config.NumberColumn("成本均價", format="%.2f"),
                "CurrentPrice": st.column_config.NumberColumn("最新市價", format="%.2f"),
                "Quantity": st.column_config.NumberColumn("庫存張數", format="%d"),
                "PnL_Amt": st.column_config.NumberColumn("未實現損益($)", format="%d"),
                "PnL_Pct": st.column_config.NumberColumn("報酬率(%)", format="%.2f %%"),
            },
            hide_index=True
        )
        
        if st.button("清空庫存"):
            st.session_state.inventory = pd.DataFrame(columns=['StockID', 'Name', 'CostPrice', 'Quantity'])
            st.rerun()
