import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os

# ==========================================
# 1. 頁面設定
# ==========================================
st.set_page_config(
    page_title="V32 智能選股 (Real Data)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stDataFrame { font-size: 14px; }
    /* V32 指定配色: 淺綠高亮 */
    .highlight-v32 { background-color: #C8E6C9 !important; color: black !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 資料讀取與即時評分引擎
# ==========================================

def load_and_score_data():
    """
    讀取 V32_Standard_Data.csv.zip 並根據原始數據計算 V32 分數
    """
    zip_path = 'V32_Standard_Data.csv.zip'
    csv_name = 'V32_Standard_Data.csv' # 假設解壓後的檔名，若不同請修改
    
    df = pd.DataFrame()

    # 1. 嘗試讀取 ZIP
    if os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                # 自動尋找 zip 內的第一個 csv 檔
                file_list = [f for f in z.namelist() if f.endswith('.csv')]
                if file_list:
                    with z.open(file_list[0]) as f:
                        df = pd.read_csv(f)
                else:
                    st.error("ZIP 檔中找不到 CSV 檔案")
                    return df
        except Exception as e:
            st.error(f"ZIP 讀取失敗: {e}")
            return df
    elif os.path.exists(csv_name):
        # 備案：讀取已解壓的 CSV
        df = pd.read_csv(csv_name)
    else:
        st.error(f"找不到資料檔！請確保 {zip_path} 在程式目錄下。")
        return df

    # 2. 資料清洗與型別轉換
    # 確保數值欄位正確 (移除可能的千分位逗號)
    cols_to_clean = ['ClosingPrice', 'Change', 'TradeVolume', 'OpeningPrice', 'HighestPrice', 'LowestPrice']
    for col in cols_to_clean:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    
    # 3. V32 核心：原始數據轉評分 (因為 CSV 只有原始行情)
    # 技術分 Proxy: 使用 '漲跌幅' (Change / (Close - Change))
    # 邏輯：漲停(+10%) = 99分, 平盤 = 60分, 跌停(-10%) = 20分
    if 'ClosingPrice' in df.columns and 'Change' in df.columns:
        df['PrevClose'] = df['ClosingPrice'] - df['Change']
        df['PctChange'] = (df['Change'] / df['PrevClose']) * 100
        
        # 線性映射: -10% -> 20分, +10% -> 100分
        df['TechScore'] = 60 + (df['PctChange'] * 4) 
        df['TechScore'] = df['TechScore'].clip(0, 100).fillna(0).astype(int)

    # 量能分 Proxy: 使用成交量對數排名
    if 'TradeVolume' in df.columns:
        # 使用 Log 避免極端值影響，並標準化到 0-100
        df['LogVol'] = np.log1p(df['TradeVolume'])
        min_vol = df['LogVol'].min()
        max_vol = df['LogVol'].max()
        df['VolScore'] = ((df['LogVol'] - min_vol) / (max_vol - min_vol) * 100).fillna(0).astype(int)

    # 4. 計算總分 (7:3)
    if 'TechScore' in df.columns and 'VolScore' in df.columns:
        df['TotalScore'] = (df['TechScore'] * 0.7) + (df['VolScore'] * 0.3)
        df['TotalScore'] = df['TotalScore'].round(2)
    
    return df

# ==========================================
# 3. V32 選股策略
# ==========================================

def strategy_v32_selection(df):
    """
    V32 標準: < 80 元, Top 20
    """
    if df.empty: return df
    
    # 1. 篩選低價股 (< 80)
    # 排除 ETF (代號 00 開頭) 以聚焦個股 (可選)
    df = df[~df['Code'].astype(str).str.startswith('00')]
    
    mask_price = df['ClosingPrice'] < 80
    df_filtered = df[mask_price].copy()
    
    # 2. 排序取 Top 20
    df_top20 = df_filtered.sort_values(by='TotalScore', ascending=False).head(20)
    
    return df_top20.reset_index(drop=True)

# ==========================================
# 4. 主程式介面
# ==========================================

st.title("📈 V32 智能選股系統 (Live Data Mode)")
st.caption("資料來源: 2025/12/19 真實盤後數據 | 模式: 嚴格執行 V32 (無模擬)")

# 初始化庫存 Session
if 'inventory' not in st.session_state:
    st.session_state.inventory = pd.DataFrame(columns=['Code', 'Name', 'Cost', 'Qty'])

tab1, tab2 = st.tabs(["🔍 市場掃描 (Top 20)", "📊 我的庫存"])

# 載入數據
df_market = load_and_score_data()

with tab1:
    col1, col2 = st.columns([4, 1])
    with col1:
        st.write(f"今日掃描結果 (基準日: 2025-12-19)")
    with col2:
        if st.button("🔄 重新掃描"):
            st.rerun()

    if not df_market.empty:
        # 執行選股
        df_top20 = strategy_v32_selection(df_market)
        
        # 準備顯示資料
        df_display = df_top20[['Code', 'Name', 'ClosingPrice', 'PctChange', 'TechScore', 'VolScore', 'TotalScore']].copy()
        df_display['Select'] = False
        df_display['Qty'] = 1
        
        # 互動表格
        edited_df = st.data_editor(
            df_display,
            column_config={
                "Select": st.column_config.CheckboxColumn("買入", width="small"),
                "Qty": st.column_config.NumberColumn("張數", min_value=1, width="small"),
                "Code": "代號",
                "Name": "名稱",
                "ClosingPrice": st.column_config.NumberColumn("收盤價", format="%.2f"),
                "PctChange": st.column_config.NumberColumn("漲跌幅%", format="%.2f%%"),
                "TechScore": st.column_config.ProgressColumn("技術分", format="%d", min_value=0, max_value=100),
                "VolScore": st.column_config.ProgressColumn("量能分", format="%d", min_value=0, max_value=100),
                "TotalScore": st.column_config.NumberColumn("V32總分", format="%.2f"),
            },
            hide_index=True,
            height=735
        )
        
        # 處理買入
        to_buy = edited_df[edited_df['Select'] == True]
        if not to_buy.empty:
            if st.button(f"下單買入 {len(to_buy)} 檔"):
                for idx, row in to_buy.iterrows():
                    if row['Code'] not in st.session_state.inventory['Code'].values:
                        new_row = pd.DataFrame([{
                            'Code': row['Code'],
                            'Name': row['Name'],
                            'Cost': float(row['ClosingPrice']),
                            'Qty': int(row['Qty'])
                        }])
                        st.session_state.inventory = pd.concat([st.session_state.inventory, new_row], ignore_index=True)
                st.success("成交回報：已加入庫存！")
                st.rerun()
    else:
        st.warning("無法載入數據，請檢查 ZIP 檔內容。")

with tab2:
    if st.session_state.inventory.empty:
        st.info("庫存為空，請至 Tab 1 選股。")
    else:
        # 簡易庫存顯示
        st.dataframe(
            st.session_state.inventory, 
            column_config={
                "Cost": st.column_config.NumberColumn("成本價", format="%.2f"),
                "Qty": st.column_config.NumberColumn("股數", format="%d")
            },
            hide_index=True,
            use_container_width=True
        )
        if st.button("清空庫存"):
            st.session_state.inventory = pd.DataFrame(columns=['Code', 'Name', 'Cost', 'Qty'])
            st.rerun()
