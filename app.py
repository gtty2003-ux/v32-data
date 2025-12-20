import streamlit as st
import pandas as pd
import numpy as np
import requests
import zipfile
import io

# ==========================================
# 1. 頁面設定
# ==========================================
st.set_page_config(
    page_title="V32 智能選股 (Cloud Direct)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stDataFrame { font-size: 14px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 雲端資料讀取引擎 (Cloud Fetcher)
# ==========================================

@st.cache_data(ttl=3600) # 設定快取 1 小時，避免頻繁重複下載
def get_data_from_google_drive():
    """
    直接從指定的 Google Drive 連結下載 ZIP 並讀取 CSV。
    目標檔案 ID: 1VKDBdxyYoqrNaKBMknmAaxEo-TLC4CVo
    """
    file_id = '1VKDBdxyYoqrNaKBMknmAaxEo-TLC4CVo'
    # 轉換為直連下載 URL
    download_url = f'https://drive.google.com/uc?id={file_id}&export=download'
    
    status_text = st.empty()
    status_text.info("☁️ 正在從 Google Drive 雲端下載最新資料，請稍候...")
    
    try:
        # 1. 發送請求下載檔案
        response = requests.get(download_url)
        
        if response.status_code == 200:
            # 2. 在記憶體中解壓縮 (不存到硬碟)
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # 找出裡面的 CSV
                csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                if not csv_files:
                    st.error("雲端壓縮檔內找不到 CSV 檔案！")
                    return pd.DataFrame()
                
                # 讀取第一個 CSV
                target_file = csv_files[0]
                df = pd.read_csv(z.open(target_file))
                
                status_text.success(f"✅ 成功讀取雲端檔案：{target_file}")
                
                # --- V32 即時評分計算 (因為原始檔只有行情) ---
                # 清洗數據 (移除逗號)
                cols = ['ClosingPrice', 'Change', 'TradeVolume']
                for c in cols:
                    if c in df.columns and df[c].dtype == object:
                        df[c] = df[c].astype(str).str.replace(',', '').astype(float)
                
                # 計算 V32 分數
                # 技術分: 用漲跌幅模擬 (漲停=100)
                if 'ClosingPrice' in df.columns and 'Change' in df.columns:
                    prev_close = df['ClosingPrice'] - df['Change']
                    pct_change = (df['Change'] / prev_close) * 100
                    df['TechScore'] = (60 + pct_change * 4).clip(0, 100).fillna(0).astype(int)
                
                # 量能分: 用成交量對數排行模擬
                if 'TradeVolume' in df.columns:
                    log_vol = np.log1p(df['TradeVolume'])
                    df['VolScore'] = ((log_vol - log_vol.min()) / (log_vol.max() - log_vol.min()) * 100).fillna(0).astype(int)
                
                # 總分
                df['TotalScore'] = (df['TechScore'] * 0.7 + df['VolScore'] * 0.3).round(2)
                
                return df
        else:
            st.error(f"無法下載檔案，HTTP 狀態碼: {response.status_code}")
            st.warning("請確認 Google Drive 檔案權限已設定為「知道連結的使用者皆可檢視」。")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"雲端連線失敗: {e}")
        return pd.DataFrame()

# ==========================================
# 3. V32 選股邏輯 (核心不變)
# ==========================================

def strategy_v32_selection(df):
    if df.empty: return df
    
    # 排除 ETF (假設 Code 是字串)
    if 'Code' in df.columns:
        df = df[~df['Code'].astype(str).str.startswith('00')]

    # 1. 價格 < 80
    mask_price = df['ClosingPrice'] < 80
    df_filtered = df[mask_price].copy()
    
    # 2. Top 20
    df_top20 = df_filtered.sort_values(by='TotalScore', ascending=False).head(20)
    return df_top20.reset_index(drop=True)

# ==========================================
# 4. 主介面
# ==========================================

st.title("📈 V32 智能選股 (Cloud Source)")
st.caption(f"資料來源: Google Drive 直連 ({'1VKDBdxyYoqrNaKBMknmAaxEo-TLC4CVo'}) | 模式: V32 標準")

if 'inventory' not in st.session_state:
    st.session_state.inventory = pd.DataFrame(columns=['Code', 'Name', 'Cost', 'Qty'])

tab1, tab2 = st.tabs(["🔍 雲端選股掃描", "📊 我的庫存"])

# 執行雲端下載與讀取
df_market = get_data_from_google_drive()

with tab1:
    col1, col2 = st.columns([4, 1])
    with col1:
        st.write("V32 篩選結果 (<$80, Top 20)")
    with col2:
        # 清除快取並重新下載
        if st.button("🔄 強制更新雲端資料"):
            st.cache_data.clear()
            st.rerun()

    if not df_market.empty:
        df_top20 = strategy_v32_selection(df_market)
        
        # 互動選股表
        df_display = df_top20.copy()
        df_display['Select'] = False
        df_display['Qty'] = 1
        
        edited_df = st.data_editor(
            df_display[['Select', 'Qty', 'Code', 'Name', 'ClosingPrice', 'TotalScore', 'TechScore', 'VolScore']],
            column_config={
                "Select": st.column_config.CheckboxColumn("買入", width="small"),
                "Qty": st.column_config.NumberColumn("張數", min_value=1, width="small"),
                "ClosingPrice": st.column_config.NumberColumn("現價", format="%.2f"),
                "TotalScore": st.column_config.ProgressColumn("總分", format="%.1f", min_value=0, max_value=100),
            },
            hide_index=True,
            height=735
        )
        
        to_buy = edited_df[edited_df['Select'] == True]
        if not to_buy.empty:
            if st.button(f"確認買入 {len(to_buy)} 檔"):
                for idx, row in to_buy.iterrows():
                    if row['Code'] not in st.session_state.inventory['Code'].values:
                        new_row = pd.DataFrame([{
                            'Code': row['Code'],
                            'Name': row['Name'],
                            'Cost': float(row['ClosingPrice']),
                            'Qty': int(row['Qty'])
                        }])
                        st.session_state.inventory = pd.concat([st.session_state.inventory, new_row], ignore_index=True)
                st.success("庫存已更新！")
                st.rerun()
    else:
        st.warning("尚無資料，請檢查網路連線或檔案權限。")

with tab2:
    if st.session_state.inventory.empty:
        st.info("目前無庫存。")
    else:
        st.dataframe(
            st.session_state.inventory,
            column_config={
                "Cost": st.column_config.NumberColumn("成本價", format="%.2f"),
                "Qty": st.column_config.NumberColumn("股數", format="%d")
            },
            hide_index=True
        )
        if st.button("清空庫存"):
            st.session_state.inventory = pd.DataFrame(columns=['Code', 'Name', 'Cost', 'Qty'])
            st.rerun()
