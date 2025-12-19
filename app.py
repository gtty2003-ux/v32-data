import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
import pytz

# --- 設定頁面資訊 ---
st.set_page_config(
    page_title="V32 戰情室",
    layout="wide",
    page_icon="📈"
)

# --- 樣式設定 (符合 V32 視覺需求) ---
# 強制設定表頭顏色為淺綠色 (#C8E6C9)
st.markdown("""
    <style>
    .stDataFrame thead tr th {
        background-color: #C8E6C9 !important;
        color: #000000 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 時間校正 (中原標準時間 UTC+8) ---
def get_taiwan_time():
    utc_now = datetime.utcnow()
    tw_time = utc_now.replace(tzinfo=pytz.utc).astimezone(pytz.timezone('Asia/Taipei'))
    return tw_time.strftime("%Y-%m-%d %H:%M:%S")

# --- 讀取資料 ---
@st.cache_data(ttl=60)  # 設定快取，避免頻繁讀取
def load_data():
    # 直接讀取同目錄下的 csv (因為 Colab 已經推上來了)
    file_path = 'v32_recommend.csv'
    
    if not os.path.exists(file_path):
        return None, "找不到資料檔 v32_recommend.csv"
    
    try:
        df = pd.read_csv(file_path)
        # 確保欄位是正確的型別
        if '總分' in df.columns:
            df['總分'] = pd.to_numeric(df['總分'], errors='coerce').fillna(0)
        return df, None
    except Exception as e:
        return None, str(e)

# --- 主程式 ---
def main():
    st.title("📈 V32 戰情室 (Top 20 監控)")
    st.caption(f"最後更新時間 (TW): {get_taiwan_time()}")

    # 建立 Tabs
    tab1, tab2 = st.tabs(["🚀 Top 20 掃描", "💼 持股監控 (開發中)"])

    with tab1:
        df, error = load_data()
        
        if error:
            st.error(f"資料讀取錯誤: {error}")
            st.warning("請確認 Colab 是否已成功上傳 v32_recommend.csv")
        elif df is None or df.empty:
            st.warning("目前沒有符合 V32 標準的標的。")
        else:
            # 確保只顯示前 20 筆 (雖然 CSV 應該已經是 Top 20，但雙重保險)
            display_df = df.head(20)
            
            # 格式化顯示 (選擇性隱藏一些技術欄位，讓表格更乾淨)
            # 這裡假設你的 CSV 有這些欄位，若沒有會自動略過
            cols_to_show = [col for col in display_df.columns if col not in ['Unnamed: 0']]
            
            # 設定表格高度為 735 (符合你的需求)
            st.dataframe(
                display_df[cols_to_show],
                height=735,
                use_container_width=True,
                hide_index=True
            )
            
            st.info("💡 評分邏輯：技術分(70%) + 量能分(30%) | 價格門檻：< 80元")

    with tab2:
        st.info("🚧 持股監控與損益管理功能開發中...")
        st.markdown("未來將整合庫存匯入與即時損益計算。")

if __name__ == "__main__":
    main()
