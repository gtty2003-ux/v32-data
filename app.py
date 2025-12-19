import streamlit as st
import pandas as pd
import os
from datetime import datetime
import pytz

# --- 設定頁面資訊 ---
st.set_page_config(
    page_title="V32 戰情室",
    layout="wide",
    page_icon="📈"
)

# --- 樣式設定 ---
st.markdown("""
    <style>
    /* 表頭顏色設定為淺綠色 */
    .stDataFrame thead tr th {
        background-color: #C8E6C9 !important;
        color: #000000 !important;
    }
    /* 調整指標數值大小 */
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

# --- 讀取與分類資料 ---
@st.cache_data(ttl=60)
def load_data():
    file_path = 'v32_recommend.csv'
    
    if not os.path.exists(file_path):
        return None, "找不到資料檔 v32_recommend.csv"
    
    try:
        df = pd.read_csv(file_path)
        
        # 1. 數值處理
        if '總分' in df.columns:
            df['總分'] = pd.to_numeric(df['總分'], errors='coerce').fillna(0)
        
        # 2. 尋找關鍵欄位 (代號 & 名稱)
        code_col = None
        name_col = None
        
        # 找代號欄位
        for c in ['代號', 'Code', 'Symbol', '股票代號']:
            if c in df.columns:
                code_col = c
                break
        
        # 找名稱欄位 (用於判斷 KY 或 特別股)
        for n in ['名稱', 'Name', '股票名稱']:
            if n in df.columns:
                name_col = n
                break
                
        # 3. 建立分類標籤
        # category: 'General' (一般個股), 'Special' (非一般: ETF/KY/特/TDR)
        if code_col:
            df[code_col] = df[code_col].astype(str)
            # 如果沒有名稱欄位，就給空字串避免報錯，但會影響 KY 判斷
            df['temp_name'] = df[name_col].astype(str) if name_col else ""
            
            def classify_stock(row):
                code = row[code_col]
                name = row['temp_name']
                
                # (1) ETF: 00 開頭
                if code.startswith('00'):
                    return 'Special'
                
                # (2) TDR: 91 開頭
                if code.startswith('91'):
                    return 'Special'
                
                # (3) 特別股: 代號含有字母 (如 2881A) 或 名稱含 "特"
                # 檢查最後一位是否為字母 (Python 的 isalpha())
                if code[-1].isalpha(): 
                    return 'Special'
                if '特' in name:
                    return 'Special'
                    
                # (4) 外國企業: 名稱含 KY
                if 'KY' in name:
                    return 'Special'
                
                # 剩下的就是一般個股
                return 'General'

            df['category'] = df.apply(classify_stock, axis=1)
            # 刪除暫存欄位
            df = df.drop(columns=['temp_name'])
        else:
            # 找不到代號欄位，無法分類，全部當作一般
            df['category'] = 'General'
            
        return df, None
    except Exception as e:
        return None, str(e)

# --- 主程式 ---
def main():
    st.title("📈 V32 戰情室")
    st.caption(f"最後更新時間 (TW): {get_taiwan_time()}")

    tab_scan, tab_monitor = st.tabs(["🚀 Top 10 掃描", "💼 持股監控 (開發中)"])

    with tab_scan:
        df, error = load_data()
        
        if error:
            st.error(f"資料讀取錯誤: {error}")
        elif df is None or df.empty:
            st.warning("目前沒有符合 V32 標準的標的。")
        else:
            # 拆分資料
            df_general = df[df['category'] == 'General'].copy() # 一般個股
            df_special = df[df['category'] == 'Special'].copy() # 非一般 (ETF/KY/特/TDR)
            
            # 建立子分頁
            sub_tab1, sub_tab2 = st.tabs(["🏢 一般個股 Top 10", "📊 ETF與其他 Top 10"])
            
            cols_to_hide = ['Unnamed: 0', 'category']
            
            # --- 表格 1: 一般個股 ---
            with sub_tab1:
                if not df_general.empty:
                    display_gen = df_general.head(10)
                    cols = [c for c in display_gen.columns if c not in cols_to_hide]
                    
                    st.dataframe(
                        display_gen[cols],
                        height=400,
                        use_container_width=True,
                        hide_index=True
                    )
                    st.caption(f"包含：純台資企業普通股 (排除 KY/TDR/特別股)。共 {len(df_general)} 檔。")
                else:
                    st.info("無符合的一般個股。")

            # --- 表格 2: 非一般 (ETF/KY/特/TDR) ---
            with sub_tab2:
                if not df_special.empty:
                    display_spec = df_special.head(10)
                    cols = [c for c in display_spec.columns if c not in cols_to_hide]
                    
                    st.dataframe(
                        display_spec[cols],
                        height=400,
                        use_container_width=True,
                        hide_index=True
                    )
                    st.caption(f"包含：ETF (00)、外國企業 (KY)、特別股、存託憑證 (91)。共 {len(df_special)} 檔。")
                else:
                    st.info("無符合的特殊類股。")

    with tab_monitor:
        st.info("🚧 持股監控與損益管理功能開發中...")

if __name__ == "__main__":
    main()
