import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from github import Github 
from datetime import datetime
import pytz

# --- 設定頁面資訊 ---
st.set_page_config(
    page_title="V32 戰情室 (Attack Focus)",
    layout="wide",
    page_icon="⚔️"
)

# --- 樣式設定 (符合你的綠色/黑色高對比需求) ---
st.markdown("""
    <style>
    /* 表頭樣式：淺綠色背景 + 黑色文字 */
    .stDataFrame thead tr th {
        background-color: #C8E6C9 !important; 
        color: black !important;
        font-weight: bold;
        font-size: 16px;
    }
    /* 指標數值加大 */
    div[data-testid="stMetricValue"] {
        font-size: 26px;
        font-weight: bold;
        color: #1b5e20;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 全域變數 ---
# 請確認你的 Repo 名稱是否正確
REPO_KEY = "gtty2003-ux/v32-data" 
FILE_PATH = "holdings.csv"

# --- 工具函數 ---
def get_taiwan_time():
    utc_now = datetime.utcnow()
    tw_time = utc_now.replace(tzinfo=pytz.utc).astimezone(pytz.timezone('Asia/Taipei'))
    return tw_time.strftime("%Y-%m-%d %H:%M:%S")

def color_surplus(val):
    """損益著色：台股慣例 紅賺/綠賠"""
    if val > 0: return 'color: #d32f2f; font-weight: bold;' # 紅
    elif val < 0: return 'color: #388e3c; font-weight: bold;' # 綠
    return 'color: black'

def color_signal_bg(val):
    """操作建議燈號"""
    if "🔴" in val: return 'background-color: #ffcdd2; color: #b71c1c; font-weight: bold;' # 淺紅底深紅字
    if "🟡" in val: return 'background-color: #fff9c4; color: #f57f17; font-weight: bold;' # 淺黃底深橘字
    if "🟢" in val: return 'background-color: #c8e6c9; color: #1b5e20; font-weight: bold;' # 淺綠底深綠字
    return ''

# --- 核心邏輯：GitHub 資料存取 ---
def load_holdings():
    try:
        token = st.secrets["general"]["GITHUB_TOKEN"]
        g = Github(token)
        repo = g.get_repo(REPO_KEY)
        contents = repo.get_contents(FILE_PATH)
        df = pd.read_csv(contents.download_url)
        # 強制轉型避免錯誤
        df['股票代號'] = df['股票代號'].astype(str).str.strip()
        # 補齊欄位防呆
        expected_cols = ["股票代號", "買入均價", "持有股數"]
        for c in expected_cols:
            if c not in df.columns: 
                df[c] = 0 if c != "股票代號" else ""
        return df[expected_cols]
    except Exception as e:
        # 若檔案不存在或讀取失敗，回傳空表
        return pd.DataFrame(columns=["股票代號", "買入均價", "持有股數"])

def save_holdings(df):
    try:
        token = st.secrets["general"]["GITHUB_TOKEN"]
        g = Github(token)
        repo = g.get_repo(REPO_KEY)
        csv_content = df.to_csv(index=False)
        
        try:
            # 嘗試更新現有檔案
            contents = repo.get_contents(FILE_PATH)
            repo.update_file(contents.path, f"Update {get_taiwan_time()}", csv_content, contents.sha)
            st.toast("✅ 庫存雲端備份成功！", icon="☁️")
        except:
            # 若檔案不存在則建立
            repo.create_file(FILE_PATH, "Create holdings.csv", csv_content)
            st.toast("✅ 庫存檔建立成功！", icon="☁️")
            
    except Exception as e:
        st.error(f"❌ 儲存失敗: {e}")

# --- 核心邏輯：V32 引擎 (簡化版，用於即時運算庫存) ---
def get_stock_health(symbol, ref_score_map):
    """
    針對單一庫存進行健康檢查
    returns: (現價, MA20, 攻擊分, 建議訊號)
    """
    try:
        ticker = yf.Ticker(f"{symbol}.TW")
        hist = ticker.history(period="3mo") # 抓長一點算 MA60 也行，這邊只用 MA20
        
        if len(hist) < 20: 
            return 0, 0, 0, "⚪ 資料不足"
            
        close = hist['Close'].iloc[-1]
        ma20 = hist['Close'].rolling(20).mean().iloc[-1]
        
        # 取得該股今日的 V32 攻擊分 (若在榜內)
        atk_score = ref_score_map.get(symbol, 0)
        
        # --- 診斷邏輯 ---
        # 1. 生死線判斷 (Price Action)
        if close < ma20:
            signal = "🔴 破線 (停損)"
        # 2. 動能判斷 (V32 Score)
        elif atk_score == 0:
            signal = "🟡 榜外 (觀察)" # 股價在均線上，但沒攻擊力
        elif atk_score < 60:
            signal = "🟡 轉弱 (注意)" # 有分數但很低
        else:
            signal = "🟢 續抱 (強勢)" # 均線上且有攻擊分
            
        return close, ma20, atk_score, signal
        
    except:
        return 0, 0, 0, "⚪ 連線失敗"

# --- 資料載入 (主榜單) ---
@st.cache_data(ttl=600) # 10分鐘快取
def load_v32_data():
    url = f"https://raw.githubusercontent.com/{REPO_KEY}/main/v32_recommend.csv"
    try:
        df = pd.read_csv(url)
        # 清洗代號
        code_col = next((c for c in ['代碼', '代號', 'Code', 'Symbol'] if c in df.columns), None)
        if code_col:
            df[code_col] = df[code_col].astype(str).str.strip()
            df = df.rename(columns={code_col: '代號'})
        return df
    except:
        return pd.DataFrame()

# --- 主程式 ---
def main():
    st.title("⚔️ V32 戰情室")
    st.caption(f"系統時間: {get_taiwan_time()} (UTC+8)")
