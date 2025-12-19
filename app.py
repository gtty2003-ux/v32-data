import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import pytz
import yfinance as yf
from github import Github 
import time

# --- 設定頁面資訊 ---
st.set_page_config(
    page_title="V32 戰情室 (Slope Logic)",
    layout="wide",
    page_icon="📈"
)

# --- 樣式設定 ---
st.markdown("""
    <style>
    .stDataFrame thead tr th {
        background-color: #C8E6C9 !important;
        color: #000000 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 全域變數 ---
REPO_KEY = "gtty2003-ux/v32-data"
FILE_PATH = "holdings.csv"

# --- 工具函數 ---
def get_taiwan_time():
    utc_now = datetime.utcnow()
    tw_time = utc_now.replace(tzinfo=pytz.utc).astimezone(pytz.timezone('Asia/Taipei'))
    return tw_time.strftime("%Y-%m-%d %H:%M:%S")

def color_surplus(val):
    if val > 0: return 'color: red'
    elif val < 0: return 'color: green'
    return 'color: black'

@st.cache_data(ttl=86400)
def fetch_name_from_web(symbol):
    try:
        t = yf.Ticker(f"{symbol}.TW")
        return t.info.get('shortName') or t.info.get('longName') or symbol
    except:
        return symbol

# --- 🔥 斜率計算函數 (核心新增) ---
def get_normalized_slope(series):
    """
    計算序列的歸一化斜率 (Normalized Slope)
    將數據以第一天為基準 (Base 100)，計算每日平均變動百分比
    """
    if len(series) < 2: return 0
    
    y = series.values
    # 歸一化：將數列變成以 100 為起點，這樣斜率代表「每日漲幅%」
    # 避免除以 0 錯誤
    start_val = y[0] if y[0] != 0 else 1
    y_norm = (y / start_val) * 100
    
    x = np.arange(len(y))
    
    # 使用 numpy 的多項式擬合 (1次 = 線性回歸) 取得斜率
    slope, intercept = np.polyfit(x, y_norm, 1)
    return slope

# --- 核心：V32 技術指標運算 (斜率版) ---
def calculate_indicators(hist):
    """
    輸入: 歷史 K 線 (DataFrame)
    輸出: 技術分, 量能分, 趨勢狀態
    """
    if len(hist) < 60: return 0, 0, "Data Insufficient"

    # 1. 準備數據
    close = hist['Close']
    vol = hist['Volume']
    open_p = hist['Open']
    
    # 均線
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    
    # 取最新值
    ma5_now = ma5.iloc[-1]
    ma20_now = ma20.iloc[-1]
    ma60_now = ma60.iloc[-1]

    # RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_now = rsi.iloc[-1]

    # MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_now = macd.iloc[-1]
    signal_now = signal.iloc[-1]

    # 均量
    vol_ma5 = vol.rolling(5).mean().iloc[-1]
    vol_ma20 = vol.rolling(20).mean().iloc[-1]

    # --- 評分開始 (Base Score: 60) ---
    
    # A. 技術分 (Technical)
    t_score = 60
    
    # 1. 趨勢 - MA20 翻揚 (改成斜率連續計分, 滿分 20)
    # 取最近 5 天的 MA20 數據
    ma20_last_5 = ma20.iloc[-5:]
    ma20_slope = get_normalized_slope(ma20_last_5)
    
    # 計分邏輯：
    # 如果 MA20 每天平均上升 0.2% (slope=0.2)，就是非常強的趨勢，得滿分 20
    # 公式：slope * 100 (放大係數) -> 限制在 0~20
    # 範例：slope 0.1 (日升0.1%) -> 10分 | slope 0.05 -> 5分 | slope <= 0 -> 0分
    ma20_score = min(20, max(0, ma20_slope * 100))
    t_score += ma20_score
    
    # 均線排列額外加分 (保留)
    if ma5_now > ma20_now and ma20_now > ma60_now: 
        t_score += 10
    
    # 2. 型態 (Structure) - 攻擊力道 (改成斜率連續計分, 滿分 30)
    # 觀察最近 5 天的收盤價走勢
    close_last_5 = close.iloc[-5:]
    price_slope = get_normalized_slope(close_last_5)
    
    # 計分邏輯：
    # 如果股價每天平均上漲 1.0% (slope=1.0)，代表攻擊型態明確，得滿分 30
    # 公式：slope * 30 (放大係數) -> 限制在 0~30
    # 範例：slope 1.0 (日漲1%) -> 30分 | slope 0.5 (緩漲) -> 15分 | slope <= 0 (盤整/跌) -> 0分
    struct_score = min(30, max(0, price_slope * 30))
    t_score += struct_score

    # 3. 動能 (Momentum) - 輔助加分 (總分不超過100)
    # 由於上面已經分配了 20+10+30 = 60 分的加分空間，加上底分 60，這裡做微調
    # 我們將動能視為「額外獎勵」，但需控制總分
    
    if rsi_now > 50: t_score += 5             # RSI 強勢
    if macd_now > signal_now: t_score += 5    # MACD 金叉狀態
    
    # 修正總分上限
    # 由於底分60 + MA20(20) + 排列(10) + 型態(30) + 動能(10) = 130
    # 我們這裡做一個動態調整，讓滿分剛好 100
    # 將底分降為 40，讓斜率的影響力更大
    
    # --- 重新加總技術分 (Base 40) ---
    final_tech_score = 40  # 基礎分
    final_tech_score += ma20_score     # 0~20 (趨勢斜率)
    final_tech_score += 10 if (ma5_now > ma20_now and ma20_now > ma60_now) else 0 # 0~10 (排列)
    final_tech_score += struct_score   # 0~30 (攻擊斜率)
    final_tech_score += 10 if (rsi_now > 50) else 0 # 0~10 (RSI)
    final_tech_score += 10 if (macd_now > signal_now) else 0 # 0~10 (MACD)
    
    # 此時滿分為 40+20+10+30+10+10 = 120，稍微縮放一下或直接截斷
    final_tech_score = min(100, final_tech_score)

    # B. 量能分 (Volume)
    v_score = 60
    
    current_vol = vol.iloc[-1]
    # 1. 均量突破
    if current_vol > vol_ma20: v_score += 10      # 大於月均量
    if current_vol > vol_ma5: v_score += 10       # 大於週均量 (攻擊量)
    
    # 2. 量價配合
    is_red = close.iloc[-1] > open_p.iloc[-1]     # 收紅
    vol_increase = current_vol > vol.iloc[-2]     # 量增
    if is_red and vol_increase: v_score += 15     # 價漲量增 (最理想)
    
    # 3. 爆量檢測
    if current_vol > vol_ma20 * 1.5: v_score += 5 # 放量 1.5 倍

    # 上限防呆
    v_score = min(100, v_score)
    
    # 趨勢標記 (用於篩選)
    # 如果 MA20 斜率是正的，且股價在 MA5 之上，視為 Rising
    trend = "Rising" if (ma20_slope > 0 and close.iloc[-1] > ma5_now) else "Consolidating"
    
    return final_tech_score, v_score, trend

# --- 批次運算引擎 (Streamlit Cache) ---
@st.cache_data(ttl=3600)
def run_v32_engine(ticker_list):
    results = []
    p_bar = st.progress(0)
    status = st.empty()
    total = len(ticker_list)
    
    for i, row in enumerate(ticker_list):
        symbol = str(row['代號'])
        name = str(row.get('名稱', ''))
        
        status.text(f"分析斜率與動能: {symbol} {name} ({i+1}/{total})...")
        p_bar.progress((i + 1) / total)
        
        try:
            # 抓 3 個月資料
            stock = yf.Ticker(f"{symbol}.TW")
            hist = stock.history(period="3mo")
            
            if not hist.empty:
                t_s, v_s, tr = calculate_indicators(hist)
                # 總分權重 (7:3)
                total_s = (t_s * 0.7) + (v_s * 0.3)
                
                results.append({
                    '代號': symbol, '名稱': name,
                    '收盤': hist['Close'].iloc[-1],
                    '成交量': hist['Volume'].iloc[-1],
                    '技術分': t_s, '量能分': v_s, '總分': total_s, '趨勢': tr
                })
            else:
                results.append(row)
        except:
            pass
            
    p_bar.empty()
    status.empty()
    return pd.DataFrame(results)

# --- 資料載入 ---
def load_and_process_data():
    url = f"https://raw.githubusercontent.com/{REPO_KEY}/main/v32_recommend.csv"
    try:
        df = pd.read_csv(url)
        # 欄位處理
        code_col = next((c for c in ['代碼', '代號', 'Code', 'Symbol'] if c in df.columns), None)
        if code_col:
            df[code_col] = df[code_col].astype(str).str.strip()
            df = df.rename(columns={code_col: '代號'})
            
        # 🔥 啟動運算引擎
        processed = run_v32_engine(df[['代號', '名稱']].to_dict('records'))
        return processed, None
    except Exception as e:
        return pd.DataFrame(), str(e)

# --- GitHub 庫存存取 ---
def load_holdings():
    try:
        token = st.secrets["general"]["GITHUB_TOKEN"]
        g = Github(token)
        repo = g.get_repo(REPO_KEY)
        contents = repo.get_contents(FILE_PATH)
        df = pd.read_csv(contents.download_url)
        df['股票代號'] = df['股票代號'].astype(str).str.strip()
        for c in ["股票代號", "買入均價", "持有股數"]:
            if c not in df.columns: df[c] = 0 if c != "股票代號" else ""
        return df[["股票代號", "買入均價", "持有股數"]]
    except:
        return pd.DataFrame(columns=["
