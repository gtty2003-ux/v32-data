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
    page_title="V32 戰情室 (Dual Core)",
    layout="wide",
    page_icon="⚔️"
)

# --- 全域變數 ---
DATA_REPO = "gtty2003-ux/v32-auto-updater" 
DATA_FILE = "v32_dataset.csv"
HOLDING_REPO = "gtty2003-ux/v32-data"
HOLDINGS_FILE = "holdings.csv"

# --- 樣式設定 ---
st.markdown("""
    <style>
    .stDataFrame thead tr th {background-color: #ffebee !important; color: #b71c1c !important; font-weight: bold;}
    div[data-testid="stMetricValue"] {font-size: 24px; font-weight: bold;}
    .stButton>button {width: 100%; border-radius: 5px; font-weight: bold;}
    div[data-testid="stCaptionContainer"] {text-align: right; align-self: center; padding-top: 10px;}
    </style>
    """, unsafe_allow_html=True)

# --- 工具函數 ---
def get_taiwan_time_str(timestamp=None):
    """將 timestamp 轉為台灣時間字串"""
    tz = pytz.timezone('Asia/Taipei')
    if timestamp:
        # 將 epoch 時間視為 UTC，然後轉台灣時間
        dt = datetime.fromtimestamp(timestamp, pytz.utc).astimezone(tz)
    else:
        dt = datetime.now(tz)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def get_taiwan_time_iso():
    """取得台灣時間 (用於存檔)"""
    return datetime.now(pytz.timezone('Asia/Taipei')).strftime("%Y-%m-%d %H:%M:%S")

def make_pastel_cmap(hex_color):
    return mcolors.LinearSegmentedColormap.from_list("pastel_cmap", ["#ffffff", hex_color])

cmap_pastel_red = make_pastel_cmap("#ef9a9a")
cmap_pastel_blue = make_pastel_cmap("#90caf9")
cmap_pastel_green = make_pastel_cmap("#a5d6a7")

def color_surplus(val):
    if not isinstance(val, (int, float)): return ''
    return 'color: #d32f2f; font-weight: bold;' if val > 0 else ('color: #388e3c; font-weight: bold;' if val < 0 else 'color: black')

def color_action(val):
    val_str = str(val)
    if "🔴" in val_str or "停損" in val_str:
        return 'color: #ffffff; background-color: #d32f2f; font-weight: bold; padding: 5px; border-radius: 5px;' # 紅底白字
    elif "🟡" in val_str or "停利" in val_str:
        return 'color: #000000; background-color: #ffeb3b; font-weight: bold; padding: 5px; border-radius: 5px;' # 黃底黑字
    elif "🟢" in val_str or "續抱" in val_str:
        return 'color: #ffffff; background-color: #2e7d32; font-weight: bold; padding: 5px; border-radius: 5px;' # 綠底白字
    return ''

def color_risk(val):
    """地雷分顏色邏輯"""
    if not isinstance(val, (int, float)): return ''
    if val >= 60:
        return 'color: #ffffff; background-color: #d32f2f; font-weight: bold;' # 紅底 (高風險)
    elif val >= 30:
        return 'color: #000000; background-color: #ffeb3b; font-weight: bold;' # 黃底 (警戒)
    return 'color: #1b5e20; font-weight: bold;' # 綠字 (安全)

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
            numeric_cols = ['ClosingPrice', 'OpeningPrice', 'HighestPrice', 'LowestPrice', 'TradeVolume']
            for c in numeric_cols:
                if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            return df
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

# --- V32 運算邏輯 ---
def calculate_v32_score(df_group):
    if len(df_group) < 60: return None 
    df = df_group.sort_values('Date').reset_index(drop=True)
    close, vol, high, open_p = df['ClosingPrice'], df['TradeVolume'], df['HighestPrice'], df['OpeningPrice']
    
    ma5, ma20, ma60 = close.rolling(5).mean(), close.rolling(20).mean(), close.rolling(60).mean()
    delta = close.diff()
    gain, loss = (delta.where(delta > 0, 0)).rolling(14).mean(), (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + (gain / loss)))
    
    exp1, exp2 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    macd, signal = (exp1 - exp2), (exp1 - exp2).ewm(span=9, adjust=False).mean()
    
    vol_ma5, vol_ma20 = vol.rolling(5).mean(), vol.rolling(20).mean()
    high_20 = high.rolling(20).max()
    
    i = -1 
    c_now, m20_now, r_now, v_now = close.iloc[i], ma20.iloc[i], rsi.iloc[i], vol.iloc[i]
    if pd.isna(c_now) or c_now == 0: return None
    
    t_score = 60
    if c_now > m20_now: t_score += 5
    if m20_now > ma20.iloc[i-1]: t_score += 5
    if ma5.iloc[i] > m20_now > ma60.iloc[i]: t_score += 10
    if r_now > 50: t_score += 5
    if r_now > 70: t_score += 5
    if macd.iloc[i] > signal.iloc[i]: t_score += 5
    if c_now > high_20.iloc[i-1]: t_score += 10
    
    v_score = 60
    if v_now > vol_ma20.iloc[i]: v_score += 10
    if v_now > vol_ma5.iloc[i]: v_score += 10
    if c_now > open_p.iloc[i] and v_now > vol.iloc[i-1]: v_score += 15
    if v_now > vol_ma20.iloc[i] * 1.5: v_score += 5
    
    return {
        '技術分': min(100, t_score), 
        '量能分': min(100, v_score), 
        '攻擊分': (min(100, t_score) * 0.7) + (min(100, v_score) * 0.3), 
        '收盤': c_now,
        '20MA': m20_now 
    }

@st.cache_data(ttl=1800)
def process_data():
    raw_df = load_data_from_github()
    if raw_df.empty: return pd.DataFrame(), pd.DataFrame(), "無法讀取數據"
    
    # --- 股票過濾邏輯 ---
    raw_df['Code_Str'] = raw_df['Code'].astype(str).str.strip()
    raw_df['Name_Str'] = raw_df['Name'].astype(str).str.strip()
    mask_common = (raw_df['Code_Str'].str.len() == 4) & (raw_df['Code_Str'].str.isdigit())
    mask_exclude = (
        raw_df['Code_Str'].str.startswith('28') |
        raw_df['Code_Str'].str.startswith('00') |
        raw_df['Code_Str'].str.startswith('91') |
        raw_df['Code_Str'].str.startswith('02') |
        raw_df['Name_Str'].str.contains('KY')   |
        raw_df['Name_Str'].str.contains('創')
    )
    raw_df = raw_df[mask_common & ~mask_exclude]

    results = []
    for code, group in raw_df.groupby('Code'):
        res = calculate_v32_score(group)
        if res:
            res.update({'代號': code, '名稱': group['Name'].iloc[-1]})
            results.append(res)
    return pd.DataFrame(results), raw_df, None

# --- 強化的即時報價模組 ---
def fetch_price_twse(code):
    try:
        stock = twstock.Realtime(code)
        if stock.realtime.get('success') and stock.realtime.get('latest_trade_price'):
            price = float(stock.realtime['latest_trade_price'])
            if price > 0: return price
        if stock.realtime.get('open'):
             return float(stock.realtime['open'])
        return None
    except:
        return None

def fetch_price_yahoo(code):
    try:
        ticker = yf.Ticker(f"{code}.TW")
        info = ticker.info
        if 'currentPrice' in info: return float(info['currentPrice'])
        if 'regularMarketPrice' in info: return float(info['regularMarketPrice'])
        data = ticker.history(period="1d", interval="1m")
        if not data.empty:
            return float(data['Close'].iloc[-1])
        return None
    except:
        return None

def fetch_price_google_backup(code):
    try:
        data = yf.download(f"{code}.TW", period="1d", interval="1m", progress=False)
        if not data.empty:
             return float(data['Close'].iloc[-1])
        return None
    except:
        return None

def get_realtime_quotes_robust(code_list):
    if not code_list: return {}
    clean_codes = [str(c).strip().split('.')[0] for c in code_list]
    realtime_data = {}
    progress_bar = st.progress(0)
    total = len(clean_codes)
    for idx, code in enumerate(clean_codes):
        price = None
        price = fetch_price_twse(code)
        if price is None: price = fetch_price_yahoo(code)
        if price is None: price = fetch_price_google_backup(code)
        if price is not None:
            realtime_data[code] = {'即時價': round(price, 2)}
        progress_bar.progress((idx + 1) / total)
    progress_bar.empty()
    return realtime_data

# --- 地雷股分數計算 (新模組) ---
@st.cache_data(ttl=86400) # 財報數據一天更新一次即可
def calculate_risk_factors(code):
    """計算四大地雷指標: 現金流、資產膨脹、償債能力、籌碼質押"""
    r1, r2, r3, r4 = 0, 0, 0, 0
    try:
        # 1. Yahoo Finance (R1, R2, R3)
        stock = yf.Ticker(f"{code}.TW")
        # 嘗試取得季度財報
        fin = stock.quarterly_financials
        bs = stock.quarterly_balance_sheet
        cf = stock.quarterly_cashflow
        
        # 備援：若無季度則取年度
        if fin.empty: fin = stock.financials
        if bs.empty: bs = stock.balance_sheet
        if cf.empty: cf = stock.cashflow
        
        # R1: 現金流品質
        if not fin.empty and not cf.empty:
            try:
                ni = fin.loc['Net Income'].iloc[0]
                ocf_key = next((k for k in cf.index if 'Operating' in k), None)
                ocf = cf.loc[ocf_key].iloc[0] if ocf_key else ni 
                
                if ocf < ni:
                    ratio = (ni - ocf) / abs(ni) if ni != 0 else 0
                    r1 = min(30, ratio * 30) 
            except: pass

        # R2: 資產膨脹
        if not fin.empty and not bs.empty and len(fin.columns) > 1:
            try:
                rev_now = fin.loc['Total Revenue'].iloc[0]
                rev_prev = fin.loc['Total Revenue'].iloc[1]
                rev_yoy = (rev_now - rev_prev) / rev_prev if rev_prev != 0 else 0
                
                inv_key = next((k for k in bs.index if 'Inventory' in k), None)
                inv_yoy = 0
                if inv_key:
                    inv_now = bs.loc[inv_key].iloc[0]
                    inv_prev = bs.loc[inv_key].iloc[1]
                    inv_yoy = (inv_now - inv_prev) / inv_prev if inv_prev != 0 else 0
                
                rec_key = next((k for k in bs.index if 'Receivables' in k), None)
                rec_yoy = 0
                if rec_key:
                    rec_now = bs.loc[rec_key].iloc[0]
                    rec_prev = bs.loc[rec_key].iloc[1]
                    rec_yoy = (rec_now - rec_prev) / rec_prev if rec_prev != 0 else 0
                
                gap = max(inv_yoy, rec_yoy) - rev_yoy
                if gap > 0:
                    r2 = min(20, gap * 50)
            except: pass
        
        # R3: 償債壓力
        if not bs.empty:
            try:
                cash_key = next((k for k in bs.index if 'Cash' in k), None)
                rec_key = next((k for k in bs.index if 'Receivables' in k), None)
                liab_key = next((k for k in bs.index if 'Current Liabilities' in k), None)
                
                cash = bs.loc[cash_key].iloc[0] if cash_key else 0
                rec = bs.loc[rec_key].iloc[0] if rec_key else 0
                liab = bs.loc[liab_key].iloc[0] if liab_key else 1
                
                qr = (cash + rec) / liab
                
                if qr <= 0.5: r3 = 20
                elif qr >= 1.5: r3 = 0
                else: r3 = 20 * (1.5 - qr)
            except: pass

        # R4: 籌碼壓力 (質押比)
        try:
            url = f"https://histock.tw/stock/large.aspx?no={code}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(url, headers=headers)
            dfs = pd.read_html(io.StringIO(r.text))
            
            pledge_ratio = 0
            for df in dfs:
                if '質押比例' in df.columns:
                    val = str(df['質押比例'].iloc[0]).replace('%', '')
                    pledge_ratio = float(val)
                    break
            
            r4 = min(30, pledge_ratio * 0.4)
        except: pass

        total = r1 + r2 + r3 + r4
        detail_str = f"現:{int(r1)} 膨:{int(r2)} 償:{int(r3)} 質:{int(r4)}"
        return total, detail_str
        
    except Exception:
        return 0, "無數據"

def get_risk_analysis_batch(code_list):
    """批次執行地雷檢測"""
    risk_data = {}
    progress_bar = st.progress(0)
    total = len(code_list)
    
    for idx, code in enumerate(code_list):
        score, detail = calculate_risk_factors(code)
        risk_data[code] = {'地雷分': score, '風險細節': detail}
        progress_bar.progress((idx + 1) / total)
        time.sleep(0.5)
        
    progress_bar.empty()
    return pd.DataFrame.from_dict(risk_data, orient='index').reset_index().rename(columns={'index': '代號'})

# --- 籌碼分析 (光速批次版) ---
def get_chip_analysis(symbol_list):
    # 1. 設定目標日期 (如果現在是下午 3 點前，資料可能還沒出來，就抓昨天)
    now = datetime.now(pytz.timezone('Asia/Taipei'))
    target_date = now
    
    # 如果是下午 3 點前，通常證交所還沒更新，直接用昨天
    if now.hour < 15:
        target_date = now - timedelta(days=1)
    
    date_str = target_date.strftime('%Y%m%d')
    
    # 2. 嘗試抓取全市場資料 (重試機制: 如果今天沒資料，就往回找一天，最多找三天)
    df_bulk = pd.DataFrame()
    
    # 建立進度條給使用者看
    p_bar = st.progress(0, text="正在連線證交所資料庫...")
    
    for days_back in range(3):
        query_date = (target_date - timedelta(days=days_back)).strftime('%Y%m%d')
        url = f"https://www.twse.com.tw/rwd/zh/fund/T86?date={query_date}&selectType=ALL&response=json"
        
        try:
            # 這是抓取全市場所有股票，只發送一次 Request，速度極快
            r = requests.get(url, timeout=5)
            data = r.json()
            
            if data['stat'] == 'OK':
                # 欄位對應 (證交所 JSON 欄位可能會變，這裡用位置抓取比較保險)
                # 通常：代號(0), 名稱(1), 外資買賣超(19), 投信買賣超(15) <- 需視實際回傳調整，以下用欄位名過濾
                cols = data['fields']
                raw_data = data['data']
                df_bulk = pd.DataFrame(raw_data, columns=cols)
                
                # 成功抓到數據，跳出迴圈
                # st.toast(f"已取得 {query_date} 籌碼資料", icon="✅")
                break 
            else:
                pass # 該日無資料 (可能是假日)，繼續迴圈找前一天
                
        except Exception as e:
            print(f"Fetch TWSE Error: {e}")
            pass
        
        time.sleep(1) # 避免過度請求

    p_bar.progress(50, text="正在分析主力動向...")

    # 3. 如果完全抓不到資料 (例如過年連假)，回傳空值
    if df_bulk.empty:
        p_bar.empty()
        return pd.DataFrame([
            {'代號': s, '投信(張)': 0, '外資(張)': 0, '主力動向': '❌ 無資料(連線失敗)'} 
            for s in symbol_list
        ])

    # 4. 資料整理與過濾
    chip_data = []
    
    # 找出對應欄位名稱 (證交所欄位名稱很長)
    col_stock_id = next((c for c in df_bulk.columns if '證券代號' in c), None)
    col_trust = next((c for c in df_bulk.columns if '投信' in c and '買賣超股數' in c), None)
    col_foreign = next((c for c in df_bulk.columns if '外陸資' in c and '買賣超股數' in c), None)

    if not (col_stock_id and col_trust and col_foreign):
        p_bar.empty()
        return pd.DataFrame([{'代號': s, '主力動向': '❌ 格式錯誤'} for s in symbol_list])

    # 轉成字典加速查詢
    # 證交所數字會有逗號 (1,000)，需要移除並轉 int
    def parse_val(val):
        try:
            return int(str(val).replace(',', '')) // 1000 # 換算成張數
        except:
            return 0

    # 建立快速查詢表 (Lookup Table)
    lookup = {}
    for _, row in df_bulk.iterrows():
        sid = str(row[col_stock_id]).strip()
        lookup[sid] = {
            'trust': parse_val(row[col_trust]),
            'foreign': parse_val(row[col_foreign])
        }

    # 5. 比對使用者的股票清單
    for symbol in symbol_list:
        symbol = str(symbol).strip()
        if symbol in lookup:
            t_buy = lookup[symbol]['trust']
            f_buy = lookup[symbol]['foreign']
            
            # 判斷動向標籤 (邏輯同前)
            status_str = ""
            if t_buy > 0: status_str += "🔴 投信買 "
            elif t_buy < 0: status_str += "🟢 投信賣 "
            
            if f_buy > 1000: status_str += "🔥 外資大買 "
            elif f_buy < -1000: status_str += "🧊 外資倒貨 "
            
            if t_buy > 0 and f_buy > 0: tag = "🚀 土洋合買"
            elif t_buy > 0 and f_buy < 0: tag = "⚔️ 土洋對作(信)"
            elif t_buy < 0 and f_buy > 0: tag = "⚔️ 土洋對作(外)"
            elif t_buy < 0 and f_buy < 0: tag = "☠️ 主力棄守"
            else: tag = "🟡 一般輪動"
            
            final_status = f"{tag} | {status_str}" if status_str else tag
            
            chip_data.append({
                '代號': symbol, 
                '投信(張)': t_buy, 
                '外資(張)': f_buy, 
                '主力動向': final_status
            })
        else:
            # 該股票不在今日證交所清單 (可能暫停交易或 ETF 類別不同)
            chip_data.append({'代號': symbol, '投信(張)': 0, '外資(張)': 0, '主力動向': '🟡 無交易/無數據'})

    p_bar.progress(100, text="完成！")
    time.sleep(0.5)
    p_bar.empty()
    
    return pd.DataFrame(chip_data)
# --- 庫存管理 ---
def load_holdings():
    try:
        g = Github(st.secrets["general"]["GITHUB_TOKEN"])
        df = pd.read_csv(g.get_repo(HOLDING_REPO).get_contents(HOLDINGS_FILE).download_url)
        df['股票代號'] = df['股票代號'].astype(str).apply(lambda x: x.split('.')[0] if '.' in x else x)
        return df[['股票代號', '買入均價', '持有股數']]
    except: return pd.DataFrame(columns=["股票代號", "買入均價", "持有股數"])

def save_holdings(df):
    try:
        g = Github(st.secrets["general"]["GITHUB_TOKEN"])
        repo = g.get_repo(HOLDING_REPO)
        csv_content = df.to_csv(index=False)
        contents = repo.get_contents(HOLDINGS_FILE)
        repo.update_file(contents.path, f"Update {get_taiwan_time_iso()}", csv_content, contents.sha)
    except: pass

# --- Tab 1 & 2 表格渲染 ---
def display_v32_tables(df, price_limit, suffix):
    filtered = df[(df['收盤'] <= price_limit) & (df['攻擊分'] >= 86) & (df['攻擊分'] <= 92)].sort_values('攻擊分', ascending=False)
    if filtered.empty: return st.warning("無符合標的")

    df_s_pre = filtered[(filtered['攻擊分'] >= 90) & (filtered['攻擊分'] <= 92)].head(10)
    df_a_pre = filtered[(filtered['攻擊分'] >= 88) & (filtered['攻擊分'] < 90)].head(10)
    df_b_pre = filtered[(filtered['攻擊分'] >= 86) & (filtered['攻擊分'] < 88)].head(10)
    target_codes = pd.concat([df_s_pre, df_a_pre, df_b_pre])['代號'].tolist()

    # --- 功能按鈕區 ---
    c_scan, c_risk, c_update, c_info = st.columns([1, 1, 1, 1.5])
    
    # 定義 Session State Key，確保不同頁面的資料不打架
    chip_key = f"chip_data_{suffix}"
    risk_key = f"risk_data_{suffix}"

    # 按鈕 1: 籌碼掃描
    with c_scan:
        if st.button(f"🚀 籌碼掃描", key=f"scan_{suffix}"):
            chip_df = get_chip_analysis(target_codes)
            st.session_state[chip_key] = chip_df # 存入 Session

    # 按鈕 2: 地雷檢測
    with c_risk:
        if st.button(f"💣 地雷檢測", key=f"risk_{suffix}"):
            with st.spinner("正在進行深度財報與質押掃描..."):
                risk_df = get_risk_analysis_batch(target_codes)
                st.session_state[risk_key] = risk_df # 存入 Session

    # 按鈕 3: 更新即時價
    with c_update:
        now = time.time()
        time_diff = now - st.session_state.get('last_update_time', 0)
        btn_label = "🔄 更新即時價"
        btn_disabled = False
        if time_diff < 60:
            btn_label = f"⏳ 冷卻 ({int(60 - time_diff)}s)"
            btn_disabled = True
            
        if st.button(btn_label, disabled=btn_disabled, key=f"update_{suffix}", type="primary"):
            with st.spinner(f"🚀 同步 Top {len(target_codes)} 檔股價..."):
                fresh_quotes = get_realtime_quotes_robust(target_codes)
                current_quotes = st.session_state.get('realtime_quotes', {})
                current_quotes.update(fresh_quotes)
                st.session_state['realtime_quotes'] = current_quotes
                st.session_state['last_update_time'] = time.time()
                st.toast(f"✅ 更新成功！", icon="🔄")
                time.sleep(1)
                st.rerun()

    with c_info:
        if st.session_state.get('last_update_time', 0) > 0:
            tw_time = get_taiwan_time_str(st.session_state['last_update_time'])
            st.caption(f"🕒 更新: {tw_time}")

    # --- 資料合併邏輯 (從 Session 讀取並合併，確保資料共存) ---
    
    # 1. 合併籌碼資料 (如果有的話)
    if chip_key in st.session_state:
        filtered = pd.merge(filtered, st.session_state[chip_key], on='代號', how='left')

    # 2. 合併地雷資料 (如果有的話)
    if risk_key in st.session_state:
        filtered = pd.merge(filtered, st.session_state[risk_key], on='代號', how='left')

    # 3. 合併即時報價
    saved_quotes = st.session_state.get('realtime_quotes', {})
    filtered['即時價'] = filtered['代號'].map(lambda x: saved_quotes.get(x, {}).get('即時價', np.nan))
    filtered['即時價'] = filtered['即時價'].fillna(filtered['收盤'])

    # --- 表格顯示 ---
    base_cols = ['代號','名稱','即時價','技術分','量能分','攻擊分']
    # 動態欄位檢查
    if '主力動向' in filtered.columns: base_cols += ['主力動向', '投信(張)', '外資(張)']
    if '地雷分' in filtered.columns: base_cols += ['地雷分', '風險細節']

    fmt = {'即時價':'{:.2f}', '攻擊分':'{:.1f}', '技術分':'{:.0f}', '量能分':'{:.0f}', '外資(張)': '{:,.0f}', '投信(張)': '{:,.0f}', '地雷分':'{:.0f}'}

    for title, score_range in [
        ("👑 S 級主力區 (90-92分)", (90, 92)),
        ("🚀 A 級蓄勢區 (88-90分)", (88, 90)),
        ("👀 B 級觀察區 (86-88分)", (86, 88))
    ]:
        st.subheader(title)
        sub = filtered[(filtered['攻擊分'] >= score_range[0]) & (filtered['攻擊分'] <= score_range[1])].head(10)
        if not sub.empty:
            st.dataframe(sub[base_cols].style.format(fmt)
                         .background_gradient(subset=['攻擊分'], cmap=cmap_pastel_red, vmin=86, vmax=92)
                         .background_gradient(subset=['技術分'], cmap=cmap_pastel_blue, vmin=60, vmax=100)
                         .background_gradient(subset=['量能分'], cmap=cmap_pastel_green, vmin=60, vmax=100)
                         .map(color_risk, subset=['地雷分'] if '地雷分' in sub.columns else []), 
                         hide_index=True, use_container_width=True)
        else: st.caption("暫無標的")
        st.divider()

# --- 主程式 ---
def main():
    st.title("⚔️ V32 戰情室 (Dual Core)")
    if 'inventory' not in st.session_state: st.session_state['inventory'] = load_holdings()
    
    if 'realtime_quotes' not in st.session_state: st.session_state['realtime_quotes'] = {}
    if 'last_update_time' not in st.session_state: st.session_state['last_update_time'] = 0
    
    with st.spinner("讀取核心資料..."):
        v32_df, raw_df, err = process_data()
    
    tab_80, tab_50, tab_inv = st.tabs(["💰 80元以下推薦", "🪙 50元以下推薦", "💼 庫存管理"])

    with tab_80:
        if not v32_df.empty: display_v32_tables(v32_df.copy(), 80, "80")

    with tab_50:
        if not v32_df.empty: display_v32_tables(v32_df.copy(), 50, "50")

    with tab_inv:
        st.subheader("📝 庫存交易管理")
        
        col_btn, col_info = st.columns([1, 4])
        with col_btn:
            now = time.time()
            time_diff = now - st.session_state.get('last_update_time', 0)
            btn_label = "🔄 更新即時股價"
            btn_disabled = False
            
            if time_diff < 60:
                btn_label = f"⏳ 冷卻中 ({int(60 - time_diff)}s)"
                btn_disabled = True
            
            if st.button(btn_label, disabled=btn_disabled, type="primary", key="btn_inv_update"):
                if not st.session_state['inventory'].empty:
                    with st.spinner("🚀 同步庫存股價..."):
                        codes = st.session_state['inventory']['股票代號'].tolist()
                        fresh_quotes = get_realtime_quotes_robust(codes)
                        
                        current_quotes = st.session_state.get('realtime_quotes', {})
                        current_quotes.update(fresh_quotes)
                        st.session_state['realtime_quotes'] = current_quotes
                        st.session_state['last_update_time'] = time.time()
                        
                        st.toast(f"✅ 更新成功！", icon="💼")
                        time.sleep(1)
                        st.rerun()
        
        with col_info:
            if st.session_state.get('last_update_time', 0) > 0:
                tw_time = get_taiwan_time_str(st.session_state['last_update_time'])
                st.caption(f"🕒 台灣時間最後更新: {tw_time}")

        name_map = dict(zip(v32_df['代號'], v32_df['名稱'])) if not v32_df.empty else {}
        score_map = v32_df.set_index('代號')['攻擊分'].to_dict() if not v32_df.empty else {}
        if '20MA' in v32_df.columns:
            ma20_map = v32_df.set_index('代號')['20MA'].to_dict()
        else:
            ma20_map = {code: 0 for code in v32_df['代號']}

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 📥 **買入**")
            edited_buy = st.data_editor(pd.DataFrame([{"股票代號": "", "持有股數": 1000, "買入均價": 0.0}]), num_rows="dynamic", key="buy_in", hide_index=True)
        with c2:
            st.markdown("##### 📤 **賣出**")
            edited_sell = st.data_editor(pd.DataFrame([{"股票代號": "", "持有股數": 1000}]), num_rows="dynamic", key="sell_out", hide_index=True)
        
        if st.button("💾 執行交易", type="primary"):
            inv = st.session_state['inventory'].copy()
            for _, r in edited_buy.iterrows():
                code = str(r['股票代號']).strip().split('.')[0]
                if code and r['持有股數'] > 0:
                    match = inv[inv['股票代號'] == code]
                    if not match.empty:
                        idx = match.index[0]
                        total_shares = inv.at[idx, '持有股數'] + r['持有股數']
                        inv.at[idx, '買入均價'] = round(((inv.at[idx, '買入均價'] * inv.at[idx, '持有股數']) + (r['買入均價'] * r['持有股數'])) / total_shares, 2)
                        inv.at[idx, '持有股數'] = total_shares
                    else:
                        inv = pd.concat([inv, pd.DataFrame([{'股票代號': code, '持有股數': r['持有股數'], '買入均價': r['買入均價']}])], ignore_index=True)
            for _, r in edited_sell.iterrows():
                code = str(r['股票代號']).strip().split('.')[0]
                if code:
                    inv = inv[~((inv['股票代號'] == code) & (inv['持有股數'] <= r['持有股數']))]
                    mask = inv['股票代號'] == code
                    if mask.any(): inv.loc[mask, '持有股數'] -= r['持有股數']
            st.session_state['inventory'] = inv
            save_holdings(inv)
            st.rerun()

        st.divider()
        if not st.session_state['inventory'].empty:
            inv_df = st.session_state['inventory'].copy()
            saved_quotes = st.session_state.get('realtime_quotes', {})
            
            res = []
            for _, r in inv_df.iterrows():
                code = str(r['股票代號'])
                curr = saved_quotes.get(code, {}).get('即時價', r['買入均價'])
                
                if (curr == 0 or curr == r['買入均價']) and not v32_df.empty:
                     backup_price = v32_df[v32_df['代號']==code]['收盤'].values
                     if len(backup_price) > 0:
                         curr = backup_price[0]

                buy_price = r['買入均價']
                qty = r['持有股數']
                
                pl = (curr - buy_price) * qty
                roi = (pl / (buy_price * qty) * 100) if buy_price > 0 else 0
                
                sc = score_map.get(code, 0)
                ma20 = ma20_map.get(code, 0)
                
                if curr < ma20:
                    action = f"🔴 停損/清倉 (破月線 {ma20:.1f})"
                elif sc >= 60:
                    action = "🟢 續抱 (動能強)"
                else:
                    action = "🟡 停利/減碼 (動能熄火)"

                res.append({
                    '代號': code, '名稱': name_map.get(code, code), 
                    '持有張數': int(qty // 1000), 
                    '買入均價': buy_price, '即時價': curr, 
                    '損益': pl, '報酬率%': roi, 
                    '攻擊分': sc, '建議操作': action
                })
            
            df_res = pd.DataFrame(res)
            c1, c2, c3 = st.columns(3)
            c1.metric("總成本", f"${(df_res['買入均價']*(inv_df['持有股數'])).sum():,.0f}")
            c2.metric("總損益", f"${df_res['損益'].sum():,.0f}", delta=f"{df_res['損益'].sum():,.0f}")
            c3.metric("總市值", f"${(df_res['即時價']*(inv_df['持有股數'])).sum():,.0f}")
            
            st.dataframe(
                df_res[['代號', '名稱', '持有張數', '買入均價', '即時價', '攻擊分', '報酬率%', '損益', '建議操作']].style
                .format({'買入均價':'{:.2f}', '即時價':'{:.2f}', '損益':'{:+,.0f}', '報酬率%':'{:+.2f}%', '攻擊分':'{:.1f}'})
                .map(color_surplus, subset=['損益','報酬率%'])
                .map(color_action, subset=['建議操作']), 
                use_container_width=True, hide_index=True
            )
        else: st.info("目前無庫存。")

if __name__ == "__main__":
    main()
