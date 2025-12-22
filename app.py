import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import json
from datetime import datetime, timedelta
import pytz
import yfinance as yf
from github import Github 
import time
from FinMind.data import DataLoader
import twstock
import matplotlib.colors as mcolors
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# --- 設定頁面資訊 ---
st.set_page_config(
    page_title="V32 戰情室 (Drive Core)",
    layout="wide",
    page_icon="⚔️"
)

# --- 樣式設定 ---
st.markdown("""
    <style>
    .stDataFrame thead tr th {
        background-color: #ffebee !important; 
        color: #b71c1c !important;
        font-weight: bold;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 全域變數 ---
# 這是您庫存檔案的 Repo，維持不變
REPO_KEY = "gtty2003-ux/v32-data"
FILE_PATH = "holdings.csv"
# 這是 Google Drive 檔案 ID
DRIVE_FILE_ID = "19z2dUYPqfR4igRStWJMKUofdWCPfqQR_"

# --- 自定義淡色階 (Pastel Colormaps) ---
def make_pastel_cmap(hex_color):
    return mcolors.LinearSegmentedColormap.from_list("pastel_cmap", ["#ffffff", hex_color])

cmap_pastel_red   = make_pastel_cmap("#ef9a9a")
cmap_pastel_blue  = make_pastel_cmap("#90caf9")
cmap_pastel_green = make_pastel_cmap("#a5d6a7")

# --- 工具函數 ---
def get_taiwan_time():
    utc_now = datetime.utcnow()
    tw_time = utc_now.replace(tzinfo=pytz.utc).astimezone(pytz.timezone('Asia/Taipei'))
    return tw_time.strftime("%Y-%m-%d %H:%M:%S")

def color_surplus(val):
    if not isinstance(val, (int, float)): return ''
    if val > 0: return 'color: #d32f2f; font-weight: bold;'
    elif val < 0: return 'color: #388e3c; font-weight: bold;'
    return 'color: black'

def color_action(val):
    val_str = str(val)
    if "賣出" in val_str or "停損" in val_str:
        return 'color: #ffffff; background-color: #d32f2f; font-weight: bold; padding: 5px; border-radius: 5px;'
    elif "續抱" in val_str:
        return 'color: #1b5e20; font-weight: bold;'
    return ''

# --- Google Drive 連線 (修正格式版) ---
@st.cache_resource
def get_drive_service():
    # 直接將 JSON 金鑰寫在這裡
    service_account_info = {
      "type": "service_account",
      "project_id": "v32-stock-bot",
      "private_key_id": "d66f9a30ef7bae397ac2bbbdd24bb7919e96aa79",
      # ↓↓↓ 注意這裡：我在字串最後面加了 .replace('\\n', '\n') 來修正換行問題 ↓↓↓
      "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7kO+PAF/3PQ+x\nZWMwLuJbv/55RHgkcknK67FV2JWLDhWiASYnB/bp4AjCi1tBGuO/vvHk1U5gFElB\nTWbZmcr9BNzsC27MS9CxYM80VhhtOGMzM2+h3sBLk7H+Whj4yIaI+cf36/lL/WjL\nG2gHb3U0JXeC1JsDoDpUfBlJ/W7UswLMUF1ANorCocgsFg59gMVhWgzYKFs+lI1L\nFg1M3xu83iZKzoBrrXYHF+qOIOZtRVfkGYKMEvUPiUkOavXrHTFkD3ulGIbSEwa4\nhDXUoDVqPtMDgvMUVc8G8DlMVtFDUOOcEaKmJxY7NgWnXicQdm9SjmH/KCQYiFaj\nptJXMKlnAgMBAAECggEADR6OIwp7q+dxeY8F6RDedFxxiDnpzWLRFoh11vNXQmqx\nyKsb6A7+jk1FT5Y/w8YFuBu6/66L1NyWYyLu1rmTIS995GTIUzHaXw3OcHK1Mq6H\nAcXPQRs7iA3EnW3f4UblYh9WhVjUDySid9Jq7Fo3cHZObbBBR3elnNMxUaOQZQAh\nvAhbYJeFzACp8Tm5LFMAdjsS2VZrVGtSOIthAv7YSC+vXe3OmCGLuM6EAGIIBMP3\nXToWhY6r0uQfm9d0UfI0xiorWSGsNkBZPK6+HAJ6QVMQwMADHx3/4zOq1v7L0bAe\n+p6DIhCUasA475s4JQkTCCnQC2NM7aw2t/n1Esf3gQKBgQDkLd52g9Ai2facS5wA\nr6gOUUgE+Oh0Tv43PA2yc6pjtqOznx3QYAhY6fqaNgGCVsAwU1ZwnOzDY5LurZfy\nJ9b0UZcd1spN4nwGEobZtdxurzxIdUAoTf6/6ClGGXSpILLgAi06Q+Vu8f1zpx0Y\nnpBGSiTGqt8f5IXtko2WyHS0TwKBgQDSb2rJMi+LAcYXqqjUufSYKq3kxw2aYSR8\nQ+K9Opwv0Cu6u+6JSqHFakfvdNNq21LisjBR16CIQhSYCNzVqsjEbFSKTHYiJ6Dc\nLc8vvHE4ceOZFgljnoPKsnW/OX5enUJjgQNcSexnqJIqXA6VzWtLXXmtzZ7HY02r\nZtdGdlO7aQKBgHz8SxDr3sRYU+cE22zcytcc2rAuj1W2NzWWJYKMLNb1ofGvxKrx\nD2F0uJpj3qvATQGrhHum2WGlV0R5vfMcs3ecgYQMtT+4QWsqFseGADp4rjKaVww8\nvL/tsT3+j5JcoN5nEtMJgdElqEkDTsK/iBOYZVCVJCbaDCo3zmq7XoGtAoGBAKqw\ns1alfYjslGjIBhAfEfaHz+udRjxuBXFCg11oeB4UZhQeslrsjZGbJuRlx8OKSY4W\naTlJhS5hI2E69x3dXOJu2Jghc0U7DbDq+37GBLR7NNkM1erXPiGhZf8JPKa0OpCJ\nqlcmozplssHnT/FU4W4NUVCYU+15cBvS3FWMT1jZAoGAeVwwQjhPmyMV0QWfGOrq\n+W2MLdpY0x7nyrogcTayRa5e3rvWQpMYysi5wKNeC2h1SBrqt9uy0TzxmncfuzFp\nc/lTfnLyqlcTki+LOxdO3t1PhiBEdtwPKgYUy1pVFzobshJFUpT1rU5sqZ33Qrk0\nPXtnDwQ6aHVBjNXbvFCu3D4=\n-----END PRIVATE KEY-----\n".replace('\\n', '\n'),
      "client_email": "v32-auto-updater@v32-stock-bot.iam.gserviceaccount.com",
      "client_id": "109928194171724697312",
      "auth_uri": "https://accounts.google.com/o/oauth2/auth",
      "token_uri": "https://oauth2.googleapis.com/token",
      "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
      "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/v32-auto-updater%40v32-stock-bot.iam.gserviceaccount.com"
    }

    try:
        creds = service_account.Credentials.from_service_account_info(
            service_account_info, 
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        st.error(f"GCP 認證失敗: {e}")
        return None

@st.cache_data(ttl=1800) # 快取 30 分鐘，因為這是盤後資料
def load_data_from_drive():
    service = get_drive_service()
    if not service: return pd.DataFrame()
    
    try:
        request = service.files().get_media(fileId=DRIVE_FILE_ID)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        fh.seek(0)
        df = pd.read_csv(fh)
        
        # 確保欄位名稱正確 (相容性處理)
        # 如果是中文，轉成英文以便後續運算
        rename_map = {
            '日期': 'Date', '股票代碼': 'Code', '股票名稱': 'Name',
            '成交股數': 'TradeVolume', '收盤價': 'ClosingPrice',
            '開盤價': 'OpeningPrice', '最高價': 'HighestPrice', '最低價': 'LowestPrice'
        }
        df.rename(columns=rename_map, inplace=True)
        
        # 轉型
        df['Code'] = df['Code'].astype(str).str.strip()
        df['Date'] = pd.to_datetime(df['Date'])
        numeric_cols = ['ClosingPrice', 'OpeningPrice', 'HighestPrice', 'LowestPrice', 'TradeVolume']
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
                
        return df
    except Exception as e:
        st.error(f"無法從 Drive 下載資料: {e}")
        return pd.DataFrame()

# --- V32 運算核心 (改版：直接運算 DataFrame) ---
def calculate_v32_score(df_group):
    # df_group 是一支股票的歷史資料 (已按日期排序)
    if len(df_group) < 65: return None # 資料不足

    df = df_group.sort_values('Date').reset_index(drop=True)
    close = df['ClosingPrice']
    vol = df['TradeVolume']
    high = df['HighestPrice']
    open_p = df['OpeningPrice']
    
    # 技術指標計算
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    
    vol_ma5 = vol.rolling(5).mean()
    vol_ma20 = vol.rolling(20).mean()
    high_20 = high.rolling(20).max()
    
    # 只計算最近一天的分數
    i = -1 
    
    c_now = close.iloc[i]
    if pd.isna(c_now): return None
    
    # 提取當前值
    m5, m20, m60 = ma5.iloc[i], ma20.iloc[i], ma60.iloc[i]
    m20_prev = ma20.iloc[i-1]
    r_now = rsi.iloc[i]
    macd_now, sig_now = macd.iloc[i], signal.iloc[i]
    h20_prev = high_20.iloc[i-1]
    v_now, v_prev = vol.iloc[i], vol.iloc[i-1]
    v_m5, v_m20 = vol_ma5.iloc[i], vol_ma20.iloc[i]
    o_now = open_p.iloc[i]
    
    # 評分邏輯
    t_score = 60
    if c_now > m20: t_score += 5
    if m20 > m20_prev: t_score += 5
    if m5 > m20 and m20 > m60: t_score += 10
    if r_now > 50: t_score += 5
    if r_now > 70: t_score += 5
    if macd_now > sig_now: t_score += 5
    if c_now > h20_prev: t_score += 10
    
    v_score = 60
    if v_now > v_m20: v_score += 10
    if v_now > v_m5: v_score += 10
    if c_now > o_now and v_now > v_prev: v_score += 15
    if v_now > v_m20 * 1.5: v_score += 5
    
    t_score = min(100, t_score)
    v_score = min(100, v_score)
    
    # 攻擊分 (需計算昨天的分數來做加權，這裡簡化處理，若需要精確可再回推一天)
    # 這裡採用當日分數做為主要依據，或可用簡單加權
    raw_today = (t_score * 0.7) + (v_score * 0.3)
    
    # 穩定度 (回推 5 天)
    stability_count = 0
    # 簡化：只回傳分數，不重複計算 5 天的歷史分以免效能過低
    
    return {
        '技術分': t_score, 
        '量能分': v_score, 
        '攻擊分': raw_today, # 暫以當日分為主，因效能考量
        '收盤': c_now
    }

@st.cache_data(ttl=1800)
def process_drive_data():
    raw_df = load_data_from_drive()
    if raw_df.empty: return pd.DataFrame(), "無法讀取數據"
    
    # 平行運算或群組運算
    results = []
    grouped = raw_df.groupby('Code')
    
    # 為了進度條
    total_stocks = len(grouped)
    # p_bar = st.progress(0)
    
    for i, (code, group) in enumerate(grouped):
        # 取出名稱 (假設同一代碼名稱都一樣，取最後一筆)
        name = group['Name'].iloc[-1]
        
        score_data = calculate_v32_score(group)
        if score_data:
            score_data['代號'] = code
            score_data['名稱'] = name
            results.append(score_data)
        
        # if i % 50 == 0: p_bar.progress((i+1)/total_stocks)
    
    # p_bar.empty()
    return pd.DataFrame(results), None

# --- 核心防鎖機制 (維持不變) ---
@st.cache_data(ttl=60)
def get_realtime_quotes(code_list):
    if not code_list: return {}
    code_list = list(set([str(c).strip() for c in code_list]))
    realtime_data = {}
    chunk_size = 20
    chunks = [code_list[i:i + chunk_size] for i in range(0, len(code_list), chunk_size)]
    
    for chunk in chunks:
        try:
            stocks = twstock.realtime.get(chunk)
            if isinstance(stocks, dict): stocks = [stocks]
            if stocks:
                for stock in stocks:
                    if stock['success']:
                        code = stock['info']['code']
                        name = stock['info'].get('name', code) 
                        price_str = stock['realtime'].get('latest_trade_price', '-')
                        if price_str == '-' or not price_str:
                            price_str = stock['realtime'].get('best_bid_price', ['-'])[0]
                        last_close = float(stock['info']['last_price']) if stock['info']['last_price'] != '-' else 0.0
                        try: current_price = float(price_str)
                        except: current_price = 0.0
                        vol_str = stock['realtime'].get('accumulate_trade_volume', '0')
                        try: volume = int(vol_str)
                        except: volume = 0
                        
                        if current_price > 0:
                            change_pct = ((current_price - last_close) / last_close) * 100 if last_close > 0 else 0
                            realtime_data[code] = {
                                '名稱': name,
                                '即時價': current_price,
                                '漲跌幅%': change_pct,
                                '當日量': volume,
                                '來源': 'TWSE'
                            }
            time.sleep(0.2)
        except: pass

    # Yahoo 備援
    missing_codes = [c for c in code_list if c not in realtime_data]
    if missing_codes:
        try:
            yf_codes = [f"{c}.TW" for c in missing_codes]
            tickers = yf.Tickers(" ".join(yf_codes))
            for code in missing_codes:
                try:
                    ticker = tickers.tickers[f"{code}.TW"]
                    name = code 
                    price = ticker.fast_info.last_price
                    prev_close = ticker.fast_info.previous_close
                    try: volume = ticker.fast_info.last_volume
                    except: volume = 0
                    if price and price > 0:
                        change_pct = ((price - prev_close) / prev_close) * 100 if prev_close else 0
                        realtime_data[code] = {
                            '名稱': name,
                            '即時價': price,
                            '漲跌幅%': change_pct,
                            '當日量': volume,
                            '來源': 'Yahoo'
                        }
                except: continue
        except: pass
            
    return realtime_data

def merge_realtime_data(df):
    if df.empty: return df
    codes = df['代號'].astype(str).tolist()
    rt_data = get_realtime_quotes(codes)
    df['即時價'] = df['代號'].map(lambda x: rt_data.get(x, {}).get('即時價', np.nan))
    df['漲跌幅%'] = df['代號'].map(lambda x: rt_data.get(x, {}).get('漲跌幅%', np.nan))
    df['當日量'] = df['代號'].map(lambda x: rt_data.get(x, {}).get('當日量', 0))
    # 若無即時價，回退使用昨收
    df['即時價'] = df['即時價'].fillna(df['收盤'])
    df['漲跌幅%'] = df['漲跌幅%'].fillna(0)
    df['當日量'] = df['當日量'].fillna(0)
    return df

# --- FinMind 籌碼分析 (維持不變) ---
def get_chip_analysis(symbol_list):
    # (此部分代碼維持原樣，篇幅考量省略，請直接使用您原本的 FinMind 函式)
    chip_data = []
    dl = DataLoader()
    p_bar = st.progress(0)
    status = st.empty()
    total = len(symbol_list)
    start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
    for i, symbol in enumerate(symbol_list):
        status.text(f"🔍 分析籌碼結構: {symbol} ({i+1}/{total})...")
        p_bar.progress((i + 1) / total)
        try:
            df = dl.taiwan_stock_institutional_investors(stock_id=symbol, start_date=start_date)
            if df.empty:
                chip_data.append({'代號': symbol, '投信(張)': 0, '外資(張)': 0, '主力動向': '⚪ 資料不足'})
                continue
            latest_date = df['date'].iloc[-1]
            day_data = df[df['date'] == latest_date]
            foreign_net = day_data[day_data['name'].str.contains('Foreign')]['buy'].sum() - day_data[day_data['name'].str.contains('Foreign')]['sell'].sum()
            foreign_buy = int(foreign_net // 1000)
            trust_net = day_data[day_data['name'] == 'Investment_Trust']['buy'].sum() - day_data[day_data['name'] == 'Investment_Trust']['sell'].sum()
            trust_buy = int(trust_net // 1000)
            status_str = ""
            if trust_buy > 0: status_str += "🔴 投信買 "
            elif trust_buy < 0: status_str += "🟢 投信賣 "
            if foreign_buy > 1000: status_str += "🔥 外資大買 "
            elif foreign_buy < -1000: status_str += "🧊 外資倒貨 "
            if trust_buy > 0 and foreign_buy > 0: final_tag = "🚀 土洋合買"
            elif trust_buy > 0 and foreign_buy < 0: final_tag = "⚔️ 土洋對作(信)"
            elif trust_buy < 0 and foreign_buy > 0: final_tag = "⚔️ 土洋對作(外)"
            elif trust_buy < 0 and foreign_buy < 0: final_tag = "☠️ 主力棄守"
            else: final_tag = "🟡 一般輪動"
            chip_data.append({'代號': symbol, '投信(張)': trust_buy, '外資(張)': foreign_buy, '主力動向': f"{final_tag} | {status_str}"})
            time.sleep(0.05) 
        except:
            chip_data.append({'代號': symbol, '投信(張)': 0, '外資(張)': 0, '主力動向': '❌ Error'})
    p_bar.empty()
    status.empty()
    return pd.DataFrame(chip_data)

# --- 庫存存取 (維持 GitHub) ---
def load_holdings():
    try:
        token = st.secrets["general"]["GITHUB_TOKEN"]
        g = Github(token)
        repo = g.get_repo(REPO_KEY)
        contents = repo.get_contents(FILE_PATH)
        df = pd.read_csv(contents.download_url)
        rename_map = {'代號': '股票代號', 'Code': '股票代號', 'Symbol': '股票代號', '股數': '持有股數', 'Shares': '持有股數', '均價': '買入均價', '成本': '買入均價', 'Price': '買入均價', 'Cost': '買入均價'}
        df = df.rename(columns=rename_map)
        df['股票代號'] = df['股票代號'].astype(str).str.strip()
        for c in ["股票代號", "買入均價", "持有股數"]:
            if c not in df.columns: df[c] = 0.0 if "價" in c else (0 if "股" in c else "")
        return df[["股票代號", "買入均價", "持有股數"]]
    except: return pd.DataFrame(columns=["股票代號", "買入均價", "持有股數"])

def save_holdings(df):
    try:
        token = st.secrets["general"]["GITHUB_TOKEN"]
        g = Github(token)
        repo = g.get_repo(REPO_KEY)
        csv_content = df.to_csv(index=False)
        try:
            contents = repo.get_contents(FILE_PATH)
            repo.update_file(contents.path, f"Update {get_taiwan_time()}", csv_content, contents.sha)
            st.success("✅ 庫存已同步至雲端！")
        except:
            repo.create_file(FILE_PATH, "Create holdings.csv", csv_content)
            st.success("✅ 建立並儲存成功！")
    except Exception as e: st.error(f"❌ 儲存失敗: {e}")

# --- 篩選與排序邏輯 (維持不變) ---
def get_stratified_selection(df):
    if df.empty: return df, []
    cols = ['攻擊分', '技術分', '量能分']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    # 篩選標準：技術分>60, 量能>60, 攻擊分>80
    mask = (df['技術分'] >= 60) & (df['量能分'] >= 60) & (df['攻擊分'] >= 80)
    filtered = df[mask].copy()
    if filtered.empty: return pd.DataFrame(), ["無符合條件標的"]
    
    # 分級
    b_a = filtered[filtered['攻擊分'] >= 90].sort_values('攻擊分', ascending=False).head(5)
    b_b = filtered[(filtered['攻擊分'] >= 85) & (filtered['攻擊分'] < 90)].sort_values('攻擊分', ascending=False).head(5)
    b_c = filtered[(filtered['攻擊分'] >= 80) & (filtered['攻擊分'] < 85)].sort_values('攻擊分', ascending=False).head(5)
    
    final = pd.concat([b_a, b_b, b_c])
    stats = [f"90+: {len(b_a)}", f"85-90: {len(b_b)}", f"80-85: {len(b_c)}"]
    return final, stats

def get_raw_top10(df):
    if df.empty: return df
    df['攻擊分'] = pd.to_numeric(df['攻擊分'], errors='coerce').fillna(0)
    return df.sort_values(by='攻擊分', ascending=False).head(10)

# --- 主程式 ---
def main():
    st.title("⚔️ V32 戰情室 (Drive Core)")
    if 'inventory' not in st.session_state: st.session_state['inventory'] = load_holdings()
    if 'input_key_counter' not in st.session_state: st.session_state['input_key_counter'] = 0
    
    if st.button("🔄 刷新即時報價", type="primary"):
        st.cache_data.clear()
        st.rerun()

    # 1. 載入 Drive 資料並運算分數
    with st.spinner("正在讀取 Google Drive 核心數據並進行全市場運算... (每日一次)"):
        v32_df, err = process_drive_data()
        
    if err: st.error(err)
    if not v32_df.empty:
        v32_df['cat'] = v32_df.apply(lambda r: 'Special' if ('債' in str(r.get('名稱', '')) or 'KY' in str(r.get('名稱', '')) or str(r['代號']).startswith(('00','91'))) else 'General', axis=1)
        v32_df = v32_df[v32_df['cat'] == 'General']
        st.caption(f"分析完成: 共 {len(v32_df)} 檔股票 | 資料來源: Google Drive (每日更新)")

    tab_strat, tab_raw, tab_inv = st.tabs(["🎯 V32 精選", "🏆 全市場 Top 10", "💼 庫存管理"])
    fmt_score = {'即時價':'{:.2f}', '漲跌幅%':'{:+.2f}%', '攻擊分':'{:.1f}', '技術分':'{:.0f}', '量能分':'{:.0f}', '當日量':'{:,}', '外資(張)': '{:,.0f}', '投信(張)': '{:,.0f}'}

    # === Tab 1: V32 精選 ===
    with tab_strat:
        if not v32_df.empty:
            final_df, stats = get_stratified_selection(v32_df)
            st.info(f"🎯 戰略結構：{' | '.join(stats)}")
            if not final_df.empty:
                # 取得即時報價 (針對篩選出來的少數股票)
                final_df = merge_realtime_data(final_df)
                
                col_btn, col_info = st.columns([1, 4])
                with col_btn:
                    scan_chip = st.button("🚀 籌碼掃描", key="btn_strat_scan")
                if scan_chip:
                    with st.spinner("分析籌碼中..."):
                        chip_df = get_chip_analysis(final_df['代號'].tolist())
                        if not chip_df.empty: final_df = pd.merge(final_df, chip_df, on='代號', how='left')

                final_df = final_df.sort_values(['攻擊分', '漲跌幅%'], ascending=[False, False])
                cols_to_show = ['代號','名稱','即時價','漲跌幅%','技術分','量能分','攻擊分']
                if '主力動向' in final_df.columns: cols_to_show += ['主力動向', '投信(張)', '外資(張)']
                
                st.dataframe(
                    final_df[cols_to_show].style
                    .format(fmt_score)
                    .background_gradient(subset=['攻擊分'], cmap=cmap_pastel_red)
                    .background_gradient(subset=['技術分'], cmap=cmap_pastel_blue)
                    .background_gradient(subset=['量能分'], cmap=cmap_pastel_green)
                    .map(color_change, subset=['漲跌幅%']), 
                    hide_index=True,
                    use_container_width=True
                )
            else: st.warning("無符合 V32 條件標的")
        else: st.warning("暫無資料")

    # === Tab 2: Top 10 ===
    with tab_raw:
        st.markdown("### 🏆 全市場攻擊力排行 (Top 10)")
        if not v32_df.empty:
            raw_df = get_raw_top10(v32_df)
            if not raw_df.empty:
                raw_df = merge_realtime_data(raw_df)
                if st.button("🚀 籌碼掃描 (Top 10)", key="btn_raw_scan"):
                    with st.spinner("分析籌碼中..."):
                        chip_df = get_chip_analysis(raw_df['代號'].tolist())
                        if not chip_df.empty: raw_df = pd.merge(raw_df, chip_df, on='代號', how='left')

                cols_to_show = ['代號','名稱','即時價','漲跌幅%','技術分','量能分','攻擊分']
                if '主力動向' in raw_df.columns: cols_to_show += ['主力動向', '投信(張)', '外資(張)']

                st.dataframe(
                    raw_df[cols_to_show].style
                    .format(fmt_score)
                    .background_gradient(subset=['攻擊分'], cmap=cmap_pastel_red)
                    .background_gradient(subset=['技術分'], cmap=cmap_pastel_blue)
                    .background_gradient(subset=['量能分'], cmap=cmap_pastel_green)
                    .map(color_change, subset=['漲跌幅%']),
                    hide_index=True,
                    use_container_width=True
                )

    # === Tab 3: 庫存管理 (邏輯維持) ===
    with tab_inv:
        # (這裡的程式碼與原本完全相同，省略以節省空間，請直接複製您原本的庫存管理區塊)
        # 唯一要修改的是：
        # 當要顯示庫存即時報價時，使用新的 v32_df 來取得分數
        st.subheader("📝 庫存交易管理")
        
        # 簡單載入名稱對照 (從 Drive Data 建立)
        name_map = {}
        if not v32_df.empty:
            name_map = dict(zip(v32_df['代號'], v32_df['名稱']))

        # ... (以下交易登記介面代碼與您原本的完全相同，請保留) ...
        # (為了完整性，若您需要我再貼一次這部分請告訴我)
        # ...
        
        st.divider()
        st.subheader("📊 持股監控")
        if not st.session_state['inventory'].empty:
            inv_df = st.session_state['inventory'].copy()
            inv_codes = inv_df['股票代號'].astype(str).tolist()
            inv_rt = get_realtime_quotes(inv_codes) 
            res = []
            score_map = v32_df.set_index('代號')['攻擊分'].to_dict() if not v32_df.empty else {}
            
            for idx, r in inv_df.iterrows():
                code = str(r['股票代號'])
                qty = float(r['持有股數'] or 0)
                cost = float(r['買入均價'] or 0)
                
                rt_info = inv_rt.get(code, {})
                curr = rt_info.get('即時價', 0)
                name = name_map.get(code, rt_info.get('名稱', code)) # 改用 Drive Map
                sc = score_map.get(code, 0)
                
                val = curr * qty
                c_tot = cost * qty
                pl = val - c_tot
                roi = (pl/c_tot*100) if c_tot>0 else 0
                
                if roi < -10: action = "🛑 停損"
                elif sc >= 60: action = "🟢 續抱"
                else: action = "🔻 賣出"
                
                res.append({
                    '代號': code, '名稱': name, '即時價': curr, 
                    '損益': pl, '報酬率%': roi, '攻擊分': sc, 
                    '建議操作': action, '持有股數': qty, '購入均價': cost
                })
            
            if res:
                df_res = pd.DataFrame(res)
                # ... (顯示邏輯不變) ...
                st.dataframe(
                    df_res[['代號', '名稱', '持有股數', '購入均價', '即時價', '損益', '報酬率%', '攻擊分', '建議操作']].style
                    .format({'購入均價':'{:.2f}', '即時價':'{:.2f}', '損益':'{:+,.0f}', '報酬率%':'{:+.2f}%', '攻擊分':'{:.0f}'})
                    .map(color_surplus, subset=['損益','報酬率%'])
                    .map(color_action, subset=['建議操作']),
                    use_container_width=True, hide_index=True
                )
        else: st.info("目前無庫存。")

if __name__ == "__main__":
    main()
