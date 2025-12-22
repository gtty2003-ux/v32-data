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

# --- 全域變數 (雙倉庫設定) ---
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
    </style>
    """, unsafe_allow_html=True)

# --- 工具函數 ---
def get_taiwan_time():
    utc_now = datetime.utcnow()
    return utc_now.replace(tzinfo=pytz.utc).astimezone(pytz.timezone('Asia/Taipei')).strftime("%Y-%m-%d %H:%M:%S")

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
    if "賣出" in val_str or "停損" in val_str:
        return 'color: #ffffff; background-color: #d32f2f; font-weight: bold; padding: 5px; border-radius: 5px;'
    elif "續抱" in val_str:
        return 'color: #1b5e20; font-weight: bold;'
    return ''

def color_change(val):
    if not isinstance(val, (int, float)): return ''
    if val > 0: return 'color: #d32f2f; background-color: rgba(255,0,0,0.1); font-weight: bold;'
    elif val < 0: return 'color: #388e3c; background-color: rgba(0,255,0,0.1); font-weight: bold;'
    return 'color: gray'

# --- 核心 1：從 Auto-Updater 讀取股價資料 ---
@st.cache_data(ttl=1800)
def load_data_from_github():
    try:
        token = st.secrets["general"]["GITHUB_TOKEN"]
        url = f"https://api.github.com/repos/{DATA_REPO}/contents/{DATA_FILE}"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3.raw"
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            df['Code'] = df['Code'].astype(str).str.strip()
            df['Date'] = pd.to_datetime(df['Date'])
            
            numeric_cols = ['ClosingPrice', 'OpeningPrice', 'HighestPrice', 'LowestPrice', 'TradeVolume']
            for c in numeric_cols:
                if c in df.columns: 
                    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            return df
        else:
            if response.status_code == 404:
                return pd.DataFrame()
            st.error(f"GitHub (Data) 連線失敗: {response.status_code}")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"讀取資料錯誤: {e}")
        return pd.DataFrame()

# --- V32 運算邏輯 ---
def calculate_v32_score(df_group):
    if len(df_group) < 60: return None 
    
    df = df_group.sort_values('Date').reset_index(drop=True)
    close = df['ClosingPrice']
    vol = df['TradeVolume']
    high = df['HighestPrice']
    open_p = df['OpeningPrice']
    
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
    
    i = -1 
    c_now = close.iloc[i]
    if pd.isna(c_now) or c_now == 0: return None
    
    m5, m20, m60 = ma5.iloc[i], ma20.iloc[i], ma60.iloc[i]
    m20_prev = ma20.iloc[i-1]
    r_now = rsi.iloc[i]
    macd_now, sig_now = macd.iloc[i], signal.iloc[i]
    h20_prev = high_20.iloc[i-1]
    v_now, v_prev = vol.iloc[i], vol.iloc[i-1]
    v_m5, v_m20 = vol_ma5.iloc[i], vol_ma20.iloc[i]
    o_now = open_p.iloc[i]
    
    t_score = 60
    if c_now > m20: t_score += 5
    if m20 > m20_prev: t_score += 5
    if m5 > m20 and m20 > m60: t_score += 10
    if r_now > 50: t_score += 5
    if r_now > 70: t_score += 5
    if macd_now > sig_now: t_score += 5
    if c_now > h20_prev: t_score += 10
    t_score = min(100, t_score)
    
    v_score = 60
    if v_now > v_m20: v_score += 10
    if v_now > v_m5: v_score += 10
    if c_now > o_now and v_now > v_prev: v_score += 15
    if v_now > v_m20 * 1.5: v_score += 5
    v_score = min(100, v_score)
    
    raw_today = (t_score * 0.7) + (v_score * 0.3)
    
    return {'技術分': t_score, '量能分': v_score, '攻擊分': raw_today, '收盤': c_now}

@st.cache_data(ttl=1800)
def process_data():
    raw_df = load_data_from_github()
    if raw_df.empty: return pd.DataFrame(), "無法讀取數據 (v32-auto-updater)，請確認 CSV 是否存在。"
    
    results = []
    grouped = raw_df.groupby('Code')
    
    for code, group in grouped:
        name = group['Name'].iloc[-1]
        score_data = calculate_v32_score(group)
        if score_data:
            score_data['代號'] = code
            score_data['名稱'] = name
            results.append(score_data)
            
    return pd.DataFrame(results), None

# --- 即時報價 ---
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
    df['即時價'] = df['即時價'].fillna(df['收盤'])
    df['漲跌幅%'] = df['漲跌幅%'].fillna(0)
    df['當日量'] = df['當日量'].fillna(0)
    return df

# --- FinMind 籌碼分析 ---
def get_chip_analysis(symbol_list):
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

# --- 核心 2：庫存存取 (v32-data 自己) ---
def load_holdings():
    try:
        token = st.secrets["general"]["GITHUB_TOKEN"]
        g = Github(token)
        repo = g.get_repo(HOLDING_REPO)
        contents = repo.get_contents(HOLDINGS_FILE)
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
        repo = g.get_repo(HOLDING_REPO)
        csv_content = df.to_csv(index=False)
        try:
            contents = repo.get_contents(HOLDINGS_FILE)
            repo.update_file(contents.path, f"Update {get_taiwan_time()}", csv_content, contents.sha)
            st.success("✅ 庫存已同步至雲端！")
        except:
            repo.create_file(HOLDINGS_FILE, "Create holdings.csv", csv_content)
            st.success("✅ 建立並儲存成功！")
    except Exception as e: st.error(f"❌ 儲存失敗: {e}")

# --- 篩選邏輯 (86-92分 + 股價<80) ---
def get_stratified_selection(df):
    if df.empty: return df
    
    # 1. 確保數值型態正確
    cols = ['攻擊分', '技術分', '量能分', '收盤']
    for c in cols: 
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    # 2. 嚴格篩選條件：
    #    (1) 技術 >= 60, 量能 >= 60
    #    (2) 股價 <= 80 (剔除高價股)
    #    (3) 攻擊分鎖定 86 ~ 92 (剔除過熱股)
    mask = (df['技術分'] >= 60) & \
           (df['量能分'] >= 60) & \
           (df['收盤'] <= 80) & \
           (df['攻擊分'] >= 86) & \
           (df['攻擊分'] <= 92)
           
    filtered = df[mask].copy()
    
    # 依照攻擊分排序
    return filtered.sort_values('攻擊分', ascending=False)

def get_raw_top10(df):
    if df.empty: return df
    df['攻擊分'] = pd.to_numeric(df['攻擊分'], errors='coerce').fillna(0)
    return df.sort_values(by='攻擊分', ascending=False).head(10)

# --- 主程式 ---
def main():
    st.title("⚔️ V32 戰情室 (Dual Core)")
    
    if 'inventory' not in st.session_state: st.session_state['inventory'] = load_holdings()
    if 'input_key_counter' not in st.session_state: st.session_state['input_key_counter'] = 0
    
    if st.button("🔄 刷新即時報價", type="primary"):
        st.cache_data.clear()
        st.rerun()

    # 1. 載入資料
    with st.spinner("正在讀取核心數據 (v32-auto-updater)..."):
        v32_df, err = process_data()
        
    if err: st.error(err)
    if not v32_df.empty:
        # 過濾非普通股 (ETF, KY, 債券等)
        v32_df['cat'] = v32_df.apply(lambda r: 'Special' if ('債' in str(r.get('名稱', '')) or 'KY' in str(r.get('名稱', '')) or str(r['代號']).startswith(('00','91'))) else 'General', axis=1)
        v32_df = v32_df[v32_df['cat'] == 'General']
        st.caption(f"分析完成: 共 {len(v32_df)} 檔股票 | 資料來源: v32-auto-updater")

    tab_strat, tab_raw, tab_inv = st.tabs(["🎯 V32 精選", "🏆 全市場 Top 10", "💼 庫存管理"])
    fmt_score = {'即時價':'{:.2f}', '漲跌幅%':'{:+.2f}%', '攻擊分':'{:.1f}', '技術分':'{:.0f}', '量能分':'{:.0f}', '當日量':'{:,}', '外資(張)': '{:,.0f}', '投信(張)': '{:,.0f}'}

    # === Tab 1: V32 精選 (三階分表 + 無漲跌幅) ===
    with tab_strat:
        if not v32_df.empty:
            # 取得所有符合 86-92分 & 股價<80 的股票
            final_df = get_stratified_selection(v32_df)
            
            if not final_df.empty:
                # 補上即時報價
                final_df = merge_realtime_data(final_df)
                
                col_btn, col_info = st.columns([1, 4])
                with col_btn:
                    # 一次掃描所有區段的籌碼
                    scan_chip = st.button("🚀 籌碼掃描", key="btn_strat_scan")
                
                if scan_chip:
                    with st.spinner("正在掃描全區段籌碼..."):
                        chip_df = get_chip_analysis(final_df['代號'].tolist())
                        if not chip_df.empty: 
                            final_df = pd.merge(final_df, chip_df, on='代號', how='left')

                # 定義顯示欄位 (已移除漲跌幅)
                base_cols = ['代號','名稱','即時價','技術分','量能分','攻擊分']
                if '主力動向' in final_df.columns: 
                    base_cols += ['主力動向', '投信(張)', '外資(張)']

                # 拆解成三個等級
                # S級: 90 <= 分數 <= 92
                df_s = final_df[(final_df['攻擊分'] >= 90) & (final_df['攻擊分'] <= 92)]
                
                # A級: 88 <= 分數 < 90
                df_a = final_df[(final_df['攻擊分'] >= 88) & (final_df['攻擊分'] < 90)]
                
                # B級: 86 <= 分數 < 88
                df_b = final_df[(final_df['攻擊分'] >= 86) & (final_df['攻擊分'] < 88)]

                # --- S 級表格 ---
                st.subheader(f"👑 S 級主力區 (90-92分) - 共 {len(df_s)} 檔")
                if not df_s.empty:
                    st.dataframe(
                        df_s[base_cols].style
                        .format(fmt_score)
                        .background_gradient(subset=['攻擊分'], cmap=cmap_pastel_red)
                        .background_gradient(subset=['技術分'], cmap=cmap_pastel_blue)
                        .background_gradient(subset=['量能分'], cmap=cmap_pastel_green),
                        hide_index=True, use_container_width=True
                    )
                else:
                    st.caption("此區段暫無標的")

                st.divider()

                # --- A 級表格 ---
                st.subheader(f"🚀 A 級蓄勢區 (88-90分) - 共 {len(df_a)} 檔")
                if not df_a.empty:
                    st.dataframe(
                        df_a[base_cols].style
                        .format(fmt_score)
                        .background_gradient(subset=['攻擊分'], cmap=cmap_pastel_red)
                        .background_gradient(subset=['技術分'], cmap=cmap_pastel_blue)
                        .background_gradient(subset=['量能分'], cmap=cmap_pastel_green),
                        hide_index=True, use_container_width=True
                    )
                else:
                    st.caption("此區段暫無標的")

                st.divider()

                # --- B 級表格 ---
                st.subheader(f"👀 B 級觀察區 (86-88分) - 共 {len(df_b)} 檔")
                if not df_b.empty:
                    st.dataframe(
                        df_b[base_cols].style
                        .format(fmt_score)
                        .background_gradient(subset=['攻擊分'], cmap=cmap_pastel_red)
                        .background_gradient(subset=['技術分'], cmap=cmap_pastel_blue)
                        .background_gradient(subset=['量能分'], cmap=cmap_pastel_green),
                        hide_index=True, use_container_width=True
                    )
                else:
                    st.caption("此區段暫無標的")

            else: 
                st.warning("無符合條件標的 (區間 86~92 分, 股價<=80)")
        else: 
            st.warning("暫無資料 (請確認 v32-auto-updater 是否已執行 Action)")

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

    # === Tab 3: 庫存管理 ===
    with tab_inv:
        st.subheader("📝 庫存交易管理")
        
        name_map = {}
        if not v32_df.empty:
            name_map = dict(zip(v32_df['代號'], v32_df['名稱']))

        input_key = st.session_state['input_key_counter']
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 📥 **買入**")
            df_buy_in = pd.DataFrame([{"股票代號": "", "持有股數": 1000, "買入均價": 0.0}])
            edited_buy = st.data_editor(df_buy_in, num_rows="dynamic", key=f"buy_{input_key}", hide_index=True)
        with c2:
            st.markdown("##### 📤 **賣出**")
            df_sell_in = pd.DataFrame([{"股票代號": "", "持有股數": 1000}])
            edited_sell = st.data_editor(df_sell_in, num_rows="dynamic", key=f"sell_{input_key}", hide_index=True)
        
        st.write("")
        if st.button("💾 執行交易並儲存", type="primary"):
            current_inv = st.session_state['inventory'].copy()
            has_update = False
            for _, row in edited_buy.iterrows():
                code = str(row['股票代號']).strip()
                shares = int(row['持有股數']) if row['持有股數'] else 0
                price = float(row['買入均價']) if row['買入均價'] else 0.0
                if code and shares > 0 and price > 0:
                    has_update = True
                    match = current_inv[current_inv['股票代號'] == code]
                    if not match.empty:
                        idx = match.index[0]
                        old_shares = float(current_inv.at[idx, '持有股數'])
                        old_cost = float(current_inv.at[idx, '買入均價'])
                        total_shares = old_shares + shares
                        new_avg = ((old_shares * old_cost) + (shares * price)) / total_shares
                        current_inv.at[idx, '持有股數'] = total_shares
                        current_inv.at[idx, '買入均價'] = round(new_avg, 2)
                    else:
                        new_row = pd.DataFrame([{'股票代號': code, '持有股數': shares, '買入均價': price}])
                        current_inv = pd.concat([current_inv, new_row], ignore_index=True)
            for _, row in edited_sell.iterrows():
                code = str(row['股票代號']).strip()
                shares = int(row['持有股數']) if row['持有股數'] else 0
                if code and shares > 0:
                    match = current_inv[current_inv['股票代號'] == code]
                    if not match.empty:
                        has_update = True
                        idx = match.index[0]
                        cur_shares = float(current_inv.at[idx, '持有股數'])
                        if cur_shares > shares: current_inv.at[idx, '持有股數'] = cur_shares - shares
                        else: current_inv = current_inv.drop(idx)
            
            if has_update:
                st.session_state['inventory'] = current_inv
                save_holdings(current_inv)
                st.session_state['input_key_counter'] += 1 
                st.rerun()
            else: st.warning("未偵測到有效交易資料")

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
                name = name_map.get(code, rt_info.get('名稱', code)) 
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
                c1, c2, c3 = st.columns(3)
                c1.metric("總成本", f"${(df_res['購入均價']*df_res['持有股數']).sum():,.0f}")
                total_pl = df_res['損益'].sum()
                c2.metric("總損益", f"${total_pl:,.0f}", delta=f"{total_pl:,.0f}")
                c3.metric("總市值", f"${(df_res['即時價']*df_res['持有股數']).sum():,.0f}")
                
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
