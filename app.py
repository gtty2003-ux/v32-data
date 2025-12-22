import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import pytz
import yfinance as yf
from github import Github 
import time
from FinMind.data import DataLoader
import twstock
import matplotlib.colors as mcolors # 新增：用於製作自定義淡色階

# --- 設定頁面資訊 ---
st.set_page_config(
    page_title="V32 戰情室 (Real-Time)",
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
REPO_KEY = "gtty2003-ux/v32-data"
FILE_PATH = "holdings.csv"

# --- 自定義淡色階 (Pastel Colormaps) ---
def make_pastel_cmap(hex_color):
    return mcolors.LinearSegmentedColormap.from_list("pastel_cmap", ["#ffffff", hex_color])

# 定義三種極淡色階 (最高分只會到這些粉嫩色，不會變深黑)
cmap_pastel_red   = make_pastel_cmap("#ef9a9a") # 粉紅 (Material Red 200)
cmap_pastel_blue  = make_pastel_cmap("#90caf9") # 粉藍 (Material Blue 200)
cmap_pastel_green = make_pastel_cmap("#a5d6a7") # 粉綠 (Material Green 200)

# --- 工具函數 ---
def get_taiwan_time():
    utc_now = datetime.utcnow()
    tw_time = utc_now.replace(tzinfo=pytz.utc).astimezone(pytz.timezone('Asia/Taipei'))
    return tw_time.strftime("%Y-%m-%d %H:%M:%S")

def color_surplus(val):
    if not isinstance(val, (int, float)): return ''
    if val > 0: return 'color: #d32f2f; font-weight: bold;' # 紅
    elif val < 0: return 'color: #388e3c; font-weight: bold;' # 綠
    return 'color: black'

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

# --- 核心防鎖機制 ---
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
                        price_str = stock['realtime'].get('latest_trade_price', '-')
                        if price_str == '-' or not price_str:
                            price_str = stock['realtime'].get('best_bid_price', ['-'])[0]
                        last_close = float(stock['info']['last_price']) if stock['info']['last_price'] != '-' else 0.0
                        try:
                            current_price = float(price_str)
                        except:
                            current_price = 0.0
                        
                        vol_str = stock['realtime'].get('accumulate_trade_volume', '0')
                        try: volume = int(vol_str)
                        except: volume = 0
                        
                        if current_price > 0:
                            change_pct = ((current_price - last_close) / last_close) * 100 if last_close > 0 else 0
                            realtime_data[code] = {
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
                    price = ticker.fast_info.last_price
                    prev_close = ticker.fast_info.previous_close
                    try: volume = ticker.fast_info.last_volume
                    except: volume = 0
                    if price and price > 0:
                        change_pct = ((price - prev_close) / prev_close) * 100 if prev_close else 0
                        realtime_data[code] = {
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

# --- V32 運算邏輯 ---
def calculate_indicators(hist):
    if len(hist) < 65: return 0, 0, 0, "0/5"
    close = hist['Close']
    vol = hist['Volume']
    high = hist['High']
    open_p = hist['Open']
    
    ma5_s = close.rolling(5).mean()
    ma20_s = close.rolling(20).mean()
    ma60_s = close.rolling(60).mean()
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi_s = 100 - (100 / (1 + rs))
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd_s = exp1 - exp2
    signal_s = macd_s.ewm(span=9, adjust=False).mean()
    vol_ma20_s = vol.rolling(20).mean()
    vol_ma5_s = vol.rolling(5).mean()
    high_20_s = high.rolling(20).max()
    raw_scores = [] 
    lookback_indices = range(-7, 0)
    for i in lookback_indices:
        c_now = close.iloc[i]
        ma5 = ma5_s.iloc[i]
        ma20 = ma20_s.iloc[i]
        ma20_prev = ma20_s.iloc[i-1] 
        ma60 = ma60_s.iloc[i]
        rsi_now = rsi_s.iloc[i]
        macd_now = macd_s.iloc[i]
        sig_now = signal_s.iloc[i]
        high_20_prev = high_20_s.iloc[i-1] 
        v_now = vol.iloc[i]
        v_prev = vol.iloc[i-1]
        v_ma20 = vol_ma20_s.iloc[i]
        v_ma5 = vol_ma5_s.iloc[i]
        o_now = open_p.iloc[i]
        t_score = 60
        if not np.isnan(ma20) and c_now > ma20: t_score += 5           
        if not np.isnan(ma20) and not np.isnan(ma20_prev) and ma20 > ma20_prev: t_score += 5       
        if not np.isnan(ma5) and not np.isnan(ma20) and not np.isnan(ma60):
            if ma5 > ma20 and ma20 > ma60: t_score += 10 
        if not np.isnan(rsi_now) and rsi_now > 50: t_score += 5           
        if not np.isnan(rsi_now) and rsi_now > 70: t_score += 5           
        if not np.isnan(macd_now) and not np.isnan(sig_now) and macd_now > sig_now: t_score += 5     
        if not np.isnan(high_20_prev) and c_now > high_20_prev: t_score += 10 
        v_score = 60
        if not np.isnan(v_ma20) and v_now > v_ma20: v_score += 10        
        if not np.isnan(v_ma5) and v_now > v_ma5: v_score += 10         
        is_red = c_now > o_now
        vol_increase = v_now > v_prev
        if is_red and vol_increase: v_score += 15 
        if not np.isnan(v_ma20) and v_now > v_ma20 * 1.5: v_score += 5 
        t_score = min(100, t_score)
        v_score = min(100, v_score)
        daily_total = (t_score * 0.7) + (v_score * 0.3)
        raw_scores.append(daily_total)
    raw_scores = [0 if np.isnan(x) else x for x in raw_scores]
    if len(raw_scores) < 2: return 0, 0, 0, "0/5"
    raw_today = raw_scores[-1]
    raw_yesterday = raw_scores[-2]
    attack_score = (raw_today * 0.7) + (raw_yesterday * 0.3)
    last_5_days = raw_scores[-5:]
    stability_count = sum(1 for s in last_5_days if s >= 70)
    stability_str = f"{stability_count}/5"
    return t_score, v_score, attack_score, stability_str

@st.cache_data(ttl=3600) 
def run_v32_engine(ticker_list):
    results = []
    p_bar = st.progress(0)
    status = st.empty()
    total = len(ticker_list)
    for i, row in enumerate(ticker_list):
        symbol = str(row['代號'])
        name = str(row.get('名稱', ''))
        status.text(f"建立 V32 戰略地圖: {symbol} {name} ({i+1}/{total})...")
        p_bar.progress((i + 1) / total)
        try:
            stock = yf.Ticker(f"{symbol}.TW")
            hist = stock.history(period="6mo")
            if len(hist) < 65: continue 
            t_s, v_s, atk_s, stab = calculate_indicators(hist)
            results.append({
                '代號': symbol, '名稱': name,
                '收盤': hist['Close'].iloc[-1],
                '技術分': t_s, '量能分': v_s, '攻擊分': atk_s, '穩定度': stab   
            })
        except: continue
    p_bar.empty()
    status.empty()
    return pd.DataFrame(results)

def load_and_process_data():
    url = f"https://raw.githubusercontent.com/{REPO_KEY}/main/v32_recommend.csv"
    try:
        df = pd.read_csv(url)
        code_col = next((c for c in ['代碼', '代號', 'Code', 'Symbol'] if c in df.columns), None)
        if code_col:
            df[code_col] = df[code_col].astype(str).str.strip()
            df = df.rename(columns={code_col: '代號'})
            processed = run_v32_engine(df[['代號', '名稱']].to_dict('records'))
            return processed, None
    except Exception as e:
        return pd.DataFrame(), str(e)

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

# --- 篩選與排序邏輯 ---
def get_stratified_selection(df):
    if df.empty: return df, []
    cols = ['攻擊分', '技術分', '量能分']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    mask = (df['技術分'] >= 80) & (df['量能分'] >= 60) & (df['攻擊分'] >= 86) & (df['攻擊分'] <= 92)
    filtered = df[mask].copy()
    if filtered.empty: return pd.DataFrame(), ["無符合條件標的"]
    b_a = filtered[(filtered['攻擊分'] >= 90) & (filtered['攻擊分'] <= 92)].sort_values('攻擊分', ascending=False).head(5)
    b_b = filtered[(filtered['攻擊分'] >= 88) & (filtered['攻擊分'] < 90)].sort_values('攻擊分', ascending=False).head(5)
    b_c = filtered[(filtered['攻擊分'] >= 86) & (filtered['攻擊分'] < 88)].sort_values('攻擊分', ascending=False).head(5)
    final = pd.concat([b_a, b_b, b_c])
    stats = [f"90-92: {len(b_a)}", f"88-90: {len(b_b)}", f"86-88: {len(b_c)}"]
    return final, stats

def get_raw_top10(df):
    if df.empty: return df
    df['攻擊分'] = pd.to_numeric(df['攻擊分'], errors='coerce').fillna(0)
    return df.sort_values(by='攻擊分', ascending=False).head(10)

# --- 主程式 ---
def main():
    st.title("⚔️ V32 戰情室 (Real-Time Mode)")
    if 'inventory' not in st.session_state: st.session_state['inventory'] = load_holdings()
    if '股票代號' not in st.session_state['inventory'].columns: st.session_state['inventory'] = load_holdings()
    if 'input_key_counter' not in st.session_state: st.session_state['input_key_counter'] = 0
    if st.button("🔄 刷新即時報價", type="primary"):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"最後更新: {get_taiwan_time()} | V32(昨收) + 即時報價(盤中) | 雙重報價來源: TWSE + Yahoo | 60秒保護啟用")
    v32_df, err = load_and_process_data()
    if err: st.error(err)
    if not v32_df.empty:
        v32_df['cat'] = v32_df.apply(lambda r: 'Special' if ('債' in str(r.get('名稱')) or 'KY' in str(r.get('名稱')) or str(r['代號']).startswith(('00','91')) or str(r['代號'])[-1].isalpha() or (len(str(r['代號']))>4 and str(r['代號']).isdigit())) else 'General', axis=1)
        v32_df = v32_df[v32_df['cat'] == 'General']

    tab_strat, tab_raw, tab_inv = st.tabs(["🎯 今日攻擊力 Top 15", "🏆 原始攻擊分 Top 10", "💼 庫存管理"])
    fmt_score = {'即時價':'{:.2f}', '漲跌幅%':'{:+.2f}%', '攻擊分':'{:.1f}', '技術分':'{:.0f}', '量能分':'{:.0f}', '當日量':'{:,}', '外資(張)': '{:,.0f}', '投信(張)': '{:,.0f}'}

    # === Tab 1: 分層精選 (已同步調整) ===
    with tab_strat:
        if not v32_df.empty:
            final_df, stats = get_stratified_selection(v32_df)
            st.info(f"🎯 戰略結構：{' | '.join(stats)}")
            if not final_df.empty:
                final_df = merge_realtime_data(final_df)
                col_btn, col_info = st.columns([1, 4])
                with col_btn:
                    scan_chip = st.button("🚀 籌碼掃描", key="btn_strat_scan")
                if scan_chip:
                    with st.spinner("分析籌碼中..."):
                        chip_df = get_chip_analysis(final_df['代號'].tolist())
                        if not chip_df.empty: final_df = pd.merge(final_df, chip_df, on='代號', how='left')

                final_df = final_df.sort_values(['攻擊分', '漲跌幅%'], ascending=[False, False])
                # 欄位同步：移除漲跌幅、當日量；新增技術分、量能分
                cols_to_show = ['代號','名稱','即時價','技術分','量能分','攻擊分','穩定度']
                if '主力動向' in final_df.columns: cols_to_show += ['主力動向', '投信(張)', '外資(張)']
                
                st.dataframe(
                    final_df[cols_to_show].style
                    .format(fmt_score)
                    # 套用極淡色階 (Pastel Colors)
                    .background_gradient(subset=['攻擊分'], cmap=cmap_pastel_red)
                    .background_gradient(subset=['技術分'], cmap=cmap_pastel_blue)
                    .background_gradient(subset=['量能分'], cmap=cmap_pastel_green)
                    .map(color_action, subset=['穩定度']), 
                    hide_index=True,
                    use_container_width=True
                )
            else: st.warning("無符合條件標的")
        else: st.warning("暫無資料")

    # === Tab 2: Top 10 (已同步調整) ===
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

                # 欄位同步
                cols_to_show = ['代號','名稱','即時價','技術分','量能分','攻擊分','穩定度']
                if '主力動向' in raw_df.columns: cols_to_show += ['主力動向', '投信(張)', '外資(張)']

                st.dataframe(
                    raw_df[cols_to_show].style
                    .format(fmt_score)
                    # 套用極淡色階
                    .background_gradient(subset=['攻擊分'], cmap=cmap_pastel_red)
                    .background_gradient(subset=['技術分'], cmap=cmap_pastel_blue)
                    .background_gradient(subset=['量能分'], cmap=cmap_pastel_green),
                    hide_index=True,
                    use_container_width=True
                )

    # === Tab 3: 庫存管理 ===
    with tab_inv:
        st.subheader("📝 庫存交易管理")
        input_key = st.session_state['input_key_counter']
        st.markdown("##### 📥 **買入登記 (Buy)** - 自動計算加權平均成本")
        df_buy_in = pd.DataFrame([{"股票代號": "", "持有股數": 1000, "買入均價": 0.0}])
        edited_buy = st.data_editor(df_buy_in, num_rows="dynamic", key=f"buy_{input_key}", use_container_width=True, hide_index=True)
        st.markdown("##### 📤 **賣出登記 (Sell)** - 扣除股數")
        df_sell_in = pd.DataFrame([{"股票代號": "", "持有股數": 1000}])
        edited_sell = st.data_editor(df_sell_in, num_rows="dynamic", key=f"sell_{input_key}", use_container_width=True, hide_index=True)
        
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
                if not code: continue
                qty = float(r['持有股數'] or 0)
                cost = float(r['買入均價'] or 0)
                curr = inv_rt.get(code, {}).get('即時價', 0)
                sc = score_map.get(code, 0)
                val = curr * qty
                c_tot = cost * qty
                pl = val - c_tot
                roi = (pl/c_tot*100) if c_tot>0 else 0
                if roi < -10: action = "🛑 停損 (虧損擴大)"
                elif sc >= 60: action = "🟢 續抱 (動能仍存)"
                else: action = "🔻 賣出 (動能熄火)"
                res.append({'代號': code, '即時價': curr, '損益': pl, '報酬率%': roi, '攻擊分': sc, '建議操作': action, '持有股數': qty, '購入均價': cost})
            if res:
                df_res = pd.DataFrame(res)
                c1, c2, c3 = st.columns(3)
                c1.metric("總成本", f"${(df_res['購入均價']*df_res['持有股數']).sum():,.0f}")
                total_pl = df_res['損益'].sum()
                c2.metric("總損益", f"${total_pl:,.0f}", delta=f"{total_pl:,.0f}")
                c3.metric("總市值", f"${(df_res['即時價']*df_res['持有股數']).sum():,.0f}")
                st.dataframe(
                    df_res[['代號', '持有股數', '購入均價', '即時價', '損益', '報酬率%', '攻擊分', '建議操作']].style
                    .format({'購入均價':'{:.2f}', '即時價':'{:.2f}', '損益':'{:+,.0f}', '報酬率%':'{:+.2f}%', '攻擊分':'{:.0f}'})
                    .map(color_surplus, subset=['損益','報酬率%'])
                    .map(color_action, subset=['建議操作']),
                    use_container_width=True, hide_index=True,
                    column_config={"購入均價": st.column_config.NumberColumn("購入均價", format="$%.2f"), "持有股數": st.column_config.NumberColumn("股數", format="%d")}
                )
        else: st.info("目前無庫存，請在上方新增交易。")

if __name__ == "__main__":
    main()
