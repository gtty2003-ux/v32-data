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
        else:
            if response.status_code == 404: return pd.DataFrame()
            st.error(f"GitHub 連線失敗: {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"讀取資料錯誤: {e}")
        return pd.DataFrame()

# --- V32 運算邏輯 ---
def calculate_v32_score(df_group):
    if len(df_group) < 60: return None 
    df = df_group.sort_values('Date').reset_index(drop=True)
    close, vol = df['ClosingPrice'], df['TradeVolume']
    high, open_p = df['HighestPrice'], df['OpeningPrice']
    
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
    
    vol_ma5, vol_ma20 = vol.rolling(5).mean(), vol.rolling(20).mean()
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
    if raw_df.empty: return pd.DataFrame(), "無法讀取數據 (v32-auto-updater)"
    results = []
    for code, group in raw_df.groupby('Code'):
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
                        if not price_str or price_str == '-': price_str = stock['realtime'].get('best_bid_price', ['-'])[0]
                        try: current_price = float(price_str)
                        except: current_price = 0.0
                        realtime_data[code] = {'名稱': name, '即時價': current_price, '來源': 'TWSE'}
            time.sleep(0.2)
        except: pass

    # Yahoo 備援
    missing = [c for c in code_list if c not in realtime_data]
    if missing:
        try:
            yf_codes = [f"{c}.TW" for c in missing]
            tickers = yf.Tickers(" ".join(yf_codes))
            for c in missing:
                try:
                    t = tickers.tickers[f"{c}.TW"]
                    p = t.fast_info.last_price
                    if p and p > 0: realtime_data[c] = {'名稱': c, '即時價': p, '來源': 'Yahoo'}
                except: continue
        except: pass
    return realtime_data

def merge_realtime_data(df):
    if df.empty: return df
    rt = get_realtime_quotes(df['代號'].astype(str).tolist())
    df['即時價'] = df['代號'].map(lambda x: rt.get(x, {}).get('即時價', np.nan))
    df['即時價'] = df['即時價'].fillna(df['收盤'])
    return df

# --- 籌碼分析 ---
def get_chip_analysis(symbol_list):
    chip_data = []
    dl = DataLoader()
    p_bar = st.progress(0)
    status = st.empty()
    total = len(symbol_list)
    start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
    for i, symbol in enumerate(symbol_list):
        status.text(f"🔍 分析籌碼: {symbol} ({i+1}/{total})")
        p_bar.progress((i + 1) / total)
        try:
            df = dl.taiwan_stock_institutional_investors(stock_id=symbol, start_date=start_date)
            if df.empty:
                chip_data.append({'代號': symbol, '投信(張)': 0, '外資(張)': 0, '主力動向': '⚪ 資料不足'})
            else:
                latest = df[df['date'] == df['date'].iloc[-1]]
                f_net = latest[latest['name'].str.contains('Foreign')]['buy'].sum() - latest[latest['name'].str.contains('Foreign')]['sell'].sum()
                t_net = latest[latest['name'] == 'Investment_Trust']['buy'].sum() - latest[latest['name'] == 'Investment_Trust']['sell'].sum()
                f_buy, t_buy = int(f_net // 1000), int(t_net // 1000)
                
                status_str = "🔴 投信買 " if t_buy > 0 else ("🟢 投信賣 " if t_buy < 0 else "")
                if f_buy > 1000: status_str += "🔥 外資大買 "
                elif f_buy < -1000: status_str += "🧊 外資倒貨 "
                
                if t_buy > 0 and f_buy > 0: tag = "🚀 土洋合買"
                elif t_buy > 0 and f_buy < 0: tag = "⚔️ 土洋對作(信)"
                elif t_buy < 0 and f_buy > 0: tag = "⚔️ 土洋對作(外)"
                elif t_buy < 0 and f_buy < 0: tag = "☠️ 主力棄守"
                else: tag = "🟡 一般輪動"
                chip_data.append({'代號': symbol, '投信(張)': t_buy, '外資(張)': f_buy, '主力動向': f"{tag} | {status_str}"})
            time.sleep(0.05)
        except: chip_data.append({'代號': symbol, '投信(張)': 0, '外資(張)': 0, '主力動向': '❌ Error'})
    p_bar.empty()
    status.empty()
    return pd.DataFrame(chip_data)

# --- 庫存存取 ---
def load_holdings():
    try:
        g = Github(st.secrets["general"]["GITHUB_TOKEN"])
        df = pd.read_csv(g.get_repo(HOLDING_REPO).get_contents(HOLDINGS_FILE).download_url)
        rename = {'Code': '股票代號', '股數': '持有股數', '均價': '買入均價'}
        df = df.rename(columns=rename)
        return df
    except: return pd.DataFrame(columns=["股票代號", "買入均價", "持有股數"])

def save_holdings(df):
    try:
        g = Github(st.secrets["general"]["GITHUB_TOKEN"])
        repo = g.get_repo(HOLDING_REPO)
        try: repo.update_file(HOLDINGS_FILE, f"Update {get_taiwan_time()}", df.to_csv(index=False), repo.get_contents(HOLDINGS_FILE).sha)
        except: repo.create_file(HOLDINGS_FILE, "Create", df.to_csv(index=False))
        st.success("✅ 儲存成功")
    except Exception as e: st.error(f"❌ 儲存失敗: {e}")

# --- 核心篩選函式 ---
def get_stratified_selection(df, price_limit):
    if df.empty: return df
    cols = ['攻擊分', '技術分', '量能分', '收盤']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    mask = (df['技術分'] >= 60) & (df['量能分'] >= 60) & \
           (df['收盤'] <= price_limit) & \
           (df['攻擊分'] >= 86) & (df['攻擊分'] <= 92)
           
    return df[mask].sort_values('攻擊分', ascending=False)

def display_graded_tables(filtered_df, key_suffix):
    """顯示 S/A/B 三級表格的共用函式"""
    if filtered_df.empty:
        st.warning("此條件下無符合標的")
        return

    # 1. 先抓出預計會顯示的前 30 檔 (S/A/B 各 Top 10)
    df_s_pre = filtered_df[(filtered_df['攻擊分'] >= 90) & (filtered_df['攻擊分'] <= 92)].head(10)
    df_a_pre = filtered_df[(filtered_df['攻擊分'] >= 88) & (filtered_df['攻擊分'] < 90)].head(10)
    df_b_pre = filtered_df[(filtered_df['攻擊分'] >= 86) & (filtered_df['攻擊分'] < 88)].head(10)
    
    # 合併這些會上榜的代號
    target_codes = pd.concat([df_s_pre, df_a_pre, df_b_pre])['代號'].tolist()

    # 2. 籌碼掃描 (只針對這 30 檔)
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        if st.button(f"🚀 籌碼掃描 (針對上榜 {len(target_codes)} 檔)", key=f"scan_{key_suffix}"):
            if target_codes:
                with st.spinner(f"正在分析最精華的 {len(target_codes)} 檔籌碼..."):
                    chip_df = get_chip_analysis(target_codes)
                    if not chip_df.empty: 
                        # 將籌碼資料合併回原始 filtered_df
                        filtered_df = pd.merge(filtered_df, chip_df, on='代號', how='left')

    # 3. 補上即時報價 (針對所有篩選結果)
    filtered_df = merge_realtime_data(filtered_df)

    # 定義顯示欄位
    base_cols = ['代號','名稱','即時價','技術分','量能分','攻擊分']
    if '主力動向' in filtered_df.columns: base_cols += ['主力動向', '投信(張)', '外資(張)']
    fmt_score = {'即時價':'{:.2f}', '攻擊分':'{:.1f}', '技術分':'{:.0f}', '量能分':'{:.0f}', '外資(張)': '{:,.0f}', '投信(張)': '{:,.0f}'}

    # 4. 再次切分資料 (這次可能包含籌碼資料)
    df_s = filtered_df[(filtered_df['攻擊分'] >= 90) & (filtered_df['攻擊分'] <= 92)].head(10)
    df_a = filtered_df[(filtered_df['攻擊分'] >= 88) & (filtered_df['攻擊分'] < 90)].head(10)
    df_b = filtered_df[(filtered_df['攻擊分'] >= 86) & (filtered_df['攻擊分'] < 88)].head(10)

    # 5. 渲染表格
    for title, df_sub, color in [
        (f"👑 S 級主力區 (90-92分) - Top {len(df_s)}", df_s, None),
        (f"🚀 A 級蓄勢區 (88-90分) - Top {len(df_a)}", df_a, None),
        (f"👀 B 級觀察區 (86-88分) - Top {len(df_b)}", df_b, None)
    ]:
        st.subheader(title)
        if not df_sub.empty:
            st.dataframe(
                df_sub[base_cols].style.format(fmt_score)
                .background_gradient(subset=['攻擊分'], cmap=cmap_pastel_red)
                .background_gradient(subset=['技術分'], cmap=cmap_pastel_blue)
                .background_gradient(subset=['量能分'], cmap=cmap_pastel_green),
                hide_index=True, use_container_width=True
            )
        else: st.caption("暫無標的")
        st.divider()

# --- 主程式 ---
def main():
    st.title("⚔️ V32 戰情室 (Dual Core)")
    if 'inventory' not in st.session_state: st.session_state['inventory'] = load_holdings()
    if 'input_key_counter' not in st.session_state: st.session_state['input_key_counter'] = 0
    
    if st.button("🔄 刷新即時報價", type="primary"): st.cache_data.clear(); st.rerun()

    # 載入資料
    with st.spinner("讀取核心數據..."):
        v32_df, err = process_data()
    if err: st.error(err)
    
    if not v32_df.empty:
        v32_df['cat'] = v32_df.apply(lambda r: 'Special' if ('債' in str(r.get('名稱', '')) or 'KY' in str(r.get('名稱', '')) or str(r['代號']).startswith(('00','91'))) else 'General', axis=1)
        v32_df = v32_df[v32_df['cat'] == 'General']
        st.caption(f"資料來源: v32-auto-updater | 總檔數: {len(v32_df)}")

    # 建立分頁
    tab_80, tab_50, tab_inv = st.tabs(["💰 80元以下推薦", "🪙 50元以下推薦", "💼 庫存管理"])

    # === Tab 1: 80元以下 ===
    with tab_80:
        if not v32_df.empty:
            df_80 = get_stratified_selection(v32_df, price_limit=80)
            display_graded_tables(df_80, "80")
        else: st.warning("等待資料載入...")

    # === Tab 2: 50元以下 ===
    with tab_50:
        if not v32_df.empty:
            df_50 = get_stratified_selection(v32_df, price_limit=50)
            display_graded_tables(df_50, "50")
        else: st.warning("等待資料載入...")

    # === Tab 3: 庫存管理 ===
    with tab_inv:
        st.subheader("📝 庫存交易管理")
        name_map = dict(zip(v32_df['代號'], v32_df['名稱'])) if not v32_df.empty else {}
        input_key = st.session_state['input_key_counter']
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 📥 **買入**")
            edited_buy = st.data_editor(pd.DataFrame([{"股票代號": "", "持有股數": 1000, "買入均價": 0.0}]), num_rows="dynamic", key=f"buy_{input_key}", hide_index=True)
        with c2:
            st.markdown("##### 📤 **賣出**")
            edited_sell = st.data_editor(pd.DataFrame([{"股票代號": "", "持有股數": 1000}]), num_rows="dynamic", key=f"sell_{input_key}", hide_index=True)
        
        if st.button("💾 執行交易", type="primary"):
            current_inv = st.session_state['inventory'].copy()
            has_update = False
            for _, r in edited_buy.iterrows():
                if r['股票代號'] and r['持有股數'] > 0:
                    code, shares, price = str(r['股票代號']).strip(), int(r['持有股數']), float(r['買入均價'])
                    match = current_inv[current_inv['股票代號'] == code]
                    if not match.empty:
                        idx = match.index[0]
                        old_s, old_p = float(current_inv.at[idx, '持有股數']), float(current_inv.at[idx, '買入均價'])
                        current_inv.at[idx, '持有股數'], current_inv.at[idx, '買入均價'] = old_s + shares, round(((old_s*old_p)+(shares*price))/(old_s+shares), 2)
                    else: current_inv = pd.concat([current_inv, pd.DataFrame([{'股票代號': code, '持有股數': shares, '買入均價': price}])], ignore_index=True)
                    has_update = True
            for _, r in edited_sell.iterrows():
                if r['股票代號'] and r['持有股數'] > 0:
                    code, shares = str(r['股票代號']).strip(), int(r['持有股數'])
                    match = current_inv[current_inv['股票代號'] == code]
                    if not match.empty:
                        idx = match.index[0]
                        if current_inv.at[idx, '持有股數'] > shares: current_inv.at[idx, '持有股數'] -= shares
                        else: current_inv = current_inv.drop(idx)
                        has_update = True
            
            if has_update:
                st.session_state['inventory'] = current_inv
                save_holdings(current_inv)
                st.session_state['input_key_counter'] += 1
                st.rerun()

        st.divider()
        if not st.session_state['inventory'].empty:
            inv_df = st.session_state['inventory'].copy()
            inv_rt = get_realtime_quotes(inv_df['股票代號'].astype(str).tolist())
            res = []
            score_map = v32_df.set_index('代號')['攻擊分'].to_dict() if not v32_df.empty else {}
            
            for _, r in inv_df.iterrows():
                code, qty, cost = str(r['股票代號']), float(r['持有股數']), float(r['買入均價'])
                curr = inv_rt.get(code, {}).get('即時價', cost) 
                name = name_map.get(code, code)
                sc = score_map.get(code, 0)
                pl = (curr - cost) * qty
                roi = (pl/(cost*qty)*100) if cost else 0
                
                # 建議操作邏輯
                if roi < -10: action = "🛑 停損"
                elif sc >= 60: action = "🟢 續抱"
                else: action = "🔻 賣出"

                res.append({
                    '代號': code, '名稱': name, '即時價': curr, 
                    '損益': pl, '報酬率%': roi, 
                    '攻擊分': sc, '建議操作': action, 
                    '持有股數': qty, '購入均價': cost
                })
            
            df_res = pd.DataFrame(res)
            c1, c2, c3 = st.columns(3)
            c1.metric("總成本", f"${(df_res['購入均價']*df_res['持有股數']).sum():,.0f}")
            c2.metric("總損益", f"${df_res['損益'].sum():,.0f}", delta=f"{df_res['損益'].sum():,.0f}")
            c3.metric("總市值", f"${(df_res['即時價']*df_res['持有股數']).sum():,.0f}")
            
            st.dataframe(
                df_res[['代號', '名稱', '持有股數', '購入均價', '即時價', '損益', '報酬率%', '攻擊分', '建議操作']].style
                .format({'購入均價':'{:.2f}', '即時價':'{:.2f}', '損益':'{:+,.0f}', '報酬率%':'{:+.2f}%', '攻擊分':'{:.0f}'})
                .map(color_surplus, subset=['損益','報酬率%'])
                .map(color_action, subset=['建議操作']), 
                use_container_width=True, hide_index=True
            )
        else: st.info("目前無庫存")

if __name__ == "__main__":
    main()
