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
    if "🔴" in val_str or "停損" in val_str:
        return 'color: #ffffff; background-color: #d32f2f; font-weight: bold; padding: 5px; border-radius: 5px;' # 紅底白字
    elif "🟡" in val_str or "停利" in val_str:
        return 'color: #000000; background-color: #ffeb3b; font-weight: bold; padding: 5px; border-radius: 5px;' # 黃底黑字
    elif "🟢" in val_str or "續抱" in val_str:
        return 'color: #ffffff; background-color: #2e7d32; font-weight: bold; padding: 5px; border-radius: 5px;' # 綠底白字
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
    results = []
    for code, group in raw_df.groupby('Code'):
        res = calculate_v32_score(group)
        if res:
            res.update({'代號': code, '名稱': group['Name'].iloc[-1]})
            results.append(res)
    return pd.DataFrame(results), raw_df, None

# --- 強化的即時報價模組 (三層備援) ---
def fetch_price_twse(code):
    """第一層：嘗試從證交所/櫃買中心抓取 (最準)"""
    try:
        stock = twstock.Realtime(code)
        if stock.realtime['latest_trade_price']:
            return float(stock.realtime['latest_trade_price'])
        # 如果還沒開盤或抓不到，嘗試抓開盤價或昨收
        elif stock.realtime['open']:
             return float(stock.realtime['open'])
        return None
    except:
        return None

def fetch_price_yahoo(code):
    """第二層：嘗試從 Yahoo 股市抓取"""
    try:
        # 簡單爬蟲或使用其他 library，這裡示範用 requests 抓取 HTML 結構變動大，暫略
        # 改用 yfinance 的快速模式作為替代 Yahoo 來源 (其實 yfinance 也是爬 yahoo)
        ticker = yf.Ticker(f"{code}.TW")
        data = ticker.history(period="1d", interval="1m")
        if not data.empty:
            return float(data['Close'].iloc[-1])
        return None
    except:
        return None

def fetch_price_google_yf(code):
    """第三層：Yfinance (備用)"""
    try:
        # 這裡作為最後手段
        data = yf.download(f"{code}.TW", period="1d", interval="1m", progress=False)
        if not data.empty:
             return float(data['Close'].iloc[-1])
        return None
    except:
        return None

# 我們不使用 cache_data，而是使用 st.session_state 手動控制更新頻率
def get_realtime_quotes_robust(code_list):
    if not code_list: return {}
    clean_codes = [str(c).strip().split('.')[0] for c in code_list]
    realtime_data = {}
    
    # 建立進度條，因為單檔抓取比較慢
    progress_bar = st.progress(0)
    total = len(clean_codes)
    
    for idx, code in enumerate(clean_codes):
        price = None
        
        # 1. 嘗試 TWSE
        price = fetch_price_twse(code)
        
        # 2. 失敗則嘗試 Yahoo (這裡直接用 yf 作為 Yahoo 介面，因為它是最穩定的 Yahoo API wrapper)
        if price is None:
            price = fetch_price_yahoo(code)
            
        # 3. 還是失敗，嘗試 Google (這裡邏輯上 yfinance 已涵蓋，若有專門 google API 可替換)
        # 暫時均以 yfinance 作為後兩道防線，但參數不同
        if price is None:
             price = fetch_price_google_yf(code)

        if price is not None:
            realtime_data[code] = {'即時價': round(price, 2)}
        
        progress_bar.progress((idx + 1) / total)
        
    progress_bar.empty()
    return realtime_data

def merge_realtime_data(df, realtime_dict=None):
    if df.empty: return df
    
    # 如果有傳入外部即時資料字典，就優先使用
    if realtime_dict:
        df['即時價'] = df['代號'].map(lambda x: realtime_dict.get(x, {}).get('即時價', np.nan))
    else:
        # 否則使用舊有邏輯(但不建議，因為這會觸發舊的 cache)
        pass 
        
    df['即時價'] = df['即時價'].fillna(df['收盤'])
    return df

# --- 籌碼分析 ---
def get_chip_analysis(symbol_list):
    chip_data = []
    dl = DataLoader()
    p_bar = st.progress(0)
    for i, symbol in enumerate(symbol_list):
        p_bar.progress((i + 1) / len(symbol_list))
        try:
            df = dl.taiwan_stock_institutional_investors(stock_id=symbol, start_date=(datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d'))
            if df.empty:
                chip_data.append({'代號': symbol, '投信(張)': 0, '外資(張)': 0, '主力動向': '🟡 一般輪動'})
            else:
                latest = df[df['date'] == df['date'].iloc[-1]]
                f_buy = int((latest[latest['name'].str.contains('Foreign')]['buy'].sum() - latest[latest['name'].str.contains('Foreign')]['sell'].sum()) // 1000)
                t_buy = int((latest[latest['name'] == 'Investment_Trust']['buy'].sum() - latest[latest['name'] == 'Investment_Trust']['sell'].sum()) // 1000)
                
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
        repo.update_file(contents.path, f"Update {get_taiwan_time()}", csv_content, contents.sha)
    except: pass

# --- Tab 1 & 2 表格渲染 ---
def display_v32_tables(df, price_limit, suffix):
    filtered = df[(df['收盤'] <= price_limit) & (df['攻擊分'] >= 86) & (df['攻擊分'] <= 92)].sort_values('攻擊分', ascending=False)
    if filtered.empty: return st.warning("無符合標的")

    df_s_pre = filtered[(filtered['攻擊分'] >= 90) & (filtered['攻擊分'] <= 92)].head(10)
    df_a_pre = filtered[(filtered['攻擊分'] >= 88) & (filtered['攻擊分'] < 90)].head(10)
    df_b_pre = filtered[(filtered['攻擊分'] >= 86) & (filtered['攻擊分'] < 88)].head(10)
    target_codes = pd.concat([df_s_pre, df_a_pre, df_b_pre])['代號'].tolist()

    if st.button(f"🚀 籌碼掃描 (Top {len(target_codes)} 檔)", key=f"scan_{suffix}"):
        chip_df = get_chip_analysis(target_codes)
        filtered = pd.merge(filtered, chip_df, on='代號', how='left')
    
    # 一般列表這裡還是用舊的快速抓取(或不抓即時，只顯示收盤，保持效能)，或可選擇要不要更新
    # 這裡暫時維持原樣，只用收盤價填充即時價，避免外部列表卡太久
    filtered['即時價'] = filtered['收盤']
    
    base_cols = ['代號','名稱','即時價','技術分','量能分','攻擊分']
    if '主力動向' in filtered.columns: base_cols += ['主力動向', '投信(張)', '外資(張)']
    fmt = {'即時價':'{:.2f}', '攻擊分':'{:.1f}', '技術分':'{:.0f}', '量能分':'{:.0f}', '外資(張)': '{:,.0f}', '投信(張)': '{:,.0f}'}

    for title, score_range in [
        ("👑 S 級主力區 (90-92分)", (90, 92)),
        ("🚀 A 級蓄勢區 (88-90分)", (88, 90)),
        ("👀 B 級觀察區 (86-88分)", (86, 88))
    ]:
        st.subheader(title)
        sub = filtered[(filtered['攻擊分'] >= score_range[0]) & (filtered['攻擊分'] <= score_range[1])].head(10)
        if not sub.empty:
            st.dataframe(sub[base_cols].style.format(fmt).background_gradient(subset=['攻擊分'], cmap=cmap_pastel_red, vmin=86, vmax=92).background_gradient(subset=['技術分'], cmap=cmap_pastel_blue, vmin=60, vmax=100).background_gradient(subset=['量能分'], cmap=cmap_pastel_green, vmin=60, vmax=100), hide_index=True, use_container_width=True)
        else: st.caption("暫無標的")
        st.divider()

# --- 主程式 ---
def main():
    st.title("⚔️ V32 戰情室 (Dual Core)")
    if 'inventory' not in st.session_state: st.session_state['inventory'] = load_holdings()
    
    # 初始化即時報價的 session state
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
        
        # --- 刷新按鈕邏輯 ---
        col_btn, col_info = st.columns([1, 4])
        with col_btn:
            now = time.time()
            time_diff = now - st.session_state['last_update_time']
            btn_label = "🔄 更新即時股價"
            btn_disabled = False
            
            if time_diff < 60:
                btn_label = f"⏳ 冷卻中 ({int(60 - time_diff)}s)"
                btn_disabled = True
            
            if st.button(btn_label, disabled=btn_disabled, type="primary"):
                if not st.session_state['inventory'].empty:
                    with st.spinner("🚀 正從證交所/Yahoo/Google 同步最新報價..."):
                        codes = st.session_state['inventory']['股票代號'].tolist()
                        # 執行強制更新
                        fresh_quotes = get_realtime_quotes_robust(codes)
                        st.session_state['realtime_quotes'] = fresh_quotes
                        st.session_state['last_update_time'] = time.time()
                        st.rerun() # 重新載入頁面以顯示新數據
        
        with col_info:
            if st.session_state['last_update_time'] > 0:
                last_time_str = datetime.fromtimestamp(st.session_state['last_update_time']).strftime('%H:%M:%S')
                st.caption(f"最後更新時間: {last_time_str}")

        # 建立快速查詢字典
        name_map = dict(zip(v32_df['代號'], v32_df['名稱'])) if not v32_df.empty else {}
        score_map = v32_df.set_index('代號')['攻擊分'].to_dict() if not v32_df.empty else {}
        # 安全取得 20MA
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
            
            # 使用 session state 中的即時報價 (如果有的話)
            saved_quotes = st.session_state.get('realtime_quotes', {})
            
            res = []
            for _, r in inv_df.iterrows():
                code = str(r['股票代號'])
                # 優先使用按鈕更新後的報價，沒有的話用買入價暫代 (或收盤價)
                curr = saved_quotes.get(code, {}).get('即時價', r['買入均價'])
                
                # 如果完全沒有即時價更新過，且買入價也為0 (異常)，嘗試從 v32_df 找收盤價
                if curr == 0 and not v32_df.empty:
                     curr = v32_df[v32_df['代號']==code]['收盤'].values[0] if not v32_df[v32_df['代號']==code].empty else 0

                buy_price = r['買入均價']
                qty = r['持有股數']
                
                pl = (curr - buy_price) * qty
                roi = (pl / (buy_price * qty) * 100) if buy_price > 0 else 0
                
                # 從字典中獲取 攻擊分 與 20MA
                sc = score_map.get(code, 0)
                ma20 = ma20_map.get(code, 0)
                
                # --- 紅綠燈判斷邏輯 (邏輯不變，只用於生成 Action 字串) ---
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
            
            # 在這裡顯示時移除 '20MA' 欄位，但邏輯中已經使用過了
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
