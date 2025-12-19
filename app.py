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
    page_title="V32 戰情室 (Pure Stock)",
    layout="wide",
    page_icon="💎"
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

# --- 核心：V32 技術指標運算 (穩定版邏輯) ---
def calculate_indicators(hist):
    if len(hist) < 60: return 0, 0, "Data Insufficient"

    # 1. 準備數據
    close = hist['Close']
    vol = hist['Volume']
    high = hist['High']
    open_p = hist['Open']
    
    # 均線
    ma5 = close.rolling(5).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma20_prev = close.rolling(20).mean().iloc[-2]
    ma60 = close.rolling(60).mean().iloc[-1]
    
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
    
    # 1. 多日趨勢
    if close.iloc[-1] > ma20: t_score += 5        # 站上月線
    if ma20 > ma20_prev: t_score += 5             # 月線翻揚
    if ma5 > ma20 and ma20 > ma60: t_score += 10  # 多頭排列
    
    # 2. 動能
    if rsi_now > 50: t_score += 5                 # RSI 強勢
    if rsi_now > 70: t_score += 5                 # RSI 過熱區
    if macd_now > signal_now: t_score += 5        # MACD 金叉
    
    # 3. 結構
    high_20 = high.rolling(20).max().iloc[-2]
    if close.iloc[-1] > high_20: t_score += 10    # 突破 20 日新高

    # B. 量能分 (Volume)
    v_score = 60
    
    current_vol = vol.iloc[-1]
    # 1. 均量突破
    if current_vol > vol_ma20: v_score += 10      # 大於月均量
    if current_vol > vol_ma5: v_score += 10       # 大於週均量
    
    # 2. 量價配合
    is_red = close.iloc[-1] > open_p.iloc[-1]
    vol_increase = current_vol > vol.iloc[-2]
    if is_red and vol_increase: v_score += 15     # 價漲量增
    
    # 3. 爆量
    if current_vol > vol_ma20 * 1.5: v_score += 5 # 放量 1.5 倍

    # 上限防呆
    t_score = min(100, t_score)
    v_score = min(100, v_score)
    
    # 趨勢標記
    trend = "Rising" if (close.iloc[-1] > ma5 and ma5 > ma20) else "Consolidating"
    
    return t_score, v_score, trend

# --- 批次運算引擎 ---
@st.cache_data(ttl=3600)
def run_v32_engine(ticker_list):
    results = []
    p_bar = st.progress(0)
    status = st.empty()
    total = len(ticker_list)
    
    for i, row in enumerate(ticker_list):
        symbol = str(row['代號'])
        name = str(row.get('名稱', ''))
        
        status.text(f"正在分析: {symbol} {name} ({i+1}/{total})...")
        p_bar.progress((i + 1) / total)
        
        try:
            stock = yf.Ticker(f"{symbol}.TW")
            hist = stock.history(period="3mo")
            
            if not hist.empty:
                t_s, v_s, tr = calculate_indicators(hist)
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
        return pd.DataFrame(columns=["股票代號", "買入均價", "持有股數"])

def save_holdings(df):
    try:
        token = st.secrets["general"]["GITHUB_TOKEN"]
        g = Github(token)
        repo = g.get_repo(REPO_KEY)
        csv_content = df.to_csv(index=False)
        try:
            contents = repo.get_contents(FILE_PATH)
            repo.update_file(contents.path, f"Update {get_taiwan_time()}", csv_content, contents.sha)
            st.success("✅ 儲存成功！")
        except:
            repo.create_file(FILE_PATH, "Create holdings.csv", csv_content)
            st.success("✅ 建立並儲存成功！")
    except Exception as e:
        st.error(f"❌ 儲存失敗: {e}")

# --- 篩選與排序邏輯 ---

def get_stratified_selection(df):
    """分層精選邏輯"""
    if df.empty: return df, []
    # 硬指標
    mask = (df['技術分'] >= 88) & (df['量能分'] >= 82) & (df['趨勢'] == 'Rising') & (df['總分'] >= 86) & (df['總分'] <= 92)
    filtered = df[mask].copy()
    if filtered.empty: return pd.DataFrame(), ["無符合條件標的"]
    
    # 分層取前5
    b_a = filtered[(filtered['總分'] >= 90) & (filtered['總分'] <= 92)].sort_values('總分', ascending=False).head(5)
    b_b = filtered[(filtered['總分'] >= 88) & (filtered['總分'] < 90)].sort_values('總分', ascending=False).head(5)
    b_c = filtered[(filtered['總分'] >= 86) & (filtered['總分'] < 88)].sort_values('總分', ascending=False).head(5)
    
    final = pd.concat([b_a, b_b, b_c])
    stats = [f"90-92: {len(b_a)}", f"88-90: {len(b_b)}", f"86-88: {len(b_c)}"]
    return final, stats

def get_raw_top10(df):
    """原始分數 Top 10"""
    if df.empty: return df
    return df.sort_values(by='總分', ascending=False).head(10)

# --- 主程式 ---
def main():
    st.title("💎 V32 戰情室 (Pure Stock)")
    st.caption(f"最後更新: {get_taiwan_time()}")
    
    v32_df, err = load_and_process_data()
    
    if err: st.error(err)

    # 🔥 關鍵修改：全域過濾 (剔除 ETF, KY, DR, 債券, 特別股)
    if not v32_df.empty:
        # 1. 標記類別
        v32_df['cat'] = v32_df.apply(lambda r: 'Special' if ('債' in str(r.get('名稱')) or 'KY' in str(r.get('名稱')) or str(r['代號']).startswith(('00','91')) or str(r['代號'])[-1].isalpha() or (len(str(r['代號']))>4 and str(r['代號']).isdigit())) else 'General', axis=1)
        
        # 2. 直接過濾：只保留 'General'
        v32_df = v32_df[v32_df['cat'] == 'General']
        
        if v32_df.empty:
            st.warning("⚠️ 過濾後沒有任何一般個股 (General Stocks)！")

    # 建立主分頁
    tab_strat, tab_raw, tab_inv = st.tabs(["🎯 分層精選 Top 15", "🏆 原始分數 Top 10", "💼 庫存管理"])
    
    fmt_score = {'收盤':'{:.2f}', '技術分':'{:.0f}', '量能分':'{:.0f}', '總分':'{:.1f}'}

    # === Tab 1: 分層精選 (Stratified) ===
    with tab_strat:
        if not v32_df.empty:
            # 直接使用過濾後的 v32_df (已確保全是一般個股)
            final_df, stats = get_stratified_selection(v32_df)
            
            st.info(f"🎯 純個股分佈：{' | '.join(stats)}")
            if not final_df.empty:
                # 這裡使用了熱力圖 (需要 matplotlib)
                st.dataframe(final_df[['代號','名稱','收盤','技術分','量能分','總分','趨勢']].style.format(fmt_score).background_gradient(subset=['總分'], cmap='Reds'), hide_index=True, use_container_width=True)
            else:
                st.warning("無符合條件的一般個股。")
        else:
            st.warning("暫無資料")

    # === Tab 2: 原始 Top 10 (Raw) ===
    with tab_raw:
        st.markdown("### 🏆 原始分數霸榜 (Top 10)")
        st.caption("排除 ETF/KY/DR 後，全市場最強 10 檔個股。")
        
        if not v32_df.empty:
            raw_df = get_raw_top10(v32_df)
            if not raw_df.empty:
                st.dataframe(raw_df[['代號','名稱','收盤','總分','技術分','量能分']].style.format(fmt_score).background_gradient(subset=['總分'], cmap='Reds'), hide_index=True, use_container_width=True)
            else:
                st.info("無資料")
        else:
            st.warning("暫無資料")

    # === Tab 3: 庫存管理 ===
    with tab_inv:
        st.subheader("📝 庫存編輯器")
        if 'editor_data' not in st.session_state:
            st.session_state['editor_data'] = load_holdings()
            
        edited = st.data_editor(
            st.session_state['editor_data'],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "股票代號": st.column_config.TextColumn("代號", required=True),
                "買入均價": st.column_config.NumberColumn("均價", format="%.2f"),
                "持有股數": st.column_config.NumberColumn("股數", step=1000)
            }, key="inv_editor"
        )
        if st.button("💾 儲存變更"):
            save_holdings(edited)
            st.rerun()
            
        st.divider()
        if not edited.empty and not v32_df.empty:
            res = []
            for _, r in edited.iterrows():
                if not r['股票代號']: continue
                code = str(r['股票代號'])
                qty = float(r['持有股數'] or 0)
                cost = float(r['買入均價'] or 0)
                
                # 從已過濾的清單找 (如果庫存是 ETF，這裡會找不到)
                # 所以我們需要一個 fallback 機制去抓現價
                match = v32_df[v32_df['代號']==code]
                if not match.empty:
                    curr = match.iloc[0]['收盤']
                    nm = match.iloc[0]['名稱']
                    sc = match.iloc[0]['總分']
                else:
                    # 榜外或被過濾掉的 (如 ETF 庫存)
                    try:
                        t = yf.Ticker(f"{code}.TW")
                        h = t.history(period='1d')
                        curr = h['Close'].iloc[-1] if not h.empty else 0
                        nm = code; sc = 0
                    except: curr=0; nm=code; sc=0
                
                val = curr * qty
                c_tot = cost * qty
                pl = val - c_tot
                roi = (pl/c_tot*100) if c_tot>0 else 0
                
                res.append({'代號':code, '名稱':nm, '現價':curr, '成本':cost, '股數':qty, '損益':pl, '報酬率%':roi, 'V32分': f"{sc:.1f}" if sc>0 else "榜外"})
            
            if res:
                df_res = pd.DataFrame(res)
                c1, c2, c3 = st.columns(3)
                c1.metric("總成本", f"${(df_res['成本']*df_res['股數']).sum():,.0f}")
                c2.metric("總損益", f"${df_res['損益'].sum():,.0f}")
                c3.metric("總市值", f"${(df_res['現價']*df_res['股數']).sum():,.0f}")
                
                st.dataframe(df_res.style.map(color_surplus, subset=['損益','報酬率%']).format({'現價':'{:.2f}','損益':'{:+,.0f}','報酬率%':'{:+.2f}%'}), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
