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

# --- 篩選邏輯 ---

def get_stratified_selection(df):
    """分層精選邏輯 (Strict)"""
    if df.empty: return df, []
    
    # 1. 硬指標
    mask = (df['技術分'] >= 88) & (df['量能分'] >= 82) & (df['趨勢'] == 'Rising') & (df['總分'] >= 86) & (df['總分'] <= 92)
    filtered = df[mask].copy()
    
    if filtered.empty: return pd.DataFrame(), ["無符合硬指標標的"]
    
    # 2. 分層 (Bucketing)
    b_a = filtered[(filtered['總分'] >= 90) & (filtered['總分'] <= 92)].sort_values('總分', ascending=False).head(5)
    b_b = filtered[(filtered['總分'] >= 88) & (filtered['總分'] < 90)].sort_values('總分', ascending=False).head(5)
    b_c = filtered[(filtered['總分'] >= 86) & (filtered['總分'] < 88)].sort_values('總分', ascending=False).head(5)
    
    final = pd.concat([b_a, b_b, b_c])
    stats = [f"90-92: {len(b_a)}", f"88-90: {len(b_b)}", f"86-88: {len(b_c)}"]
    return final, stats

def get_raw_top10(df):
    """原始分數 Top 10 (Raw Logic)"""
    if df.empty: return df
    return df.sort_values(by='總分', ascending=False).head(10)

# --- 主程式 ---
def main():
    st.title("🔥 V32 戰情室 (Slope Logic)")
    st.caption(f"最後更新: {get_taiwan_time()}")
    
    v32_df, err = load_and_process_data()
    
    tab_strat, tab_raw, tab_inv = st.tabs(["🎯 分層精選 Top 15", "🏆 原始分數 Top 10", "💼 庫存管理"])
    
    if not v32_df.empty:
        v32_df['cat'] = v32_df.apply(lambda r: 'Special' if ('債' in str(r.get('名稱')) or 'KY' in str(r.get('名稱')) or str(r['代號']).startswith(('00','91')) or str(r['代號'])[-1].isalpha() or (len(str(r['代號']))>4 and str(r['代號']).isdigit())) else 'General', axis=1)

    fmt_score = {'收盤':'{:.2f}', '技術分':'{:.0f}', '量能分':'{:.0f}', '總分':'{:.1f}'}

    # === Tab 1: 分層精選 (Stratified) ===
    with tab_strat:
        if err: st.error(err)
        if not v32_df.empty:
            gen, stats_g = get_stratified_selection(v32_df[v32_df['cat']=='General'])
            spec, stats_s = get_stratified_selection(v32_df[v32_df['cat']=='Special'])
            
            t1, t2 = st.tabs(["🏢 一般個股", "📊 特殊/ETF"])
            with t1:
                st.info(f"分佈：{' | '.join(stats_g)}")
                if not gen.empty: st.dataframe(gen[['代號','名稱','收盤','技術分','量能分','總分','趨勢']].style.format(fmt_score), hide_index=True, use_container_width=True)
                else: st.warning("無符合條件標的")
            with t2:
                st.info(f"分佈：{' | '.join(stats_s)}")
                if not spec.empty: st.dataframe(spec[['代號','名稱','收盤','技術分','量能分','總分','趨勢']].style.format(fmt_score), hide_index=True, use_container_width=True)
                else: st.warning("無符合條件標的")
        else: st.warning("暫無資料")

    # === Tab 2: 原始 Top 10 (Raw) ===
    with tab_raw:
        st.markdown("### 🏆 全市場原始分數霸榜 (Top 10)")
        if not v32_df.empty:
            raw_gen = get_raw_top10(v32_df[v32_df['cat']=='General'])
            raw_spec = get_raw_top10(v32_df[v32_df['cat']=='Special'])
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🏢 一般個股")
                if not raw_gen.empty:
                    # 使用普通的 dataframe 顯示，避免 matplotlib 依賴報錯
                    st.dataframe(raw_gen[['代號','名稱','收盤','總分','技術分','量能分']].style.format(fmt_score), hide_index=True, use_container_width=True)
            with c2:
                st.subheader("📊 特殊/ETF")
                if not raw_spec.empty:
                    st.dataframe(raw_spec[['代號','名稱','收盤','總分','技術分','量能分']].style.format(fmt_score), hide_index=True, use_container_width=True)
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
                
                match = v32_df[v32_df['代號']==code]
                if not match.empty:
                    curr = match.iloc[0]['收盤']
                    nm = match.iloc[0]['名稱']
                    sc = match.iloc[0]['總分']
                else:
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
