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
    page_title="V32 戰情室 (Evolution Ver.)",
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

def color_stability(val):
    """
    視覺化 C 模組：
    1/5, 2/5 -> 剛起步/不穩 (橘色)
    3/5, 4/5, 5/5 -> 穩定 (綠色)
    """
    if not isinstance(val, str): return ''
    score = int(val.split('/')[0])
    if score <= 2:
        return 'color: #E65100; font-weight: bold;' # 橘色：注意風險，可能是假突破
    elif score >= 3:
        return 'color: #2E7D32; font-weight: bold;' # 綠色：結構紮實
    return ''

@st.cache_data(ttl=86400)
def fetch_name_from_web(symbol):
    try:
        t = yf.Ticker(f"{symbol}.TW")
        return t.info.get('shortName') or t.info.get('longName') or symbol
    except:
        return symbol

# --- 核心：V32 技術指標運算 (B + C 進化版) ---
def calculate_indicators(hist):
    # 需要至少 65 天資料才能確保 MA60 和回溯運算不出錯
    if len(hist) < 65: return 0, 0, "Data Insufficient", "0/5"

    # 1. 預先計算所有指標 (向量化運算，一次算完)
    close = hist['Close']
    vol = hist['Volume']
    high = hist['High']
    open_p = hist['Open']
    
    # 均線
    ma5_s = close.rolling(5).mean()
    ma20_s = close.rolling(20).mean()
    ma60_s = close.rolling(60).mean()
    
    # RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi_s = 100 - (100 / (1 + rs))

    # MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd_s = exp1 - exp2
    signal_s = macd_s.ewm(span=9, adjust=False).mean()

    # 均量
    vol_ma5_s = vol.rolling(5).mean()
    vol_ma20_s = vol.rolling(20).mean()
    
    # 20日高點
    high_20_s = high.rolling(20).max()

    # ---------------------------------------------------------
    # 迴圈回溯：計算過去 7 天的「原始分數」
    # 用途：為了執行 B (Smoothing) 和 C (Stability)
    # ---------------------------------------------------------
    raw_scores = [] 
    
    # 取最後 7 天的 index 位置 (例如: -7, -6, ... -1)
    # 為什麼要 7 天？因為要算 Smoothing 需要前一天，要算 Stability 需要前 5 天
    lookback_indices = range(-7, 0) 

    for i in lookback_indices:
        # 在第 i 天的當下數據
        c_now = close.iloc[i]
        ma5 = ma5_s.iloc[i]
        ma20 = ma20_s.iloc[i]
        ma20_prev = ma20_s.iloc[i-1] # 當時的前一天
        ma60 = ma60_s.iloc[i]
        rsi_now = rsi_s.iloc[i]
        macd_now = macd_s.iloc[i]
        sig_now = signal_s.iloc[i]
        high_20_prev = high_20_s.iloc[i-1] # 突破前一日的20日高
        
        v_now = vol.iloc[i]
        v_prev = vol.iloc[i-1]
        v_ma20 = vol_ma20_s.iloc[i]
        v_ma5 = vol_ma5_s.iloc[i]
        o_now = open_p.iloc[i]

        # --- A. 技術分 (Technical) ---
        t_score = 60
        if c_now > ma20: t_score += 5         # 站上月線
        if ma20 > ma20_prev: t_score += 5     # 月線翻揚
        if ma5 > ma20 and ma20 > ma60: t_score += 10 # 多頭排列
        
        if rsi_now > 50: t_score += 5         # RSI 強勢
        if rsi_now > 70: t_score += 5         # RSI 過熱區 (強勢特徵)
        if macd_now > sig_now: t_score += 5   # MACD 金叉
        
        if c_now > high_20_prev: t_score += 10 # 突破 20 日新高

        # --- B. 量能分 (Volume) ---
        v_score = 60
        if v_now > v_ma20: v_score += 10      # 大於月均量
        if v_now > v_ma5: v_score += 10       # 大於週均量
        
        is_red = c_now > o_now
        vol_increase = v_now > v_prev
        if is_red and vol_increase: v_score += 15 # 價漲量增
        
        if v_now > v_ma20 * 1.5: v_score += 5 # 放量 1.5 倍

        # 上限與加權
        t_score = min(100, t_score)
        v_score = min(100, v_score)
        
        # 算出「當日原始總分」
        daily_total = (t_score * 0.7) + (v_score * 0.3)
        raw_scores.append(daily_total)

    # ---------------------------------------------------------
    # 3. 模組實裝
    # ---------------------------------------------------------
    
    # 取得今日與昨日的原始分 (list 的最後兩個)
    raw_today = raw_scores[-1]
    raw_yesterday = raw_scores[-2]

    # [模組 B] 連續化分數 (Smoothing)
    # 70% 看今天，30% 看昨天的慣性
    final_v32_score = (raw_today * 0.7) + (raw_yesterday * 0.3)

    # [模組 C] 穩定度 (Stability)
    # 檢查過去 5 天 (含今天)，有幾天原始分 > 70
    # list slicing: raw_scores[-5:] 取最後 5 筆
    last_5_days = raw_scores[-5:]
    stability_count = sum(1 for s in last_5_days if s >= 70)
    stability_str = f"{stability_count}/5"

    # 趨勢標記 (沿用舊邏輯，僅供參考)
    trend = "Rising" if (close.iloc[-1] > ma5_s.iloc[-1] and ma5_s.iloc[-1] > ma20_s.iloc[-1]) else "Consolidating"
    
    # 回傳：技術分(今日), 量能分(今日), V32總分(B模組), 穩定度(C模組)
    return t_score, v_score, final_v32_score, stability_str

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
            # 抓取 3 個月資料以確保有足夠的歷史做回溯計算
            hist = stock.history(period="3mo")
            
            if not hist.empty:
                # 呼叫新的指標計算函數
                t_s, v_s, final_s, stab = calculate_indicators(hist)
                
                results.append({
                    '代號': symbol, '名稱': name,
                    '收盤': hist['Close'].iloc[-1],
                    '成交量': hist['Volume'].iloc[-1],
                    '技術分': t_s,   # 顯示用 (今日)
                    '量能分': v_s,   # 顯示用 (今日)
                    'V32總分': final_s, # 排序用 (B模組結果)
                    '穩定度': stab   # 決策用 (C模組結果)
                })
            else:
                # 沒資料的處理
                pass
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

# --- GitHub 庫存存取 (維持原樣) ---
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

# --- 篩選與排序邏輯 (更新為 V32 總分) ---
def get_stratified_selection(df):
    """分層精選邏輯"""
    if df.empty: return df, []
    
    # 核心過濾：使用「V32總分」(B模組) 而非舊總分
    # 門檻：V32總分 >= 86, 且穩定度不能太差 (這裡暫不強制過濾穩定度，讓使用者自己看)
    # 技術分與量能分維持一定的基礎門檻
    mask = (df['技術分'] >= 80) & (df['量能分'] >= 60) & (df['V32總分'] >= 86) & (df['V32總分'] <= 92)
    
    filtered = df[mask].copy()
    if filtered.empty: return pd.DataFrame(), ["無符合條件標的"]
    
    # 分層取前 5 (根據 B 模組分數排序)
    b_a = filtered[(filtered['V32總分'] >= 90) & (filtered['V32總分'] <= 92)].sort_values('V32總分', ascending=False).head(5)
    b_b = filtered[(filtered['V32總分'] >= 88) & (filtered['V32總分'] < 90)].sort_values('V32總分', ascending=False).head(5)
    b_c = filtered[(filtered['V32總分'] >= 86) & (filtered['V32總分'] < 88)].sort_values('V32總分', ascending=False).head(5)
    
    final = pd.concat([b_a, b_b, b_c])
    stats = [f"90-92: {len(b_a)}", f"88-90: {len(b_b)}", f"86-88: {len(b_c)}"]
    return final, stats

def get_raw_top10(df):
    """V32 總分 Top 10"""
    if df.empty: return df
    return df.sort_values(by='V32總分', ascending=False).head(10)

# --- 主程式 ---
def main():
    st.title("💎 V32 戰情室 (Evolution Ver.)")
    st.caption(f"最後更新: {get_taiwan_time()} | 核心: B(連續化) + C(穩定度)")
    
    v32_df, err = load_and_process_data()
    
    if err: st.error(err)

    # 過濾：只保留 'General'
    if not v32_df.empty:
        v32_df['cat'] = v32_df.apply(lambda r: 'Special' if ('債' in str(r.get('名稱')) or 'KY' in str(r.get('名稱')) or str(r['代號']).startswith(('00','91')) or str(r['代號'])[-1].isalpha() or (len(str(r['代號']))>4 and str(r['代號']).isdigit())) else 'General', axis=1)
        v32_df = v32_df[v32_df['cat'] == 'General']
        
        if v32_df.empty:
            st.warning("⚠️ 過濾後沒有任何一般個股！")

    # 建立主分頁
    tab_strat, tab_raw, tab_inv = st.tabs(["🎯 分層精選 Top 15", "🏆 V32 總分 Top 10", "💼 庫存管理"])
    
    fmt_score = {'收盤':'{:.2f}', '技術分':'{:.0f}', '量能分':'{:.0f}', 'V32總分':'{:.1f}'}

    # === Tab 1: 分層精選 (Stratified) ===
    with tab_strat:
        if not v32_df.empty:
            final_df, stats = get_stratified_selection(v32_df)
            
            st.info(f"🎯 純個股分佈：{' | '.join(stats)}")
            if not final_df.empty:
                # 這裡加入了 '穩定度' 的顏色顯示
                st.dataframe(
                    final_df[['代號','名稱','收盤','V32總分','穩定度','技術分','量能分']]
                    .style
                    .format(fmt_score)
                    .background_gradient(subset=['V32總分'], cmap='Reds')
                    .map(color_stability, subset=['穩定度']), # 應用 C 模組視覺化
                    hide_index=True, 
                    use_container_width=True
                )
            else:
                st.warning("無符合條件的一般個股。")
        else:
            st.warning("暫無資料")

    # === Tab 2: V32 Top 10 ===
    with tab_raw:
        st.markdown("### 🏆 V32 總分霸榜 (Top 10)")
        st.caption("結合 B(爬坡力) 與 C(穩定度) 的最終排序。")
        
        if not v32_df.empty:
            raw_df = get_raw_top10(v32_df)
            if not raw_df.empty:
                st.dataframe(
                    raw_df[['代號','名稱','收盤','V32總分','穩定度','技術分','量能分']]
                    .style
                    .format(fmt_score)
                    .background_gradient(subset=['V32總分'], cmap='Reds')
                    .map(color_stability, subset=['穩定度']),
                    hide_index=True, 
                    use_container_width=True
                )
            else:
                st.info("無資料")
        else:
            st.warning("暫無資料")

    # === Tab 3: 庫存管理 (邏輯不變，僅更新分數欄位名稱) ===
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
                    sc = match.iloc[0]['V32總分'] # 更新欄位
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
