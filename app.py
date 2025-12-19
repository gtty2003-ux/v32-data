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
    page_title="V32 戰情室 (Pro Calculation)",
    layout="wide",
    page_icon="🧠"
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

# --- 技術指標計算函數 ---
def calculate_indicators(hist):
    """輸入歷史資料 DataFrame，回傳計算好的技術與量能分"""
    if len(hist) < 60: return 0, 0, "Data Insufficient" # 資料不足

    # 1. 計算指標
    close = hist['Close']
    vol = hist['Volume']
    
    # 均線 (MA)
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

    # MACD (12, 26, 9)
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_now = macd.iloc[-1]
    signal_now = signal.iloc[-1]

    # 均量 (Vol MA)
    vol_ma5 = vol.rolling(5).mean().iloc[-1]
    vol_ma20 = vol.rolling(20).mean().iloc[-1]

    # --- 評分邏輯 (V32 Formula) ---
    
    # A. 技術分 (滿分 100)
    tech_score = 60 # 基礎分
    
    # 趨勢 (Trend)
    if close.iloc[-1] > ma20 and ma20 > ma20_prev: tech_score += 10 # 站上月線且月線上彎
    if ma5 > ma20 and ma20 > ma60: tech_score += 10 # 多頭排列
    
    # 動能 (Momentum)
    if rsi_now > 50: tech_score += 5
    if rsi_now > 70: tech_score += 5 # 強勢區
    if macd_now > signal_now: tech_score += 5 # MACD 金叉狀態
    
    # 型態 (Structure) - 突破 20 日高點
    high_20 = hist['High'].rolling(20).max().iloc[-2] # 昨日為止的20日高
    if close.iloc[-1] > high_20: tech_score += 15 # 突破

    # B. 量能分 (滿分 100)
    vol_score = 60 # 基礎分
    
    current_vol = vol.iloc[-1]
    if current_vol > vol_ma20: vol_score += 15 # 大於月均量
    if current_vol > vol_ma5: vol_score += 10 # 大於週均量
    
    # 價量配合 (收紅且量增)
    is_red = close.iloc[-1] > hist['Open'].iloc[-1]
    vol_up = current_vol > vol.iloc[-2]
    if is_red and vol_up: vol_score += 15

    # 上限防呆
    tech_score = min(100, tech_score)
    vol_score = min(100, vol_score)
    
    # 趨勢標記
    trend_status = "Rising" if (close.iloc[-1] > ma5 and ma5 > ma20) else "Consolidating"
    
    return tech_score, vol_score, trend_status

# --- 批次運算引擎 ---
@st.cache_data(ttl=3600) # 快取 1 小時，避免重複運算
def run_v32_engine(ticker_list):
    """
    針對清單中的股票，使用 yfinance 抓取歷史資料並重新計算分數
    """
    results = []
    
    # 為了顯示進度條
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(ticker_list)
    
    for i, row in enumerate(ticker_list):
        symbol = str(row['代號'])
        name = str(row.get('名稱', ''))
        
        status_text.text(f"正在分析技術指標: {symbol} {name} ({i+1}/{total})...")
        progress_bar.progress((i + 1) / total)
        
        try:
            # 抓取 3 個月資料以計算 MA60
            stock = yf.Ticker(f"{symbol}.TW")
            hist = stock.history(period="3mo")
            
            if not hist.empty:
                t_score, v_score, trend = calculate_indicators(hist)
                
                # V32 總分公式
                total_score = (t_score * 0.7) + (v_score * 0.3)
                
                results.append({
                    '代號': symbol,
                    '名稱': name,
                    '收盤': hist['Close'].iloc[-1],
                    '成交量': hist['Volume'].iloc[-1],
                    '技術分': t_score,
                    '量能分': v_score,
                    '總分': total_score,
                    '趨勢': trend
                })
            else:
                # 抓不到資料，保留原始資訊但分數歸零
                results.append(row)
        except:
            pass
            
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

# --- 資料讀取與前處理 ---
def load_and_process_data():
    url = f"https://raw.githubusercontent.com/{REPO_KEY}/main/v32_recommend.csv"
    try:
        # 1. 讀取 CSV (只為了拿到股票代號清單)
        df = pd.read_csv(url)
        
        # 欄位標準化
        code_col = next((c for c in ['代碼', '代號', 'Code', 'Symbol'] if c in df.columns), None)
        if code_col:
            df[code_col] = df[code_col].astype(str).str.strip()
            df = df.rename(columns={code_col: '代號'})
        
        # 2. 🔥 啟動 V32 運算引擎 (這是最耗時的步驟)
        # 為了避免每次重整都跑，Streamlit Cache 會幫忙存下來
        processed_df = run_v32_engine(df[['代號', '名稱']].to_dict('records'))
        
        return processed_df, None
    except Exception as e:
        return pd.DataFrame(), str(e)

# --- GitHub 存取 (庫存用) ---
def load_data_from_github():
    try:
        token = st.secrets["general"]["GITHUB_TOKEN"]
        g = Github(token)
        repo = g.get_repo(REPO_KEY)
        contents = repo.get_contents(FILE_PATH)
        df = pd.read_csv(contents.download_url)
        df['股票代號'] = df['股票代號'].astype(str).str.strip()
        required_cols = ["股票代號", "買入均價", "持有股數"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0 if col != "股票代號" else ""
        return df[required_cols]
    except:
        return pd.DataFrame(columns=["股票代號", "買入均價", "持有股數"])

def save_data_to_github(df):
    try:
        token = st.secrets["general"]["GITHUB_TOKEN"]
        g = Github(token)
        repo = g.get_repo(REPO_KEY)
        csv_content = df.to_csv(index=False)
        try:
            contents = repo.get_contents(FILE_PATH)
            repo.update_file(contents.path, f"Update holdings {get_taiwan_time()}", csv_content, contents.sha)
            st.success(f"✅ 資料已成功儲存！")
        except:
            repo.create_file(FILE_PATH, "Create holdings.csv", csv_content)
            st.success("✅ 已建立新庫存檔並儲存！")
    except Exception as e:
        st.error(f"❌ 儲存失敗: {e}")

# --- 分層選股策略 ---
def get_stratified_selection(df):
    if df.empty: return df, []

    # 1. 硬指標過濾 (Tech>=88, Vol>=82, Trend=Rising, Total 86-92)
    filtered = df[
        (df['技術分'] >= 88) & 
        (df['量能分'] >= 82) & 
        (df['趨勢'] == 'Rising') &
        (df['總分'] >= 86) &
        (df['總分'] <= 92)
    ].copy()
    
    if filtered.empty: return pd.DataFrame(), ["無符合硬指標 (技≥88/量≥82/趨勢Up) 的標的"]

    # 2. 分層選取 Top 5
    bucket_a = filtered[(filtered['總分'] >= 90) & (filtered['總分'] <= 92)].sort_values(by='總分', ascending=False).head(5)
    bucket_b = filtered[(filtered['總分'] >= 88) & (filtered['總分'] < 90)].sort_values(by='總分', ascending=False).head(5)
    bucket_c = filtered[(filtered['總分'] >= 86) & (filtered['總分'] < 88)].sort_values(by='總分', ascending=False).head(5)
    
    final_selection = pd.concat([bucket_a, bucket_b, bucket_c])
    
    stats = [
        f"90-92分: {len(bucket_a)} 檔",
        f"88-90分: {len(bucket_b)} 檔",
        f"86-88分: {len(bucket_c)} 檔"
    ]
    
    return final_selection, stats

# --- 主程式 ---
def main():
    st.title("🧠 V32 戰情室 (Pro Calculation)")
    st.caption(f"最後更新: {get_taiwan_time()}")
    
    # 載入並計算
    v32_df, err = load_and_process_data()

    tab_scan, tab_holdings = st.tabs(["🚀 精選 Top 15", "💼 庫存管理"])

    # === Tab 1: 掃描 ===
    with tab_scan:
        if err: st.error(err)
        
        if not v32_df.empty:
            # 分類
            def get_cat(row):
                c = str(row['代號'])
                n = str(row.get('名稱', ''))
                if '債' in n or 'KY' in n or c.startswith('00') or c.startswith('91') or c[-1].isalpha() or (len(c)>4 and c.isdigit()):
                    return 'Special'
                return 'General'
            v32_df['cat'] = v32_df.apply(get_cat, axis=1)
            
            # 分層篩選
            final_gen, stats_g = get_stratified_selection(v32_df[v32_df['cat']=='General'])
            final_spec, stats_s = get_stratified_selection(v32_df[v32_df['cat']=='Special'])
            
            t1, t2 = st.tabs(["🏢 一般個股", "📊 特殊/ETF"])
            
            # 顯示格式
            fmt = {'收盤':'{:.2f}', '技術分':'{:.0f}', '量能分':'{:.0f}', '總分':'{:.1f}'}
            
            with t1:
                st.info(f"🎯 分層結果：{' | '.join(stats_g)}")
                if not final_gen.empty:
                    st.dataframe(final_gen[['代號','名稱','收盤','技術分','量能分','總分','趨勢']].style.format(fmt), use_container_width=True, hide_index=True)
                else:
                    st.warning("無符合條件的一般個股。")

            with t2:
                st.info(f"🎯 分層結果：{' | '.join(stats_s)}")
                if not final_spec.empty:
                    st.dataframe(final_spec[['代號','名稱','收盤','技術分','量能分','總分','趨勢']].style.format(fmt), use_container_width=True, hide_index=True)
                else:
                    st.warning("無符合條件的特殊/ETF。")
        else:
            st.warning("無法取得股票清單。")

    # === Tab 2: 庫存 ===
    with tab_holdings:
        st.subheader("📝 庫存編輯器")
        if 'editor_data' not in st.session_state:
            st.session_state['editor_data'] = load_data_from_github()

        edited_df = st.data_editor(
            st.session_state['editor_data'],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "股票代號": st.column_config.TextColumn("代號", required=True),
                "買入均價": st.column_config.NumberColumn("均價", min_value=0, format="%.2f"),
                "持有股數": st.column_config.NumberColumn("股數", min_value=0, step=1000),
            },
            key="holdings_editor" 
        )
        if st.button("💾 儲存變更"):
            save_data_to_github(edited_df)
            st.rerun()
            
        # 簡單損益計算 (直接利用上面算好的 v32_df 現價)
        st.divider()
        if not edited_df.empty and not v32_df.empty:
            display_data = []
            for i, row in edited_df.iterrows():
                if not row['股票代號']: continue
                code = str(row['股票代號'])
                cost = float(row['買入均價'] or 0)
                qty = float(row['持有股數'] or 0)
                
                # 從已運算的 v32_df 找現價 (最快)
                match = v32_df[v32_df['代號'] == code]
                if not match.empty:
                    curr = match.iloc[0]['收盤']
                    name = match.iloc[0]['名稱']
                    score = match.iloc[0]['總分']
                else:
                    # 榜外股要另外抓
                    try:
                        t = yf.Ticker(f"{code}.TW")
                        hist = t.history(period='1d')
                        curr = hist['Close'].iloc[-1] if not hist.empty else 0
                        name = code
                        score = 0
                    except:
                        curr=0; name=code; score=0

                val = curr * qty
                cost_total = cost * qty
                pl = val - cost_total
                roi = (pl/cost_total*100) if cost_total>0 else 0
                
                display_data.append({
                    "代號": code, "名稱": name, "現價": curr, "成本": cost, 
                    "股數": qty, "損益": pl, "報酬率%": roi, "V32分": f"{score:.1f}" if score>0 else "榜外"
                })
            
            if display_data:
                res = pd.DataFrame(display_data)
                st.dataframe(res.style.map(color_surplus, subset=['損益','報酬率%']).format({'現價':'{:.2f}', '損益':'{:+,.0f}', '報酬率%':'{:+.2f}%'}), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
