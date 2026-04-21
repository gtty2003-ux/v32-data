import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from github import Github 
import time
import requests
# --- 富果 API 套件 ---
from fugle_marketdata import RestClient

# --- 設定頁面資訊 ---
st.set_page_config(
    page_title="真實資產戰情室",
    layout="wide",
    page_icon="💰"
)

# --- 全域變數 ---
HOLDING_REPO = "gtty2003-ux/v32-data"
HOLDINGS_FILE = "holdings.csv"

# --- 樣式設定 ---
st.markdown("""
    <style>
    div[data-testid="stMetricValue"] {font-size: 24px; font-weight: bold;}
    .stButton>button {width: 100%; border-radius: 5px; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

# --- 工具函數 ---
def get_taiwan_time_str(timestamp=None):
    tz = pytz.timezone('Asia/Taipei')
    dt = datetime.fromtimestamp(timestamp, pytz.utc).astimezone(tz) if timestamp else datetime.now(tz)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def get_taiwan_time_iso():
    return datetime.now(pytz.timezone('Asia/Taipei')).strftime("%Y-%m-%d %H:%M:%S")

def color_surplus(val):
    if not isinstance(val, (int, float)): return ''
    return 'color: #d32f2f; font-weight: bold;' if val > 0 else ('color: #388e3c; font-weight: bold;' if val < 0 else 'color: black')

# --- 即時報價模組 (Fugle API) ---
def get_realtime_quotes_robust(code_list):
    realtime_data = {}
    try:
        api_key = st.secrets["general"]["FUGLE_API_KEY"]
        client = RestClient(api_key=api_key)
    except Exception as e:
        st.error(f"Fugle 連線失敗或未設定 API Key: {e}")
        return {}
    
    progress_bar = st.progress(0, text="🚀 同步即時股價中 (Fugle API)...")
    total = len(code_list)
    
    for idx, code in enumerate(code_list):
        clean_code = str(code).strip().split('.')[0]
        try:
            q = client.stock.intraday.quote(symbol=clean_code)
            price = q.get('closePrice') or q.get('lastPrice') or q.get('avgPrice')
            if price:
                realtime_data[clean_code] = {'即時價': float(price)}
        except:
            pass
        progress_bar.progress((idx + 1) / total)
    
    progress_bar.empty()
    return realtime_data

# --- 庫存與配息管理 ---
def load_holdings():
    try:
        g = Github(st.secrets["general"]["GITHUB_TOKEN"])
        df = pd.read_csv(g.get_repo(HOLDING_REPO).get_contents(HOLDINGS_FILE).download_url)
        df['股票代號'] = df['股票代號'].astype(str).apply(lambda x: x.split('.')[0] if '.' in x else x)
        
        # 相容舊版資料：如果沒有配息欄位，自動補上 0
        if '累積配息' not in df.columns:
            df['累積配息'] = 0.0
            
        return df[['股票代號', '買入均價', '持有股數', '累積配息']]
    except: 
        return pd.DataFrame(columns=["股票代號", "買入均價", "持有股數", "累積配息"])

def save_holdings(df):
    try:
        g = Github(st.secrets["general"]["GITHUB_TOKEN"])
        repo = g.get_repo(HOLDING_REPO)
        csv_content = df.to_csv(index=False)
        contents = repo.get_contents(HOLDINGS_FILE)
        repo.update_file(contents.path, f"Update Data {get_taiwan_time_iso()}", csv_content, contents.sha)
    except Exception as e:
        st.error(f"儲存失敗: {e}")

# --- 主程式 ---
def main():
    st.title("💰 股票配息與真實損益追蹤系統")
    
    if 'inventory' not in st.session_state: st.session_state['inventory'] = load_holdings()
    if 'realtime_quotes' not in st.session_state: st.session_state['realtime_quotes'] = {}
    if 'last_update_time' not in st.session_state: st.session_state['last_update_time'] = 0

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        now = time.time()
        time_diff = now - st.session_state.get('last_update_time', 0)
        btn_label = "🔄 更新即時股價"
        btn_disabled = time_diff < 60
        if btn_disabled: btn_label = f"⏳ 冷卻中 ({int(60 - time_diff)}s)"
        
        if st.button(btn_label, disabled=btn_disabled, type="primary"):
            if not st.session_state['inventory'].empty:
                codes = st.session_state['inventory']['股票代號'].tolist()
                fresh_quotes = get_realtime_quotes_robust(codes)
                st.session_state['realtime_quotes'].update(fresh_quotes)
                st.session_state['last_update_time'] = time.time()
                st.rerun()

    with col_info:
        if st.session_state.get('last_update_time', 0) > 0:
            st.caption(f"🕒 股價最後更新: {get_taiwan_time_str(st.session_state['last_update_time'])}")

    st.divider()

    # --- 核心計算區塊 ---
    inv_df = st.session_state['inventory'].copy()
    if not inv_df.empty:
        saved_quotes = st.session_state.get('realtime_quotes', {})
        res = []
        for _, r in inv_df.iterrows():
            code = str(r['股票代號'])
            curr = saved_quotes.get(code, {}).get('即時價', r['買入均價']) # 抓不到就用均價代替
            
            cost = r['買入均價']
            qty = r['持有股數']
            dividend = r['累積配息']
            
            total_cost = cost * qty
            market_value = curr * qty
            unrealized_pl = market_value - total_cost # 帳面價差
            true_total_pl = unrealized_pl + dividend  # 真實損益 = 價差 + 配息
            true_roi = (true_total_pl / total_cost * 100) if total_cost > 0 else 0
            
            res.append({
                '代號': code,
                '持有股數': int(qty),
                '買入均價': cost,
                '即時價': curr,
                '總成本': total_cost,
                '目前市值': market_value,
                '帳面損益 (價差)': unrealized_pl,
                '累積配息': dividend,
                '真實總損益': true_total_pl,
                '真實報酬率%': true_roi
            })
            
        df_res = pd.DataFrame(res)
        
        # --- 戰情儀表板 ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("投入總成本", f"${df_res['總成本'].sum():,.0f}")
        c2.metric("目前總市值", f"${df_res['目前市值'].sum():,.0f}")
        c3.metric("已領總配息", f"${df_res['累積配息'].sum():,.0f}", delta="現金入袋")
        c4.metric("總體真實損益", f"${df_res['真實總損益'].sum():,.0f}", delta=f"總報酬 {(df_res['真實總損益'].sum() / df_res['總成本'].sum() * 100):.2f}%")
        
        st.markdown("### 📊 詳細資產狀況")
        st.dataframe(
            df_res[['代號', '持有股數', '買入均價', '即時價', '帳面損益 (價差)', '累積配息', '真實總損益', '真實報酬率%']].style
            .format({'買入均價':'{:.2f}', '即時價':'{:.2f}', '帳面損益 (價差)':'{:+,.0f}', '累積配息':'{:+,.0f}', '真實總損益':'{:+,.0f}', '真實報酬率%':'{:+.2f}%'})
            .map(color_surplus, subset=['帳面損益 (價差)', '真實總損益', '真實報酬率%']), 
            use_container_width=True, hide_index=True
        )
    else:
        st.info("目前無庫存資料，請在下方新增。")

    st.divider()
    
    # --- 資料編輯區塊 ---
    st.markdown("### 📝 更新庫存與配息 (直接在表格內修改後儲存)")
    edited_df = st.data_editor(inv_df, num_rows="dynamic", use_container_width=True, key="inventory_editor", hide_index=True)
    
    if st.button("💾 儲存變更至 GitHub", type="primary"):
        st.session_state['inventory'] = edited_df
        save_holdings(edited_df)
        st.success("✅ 資料已成功同步至 GitHub！")
        time.sleep(1)
        st.rerun()

if __name__ == "__main__":
    main()
