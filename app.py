import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from github import Github 
import time
import requests
import io

# --- 設定頁面資訊 ---
st.set_page_config(page_title="V32 真實收益戰情室", layout="wide", page_icon="💰")

DATA_REPO = "gtty2003-ux/v32-auto-updater" 
DATA_FILE = "v32_dataset.csv"
HOLDING_REPO = "gtty2003-ux/v32-data"
HOLDINGS_FILE = "holdings.csv"

st.markdown("""
    <style>
    div[data-testid="stMetricValue"] {font-size: 24px; font-weight: bold;}
    .stButton>button {width: 100%; border-radius: 5px; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

def color_surplus(val):
    if not isinstance(val, (int, float)): return ''
    return 'color: #d32f2f; font-weight: bold;' if val > 0 else ('color: #388e3c; font-weight: bold;' if val < 0 else 'color: black')

@st.cache_data(ttl=600)
def load_v32_crawler_data():
    try:
        token = st.secrets["general"]["GITHUB_TOKEN"]
        url = f"https://api.github.com/repos/{DATA_REPO}/contents/{DATA_FILE}"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3.raw"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text), dtype={'Code': str})
            df['Code'] = df['Code'].str.split('.').str[0].str.zfill(4)
            return df[['Code', 'ClosingPrice']].set_index('Code')['ClosingPrice'].to_dict()
        return {}
    except:
        return {}

def load_holdings():
    try:
        g = Github(st.secrets["general"]["GITHUB_TOKEN"])
        df = pd.read_csv(g.get_repo(HOLDING_REPO).get_contents(HOLDINGS_FILE).download_url, dtype={'股票代號': str})
        df['股票代號'] = df['股票代號'].str.strip().str.zfill(4)
        if '累積配息' not in df.columns: df['累積配息'] = 0.0
        df['累積配息'] = pd.to_numeric(df['累積配息'], errors='coerce').fillna(0.0)
        df['買入均價'] = pd.to_numeric(df['買入均價'], errors='coerce').fillna(0.0)
        df['持有股數'] = pd.to_numeric(df['持有股數'], errors='coerce').fillna(0.0)
        return df[['股票代號', '買入均價', '持有股數', '累積配息']]
    except: 
        return pd.DataFrame(columns=["股票代號", "買入均價", "持有股數", "累積配息"])

def save_holdings(df):
    try:
        g = Github(st.secrets["general"]["GITHUB_TOKEN"])
        repo = g.get_repo(HOLDING_REPO)
        contents = repo.get_contents(HOLDINGS_FILE)
        df['股票代號'] = df['股票代號'].astype(str).str.zfill(4)
        repo.update_file(contents.path, f"Sync {datetime.now().strftime('%H:%M:%S')}", df.to_csv(index=False), contents.sha)
        st.toast("✅ 資料已同步至 GitHub", icon="☁️")
    except:
        st.error("儲存失敗，請檢查網路或 Token")

def main():
    st.title("💰 V32 真實收益儀表板")
    
    # 【修復核心】：只在初次載入或手動重整時才去 GitHub 抓資料
    if 'inventory' not in st.session_state:
        st.session_state['inventory'] = load_holdings()
        
    crawler_prices = load_v32_crawler_data()
    inv = st.session_state['inventory'].copy()

    if not inv.empty:
        res = []
        for _, r in inv.iterrows():
            code = r['股票代號']
            curr = crawler_prices.get(code)
            if curr is None: curr = crawler_prices.get(str(int(code))) if code.isdigit() else None
            
            price_display = curr if curr is not None else r['買入均價']
            status_symbol = "✅" if curr is not None else "⚠️"
            
            total_cost = r['買入均價'] * r['持有股數']
            market_val = price_display * r['持有股數']
            unrealized_pl = market_val - total_cost
            true_pl = unrealized_pl + r['累積配息']
            
            res.append({
                '代號': code, '狀態': status_symbol, '持有股數': int(r['持有股數']), 
                '買入均價': r['買入均價'], '最新市價': price_display, 
                '累積配息': r['累積配息'], '真實總損益': true_pl,
                '真實報酬率%': (true_pl / total_cost * 100) if total_cost > 0 else 0
            })
        
        df_res = pd.DataFrame(res)
        
        # --- 選擇一：四聯排 ---
        c1, c2, c3, c4 = st.columns(4)
        cost_sum = (df_res['買入均價'] * df_res['持有股數']).sum()
        market_sum = (df_res['最新市價'] * df_res['持有股數']).sum() # 新增總市值計算
        pl_sum = df_res['真實總損益'].sum()
        
        c1.metric("總投入成本", f"${cost_sum:,.0f}")
        c2.metric("目前總市值", f"${market_sum:,.0f}", delta=f"帳面價差 {(market_sum - cost_sum):+,.0f} 元")
        c3.metric("累積總配息", f"${df_res['累積配息'].sum():,.0f}")
        c4.metric("總體真實報酬", f"${pl_sum:,.0f}", delta=f"{(pl_sum/cost_sum*100):.2f}%" if cost_sum > 0 else "0%")

        st.dataframe(
            df_res.style.format({
                '買入均價':'{:.2f}', '最新市價':'{:.2f}', 
                '累積配息':'{:,.0f}', '真實總損益':'{:+,.0f}', '真實報酬率%':'{:+.2f}%'
            }).map(color_surplus, subset=['真實總損益', '真實報酬率%']),
            column_config={"代號": st.column_config.TextColumn("代號")},
            use_container_width=True, hide_index=True
        )
    
    st.divider()

    st.subheader("🛠️ 數據更新工具")
    
    # 恢復獨立頁籤設計，拔除表單束縛
    tab_dividend, tab_trade = st.tabs(["💰 登記配息", "🛒 庫存異動 (加碼買進 / 校正)"])
    exist_codes = inv['股票代號'].tolist() if not inv.empty else []

    # === 分頁 1：專屬領息區 ===
    with tab_dividend:
        c_sel_div, c_amt_div, c_btn_div = st.columns([2, 2, 1])
        div_target = c_sel_div.selectbox("選擇配息股票", options=exist_codes if exist_codes else ["無庫存"])
        div_amt = c_amt_div.number_input("本次領息總額 ($)", min_value=0.0, step=1.0)
        
        c_btn_div.write("") 
        c_btn_div.write("")
        if c_btn_div.button("💾 儲存配息", type="primary", use_container_width=True):
            if div_target == "無庫存" or not div_target:
                st.error("請先新增股票標的！")
            else:
                new_inv = inv.copy()
                match = new_inv[new_inv['股票代號'] == div_target]
                old_p = match['買入均價'].values[0] if not match.empty else 0.0
                old_q = match['持有股數'].values[0] if not match.empty else 0.0
                old_d = match['累積配息'].values[0] if not match.empty else 0.0
                
                row = {'股票代號': div_target, '買入均價': old_p, '持有股數': old_q, '累積配息': old_d + div_amt}
                new_inv = new_inv[new_inv['股票代號'] != div_target]
                new_inv = pd.concat([new_inv, pd.DataFrame([row])], ignore_index=True)
                
                st.session_state['inventory'] = new_inv
                save_holdings(new_inv)
                st.rerun()

    # === 分頁 2：買賣與校正區 ===
    with tab_trade:
        c_sel, c_act, c_p, c_q = st.columns([1.5, 1.5, 1.5, 1.5])
        target = c_sel.selectbox("選擇或新增股票", options=["+ 新增標的"] + exist_codes)
        action = c_act.selectbox("動作", options=["加碼買入", "手動校正"])
        
        if target == "+ 新增標的":
            t_code = st.text_input("輸入代號 (如 00919)").strip().zfill(4)
        else:
            t_code = target

        p_val = c_p.number_input("單價", min_value=0.0, step=0.01)
        q_val = c_q.number_input("股數", min_value=0, step=100)

        st.write("")
        if st.button("💾 確認庫存異動", type="primary"):
            if not t_code or t_code == "0000":
                st.error("請輸入正確代號")
            else:
                new_inv = inv.copy()
                match = new_inv[new_inv['股票代號'] == t_code]
                
                old_p = match['買入均價'].values[0] if not match.empty else 0.0
                old_q = match['持有股數'].values[0] if not match.empty else 0.0
                old_d = match['累積配息'].values[0] if not match.empty else 0.0

                if action == "加碼買入":
                    new_q = old_q + q_val
                    new_p = ((old_p * old_q) + (p_val * q_val)) / new_q if new_q > 0 else 0
                    row = {'股票代號': t_code, '買入均價': round(new_p, 2), '持有股數': new_q, '累積配息': old_d}
                else: 
                    row = {'股票代號': t_code, '買入均價': p_val, '持有股數': q_val, '累積配息': old_d}

                new_inv = new_inv[new_inv['股票代號'] != t_code]
                new_inv = pd.concat([new_inv, pd.DataFrame([row])], ignore_index=True)
                
                st.session_state['inventory'] = new_inv
                save_holdings(new_inv)
                st.rerun()

    st.divider()
    if st.button("🔄 強制重新載入遠端資料", use_container_width=True):
        load_v32_crawler_data.clear()
        st.session_state['inventory'] = load_holdings()
        st.rerun()

if __name__ == "__main__":
    main()
