import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from github import Github 
import time
import requests
from fugle_marketdata import RestClient

# --- 設定頁面資訊 ---
st.set_page_config(page_title="V32 真實收益戰情室", layout="wide", page_icon="💰")

# --- 全域變數 ---
HOLDING_REPO = "gtty2003-ux/v32-data"
HOLDINGS_FILE = "holdings.csv"

# --- 樣式與工具函數 ---
def get_taiwan_time_str():
    return datetime.now(pytz.timezone('Asia/Taipei')).strftime("%Y-%m-%d %H:%M:%S")

def color_surplus(val):
    if not isinstance(val, (int, float)): return ''
    return 'color: #d32f2f; font-weight: bold;' if val > 0 else ('color: #388e3c; font-weight: bold;' if val < 0 else 'color: black')

# --- 核心資料讀寫 (GitHub) ---
def load_holdings():
    try:
        g = Github(st.secrets["general"]["GITHUB_TOKEN"])
        df = pd.read_csv(g.get_repo(HOLDING_REPO).get_contents(HOLDINGS_FILE).download_url)
        df['股票代號'] = df['股票代號'].astype(str).apply(lambda x: x.split('.')[0] if '.' in x else x)
        # 確保必要欄位存在
        for col in ['累積配息', '買入均價', '持有股數']:
            if col not in df.columns: df[col] = 0.0
        return df[['股票代號', '買入均價', '持有股數', '累積配息']]
    except: 
        return pd.DataFrame(columns=["股票代號", "買入均價", "持有股數", "累積配息"])

def save_holdings(df):
    try:
        g = Github(st.secrets["general"]["GITHUB_TOKEN"])
        repo = g.get_repo(HOLDING_REPO)
        contents = repo.get_contents(HOLDINGS_FILE)
        repo.update_file(contents.path, f"V32 Sync {get_taiwan_time_str()}", df.to_csv(index=False), contents.sha)
        st.toast("✅ 資料已同步至雲端", icon="☁️")
    except Exception as e:
        st.error(f"儲存失敗: {e}")

# --- 即時報價 (Fugle) ---
def get_realtime_quotes(code_list):
    realtime_data = {}
    try:
        client = RestClient(api_key=st.secrets["general"]["FUGLE_API_KEY"])
        for code in code_list:
            q = client.stock.intraday.quote(symbol=str(code).strip())
            p = q.get('closePrice') or q.get('lastPrice') or q.get('avgPrice')
            if p: realtime_data[str(code)] = float(p)
    except: pass
    return realtime_data

# --- 主程式 ---
def main():
    st.title("💰 V32 股票配息與真實損益追蹤")
    
    if 'inventory' not in st.session_state: st.session_state['inventory'] = load_holdings()
    if 'quotes' not in st.session_state: st.session_state['quotes'] = {}

    # --- 第一區：資產總覽 ---
    inv = st.session_state['inventory'].copy()
    if not inv.empty:
        quotes = st.session_state['quotes']
        res = []
        for _, r in inv.iterrows():
            code = r['股票代號']
            curr = quotes.get(code, r['買入均價'])
            total_cost = r['買入均價'] * r['持有股數']
            market_val = curr * r['持有股數']
            unrealized_pl = market_val - total_cost
            true_pl = unrealized_pl + r['累積配息']
            
            res.append({
                '代號': code, '持有股數': int(r['持有股數']), '買入均價': r['買入均價'],
                '即時價': curr, '累積配息': r['累積配息'], '真實總損益': true_pl,
                '真實報酬率%': (true_pl / total_cost * 100) if total_cost > 0 else 0
            })
        
        df_res = pd.DataFrame(res)
        c1, c2, c3 = st.columns(3)
        c1.metric("投入總成本", f"${df_res['買入均價'].mul(df_res['持有股數']).sum():,.0f}")
        c2.metric("累積總配息", f"${df_res['累積配息'].sum():,.0f}")
        c3.metric("總真實損益", f"${df_res['真實總損益'].sum():,.0f}", 
                  delta=f"{(df_res['真實總損益'].sum() / (df_res['買入均價'].mul(df_res['持有股數']).sum() or 1) * 100):.2f}%")

        st.dataframe(df_res.style.format({'買入均價':'{:.2f}', '即時價':'{:.2f}', '真實總損益':'{:+,.0f}', '真實報酬率%':'{:+.2f}%'})
                     .map(color_surplus, subset=['真實總損益', '真實報酬率%']), use_container_width=True, hide_index=True)

    st.divider()

    # --- 第二區：動作計算機 (這是你要的核心功能) ---
    st.subheader("🛠️ 庫存異動與領息登記")
    with st.expander("點擊展開：加碼買入 / 領取配息小工具", expanded=True):
        col_sel, col_act, col_p, col_q = st.columns([1, 1, 1, 1])
        
        # 建立現有代號清單，並加入「新增代號」選項
        exist_codes = inv['股票代號'].tolist() if not inv.empty else []
        target = col_sel.selectbox("選擇股票", options=["+ 新增標的"] + exist_codes)
        action = col_act.selectbox("執行動作", options=["加碼買入", "登記領息", "手動校正"])
        
        if target == "+ 新增標的":
            new_code = st.text_input("輸入新股票代號 (如: 00919)")
            current_target = new_code
        else:
            current_target = target

        if action == "加碼買入":
            price = col_p.number_input("買入單價", min_value=0.0, step=0.01)
            qty = col_q.number_input("買入股數", min_value=0, step=1000)
        elif action == "登記領息":
            div_amt = col_p.number_input("本次領息總額", min_value=0.0, step=1.0)
            st.caption("※ 系統會將此金額直接累加至該標的的『累積配息』欄位")
        else: # 手動校正
            price = col_p.number_input("校正均價", min_value=0.0, step=0.01)
            qty = col_q.number_input("校正總股數", min_value=0, step=1)

        if st.button("🔥 執行並更新狀態", type="primary"):
            temp_inv = st.session_state['inventory'].copy()
            
            # 取得該標的舊有數據
            match = temp_inv[temp_inv['股票代號'] == current_target]
            old_q = match['持有股數'].values[0] if not match.empty else 0
            old_p = match['買入均價'].values[0] if not match.empty else 0
            old_d = match['累積配息'].values[0] if not match.empty else 0
            
            if action == "加碼買入" and current_target:
                new_total_q = old_q + qty
                new_avg_p = ((old_p * old_q) + (price * qty)) / new_total_q if new_total_q > 0 else 0
                new_row = {'股票代號': current_target, '買入均價': round(new_avg_p, 2), '持有股數': new_total_q, '累積配息': old_d}
            elif action == "登記領息":
                new_row = {'股票代號': current_target, '買入均價': old_p, '持有股數': old_q, '累積配息': old_d + div_amt}
            else: # 手動校正
                new_row = {'股票代號': current_target, '買入均價': price, '持有股數': qty, '累積配息': old_d}

            # 更新 DataFrame
            temp_inv = temp_inv[temp_inv['股票代號'] != current_target]
            temp_inv = pd.concat([temp_inv, pd.DataFrame([new_row])], ignore_index=True)
            
            st.session_state['inventory'] = temp_inv
            save_holdings(temp_inv)
            st.success(f"✅ {current_target} 更新成功！")
            time.sleep(1)
            st.rerun()

    # --- 第三區：功能按鈕 ---
    c_u, c_s = st.columns([1, 4])
    if c_u.button("🔄 更新即時股價"):
        codes = st.session_state['inventory']['股票代號'].tolist()
        st.session_state['quotes'] = get_realtime_quotes(codes)
        st.rerun()

if __name__ == "__main__":
    main()
