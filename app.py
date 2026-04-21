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

# --- 全域變數 ---
DATA_REPO = "gtty2003-ux/v32-auto-updater" 
DATA_FILE = "v32_dataset.csv"
HOLDING_REPO = "gtty2003-ux/v32-data"
HOLDINGS_FILE = "holdings.csv"

# --- 樣式設定 ---
st.markdown("""
    <style>
    div[data-testid="stMetricValue"] {font-size: 24px; font-weight: bold;}
    .stButton>button {width: 100%; border-radius: 5px; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

def color_surplus(val):
    if not isinstance(val, (int, float)): return ''
    return 'color: #d32f2f; font-weight: bold;' if val > 0 else ('color: #388e3c; font-weight: bold;' if val < 0 else 'color: black')

# --- 核心資料讀取 (GitHub) ---
@st.cache_data(ttl=600)
def load_v32_crawler_data():
    """讀取 V32 爬蟲系統，並建立強健的價格字典"""
    try:
        token = st.secrets["general"]["GITHUB_TOKEN"]
        url = f"https://api.github.com/repos/{DATA_REPO}/contents/{DATA_FILE}"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3.raw"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            # 讀取時就強制 Code 為字串
            df = pd.read_csv(io.StringIO(response.text), dtype={'Code': str})
            
            # 處理代號：補0並去掉 .TW 或 .TWO 尾綴
            df['Code'] = df['Code'].str.split('.').str[0].str.zfill(4)
            return df[['Code', 'ClosingPrice']].set_index('Code')['ClosingPrice'].to_dict()
        return {}
    except:
        return {}

def load_holdings():
    """讀取庫存，強制維持代號格式"""
    try:
        g = Github(st.secrets["general"]["GITHUB_TOKEN"])
        # 使用 dtype 參數強制讀取為字串，防止 0056 變成 56
        df = pd.read_csv(g.get_repo(HOLDING_REPO).get_contents(HOLDINGS_FILE).download_url, dtype={'股票代號': str})
        
        # 二次補強：確保補滿 4 碼
        df['股票代號'] = df['股票代號'].str.strip().str.zfill(4)
        
        # 數值防呆
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
        # 存檔時確保格式
        df['股票代號'] = df['股票代號'].astype(str).str.zfill(4)
        repo.update_file(contents.path, f"Sync {datetime.now().strftime('%H:%M:%S')}", df.to_csv(index=False), contents.sha)
        st.toast("✅ 資料已同步", icon="☁️")
    except:
        st.error("儲存失敗")

# --- 主程式 ---
def main():
    st.title("💰 V32 真實收益儀表板")
    
    # 每次跑程式都重刷資料庫與庫存
    crawler_prices = load_v32_crawler_data()
    inv = load_holdings()

    if not inv.empty:
        res = []
        for _, r in inv.iterrows():
            code = r['股票代號']
            # 查找價格字典
            curr = crawler_prices.get(code)
            
            # 如果爬蟲沒抓到，嘗試在字典裡找去 0 版 (預防爬蟲資料沒補0)
            if curr is None:
                curr = crawler_prices.get(str(int(code))) if code.isdigit() else None
            
            # 最終還是沒抓到，才用均價 (並加上警告符號)
            price_display = curr if curr is not None else r['買入均價']
            status_symbol = "✅" if curr is not None else "⚠️"
            
            total_cost = r['買入均價'] * r['持有股數']
            market_val = price_display * r['持有股數']
            unrealized_pl = market_val - total_cost
            true_pl = unrealized_pl + r['累積配息']
            
            res.append({
                '代號': code, 
                '狀態': status_symbol,
                '持有股數': int(r['持有股數']), 
                '買入均價': r['買入均價'],
                '最新市價': price_display, 
                '累積配息': r['累積配息'], 
                '真實總損益': true_pl,
                '真實報酬率%': (true_pl / total_cost * 100) if total_cost > 0 else 0
            })
        
        df_res = pd.DataFrame(res)
        
        # 總計欄位
        c1, c2, c3 = st.columns(3)
        cost_sum = (df_res['買入均價'] * df_res['持有股數']).sum()
        pl_sum = df_res['真實總損益'].sum()
        c1.metric("總投入成本", f"${cost_sum:,.0f}")
        c2.metric("累積總配息", f"${df_res['累積配息'].sum():,.0f}")
        c3.metric("總體真實報酬", f"${pl_sum:,.0f}", delta=f"{(pl_sum/cost_sum*100):.2f}%" if cost_sum > 0 else "0%")

        # 表格顯示 (關鍵：使用 column_config 鎖定文字型態)
        st.dataframe(
            df_res.style.format({
                '買入均價':'{:.2f}', '最新市價':'{:.2f}', 
                '累積配息':'{:,.0f}', '真實總損益':'{:+,.0f}', '真實報酬率%':'{:+.2f}%'
            }).map(color_surplus, subset=['真實總損益', '真實報酬率%']),
            column_config={
                "代號": st.column_config.TextColumn("代號"), # 強制為文字，防止去0
            },
            use_container_width=True, 
            hide_index=True
        )
        if "⚠️" in df_res['狀態'].values:
            st.caption("⚠️ 表示爬蟲資料庫中找不到該代號的最新價格，暫時以均價計算。")
    
    st.divider()

    # --- 加碼/領息小工具 ---
    st.subheader("🛠️ 數據更新工具")
    with st.expander("執行買入或領息登記", expanded=True):
        c_sel, c_act, c_p, c_q = st.columns(4)
        target = c_sel.selectbox("選擇股票", options=["+ 新增標的"] + (inv['股票代號'].tolist() if not inv.empty else []))
        action = c_act.selectbox("動作", options=["加碼買入", "登記領息", "手動校正"])
        
        if target == "+ 新增標的":
            t_code = st.text_input("輸入代號").strip().zfill(4)
        else:
            t_code = target

        p_val = c_p.number_input("價格 / 總配息金額", min_value=0.0, step=0.01)
        q_val = c_q.number_input("股數變動", min_value=0, step=100)

        if st.button("💾 確認更新", type="primary"):
            if not t_code or t_code == "0000":
                st.error("請輸入正確代號")
                return

            new_inv = inv.copy()
            match = new_inv[new_inv['股票代號'] == t_code]
            
            old_p = match['買入均價'].values[0] if not match.empty else 0.0
            old_q = match['持有股數'].values[0] if not match.empty else 0.0
            old_d = match['累積配息'].values[0] if not match.empty else 0.0

            if action == "加碼買入":
                new_q = old_q + q_val
                new_p = ((old_p * old_q) + (p_val * q_val)) / new_q if new_q > 0 else 0
                row = {'股票代號': t_code, '買入均價': round(new_p, 2), '持有股數': new_q, '累積配息': old_d}
            elif action == "登記領息":
                row = {'股票代號': t_code, '買入均價': old_p, '持有股數': old_q, '累積配息': old_d + p_val}
            else:
                row = {'股票代號': t_code, '買入均價': p_val, '持有股數': q_val, '累積配息': old_d}

            new_inv = new_inv[new_inv['股票代號'] != t_code]
            new_inv = pd.concat([new_inv, pd.DataFrame([row])], ignore_index=True)
            save_holdings(new_inv)
            st.rerun()

if __name__ == "__main__":
    main()
