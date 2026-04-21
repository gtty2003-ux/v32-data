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

# --- 全域變數 (對接你現有的 Repo) ---
DATA_REPO = "gtty2003-ux/v32-auto-updater" 
DATA_FILE = "v32_dataset.csv"
HOLDING_REPO = "gtty2003-ux/v32-data"
HOLDINGS_FILE = "holdings.csv"

# --- 樣式與工具函數 ---
def get_taiwan_time_str():
    return datetime.now(pytz.timezone('Asia/Taipei')).strftime("%Y-%m-%d %H:%M:%S")

def color_surplus(val):
    if not isinstance(val, (int, float)): return ''
    return 'color: #d32f2f; font-weight: bold;' if val > 0 else ('color: #388e3c; font-weight: bold;' if val < 0 else 'color: black')

# --- 核心資料讀寫 (GitHub) ---
@st.cache_data(ttl=1800) # 每 30 分鐘快取一次爬蟲數據
def load_v32_crawler_data():
    """讀取 V32 爬蟲系統的收盤價數據"""
    try:
        token = st.secrets["general"]["GITHUB_TOKEN"]
        url = f"https://api.github.com/repos/{DATA_REPO}/contents/{DATA_FILE}"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3.raw"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            # 關鍵修正 1：強制代號為字串，並補齊 4 碼 (讓 56 變回 0056)
            df['Code'] = df['Code'].astype(str).str.strip().str.zfill(4)
            return df[['Code', 'ClosingPrice']].set_index('Code')['ClosingPrice'].to_dict()
        return {}
    except Exception as e:
        return {}

def load_holdings():
    """讀取你的庫存資料，並進行防呆處理"""
    try:
        g = Github(st.secrets["general"]["GITHUB_TOKEN"])
        df = pd.read_csv(g.get_repo(HOLDING_REPO).get_contents(HOLDINGS_FILE).download_url)
        
        # 關鍵修正 2：確保庫存代號也是 4 碼字串
        df['股票代號'] = df['股票代號'].astype(str).str.strip().str.zfill(4)
        
        # 關鍵修正 3：處理空值 (None / NaN)，強制轉為 0，避免運算崩潰
        if '累積配息' not in df.columns: 
            df['累積配息'] = 0.0
        
        df['累積配息'] = pd.to_numeric(df['累積配息'], errors='coerce').fillna(0.0)
        df['買入均價'] = pd.to_numeric(df['買入均價'], errors='coerce').fillna(0.0)
        df['持有股數'] = pd.to_numeric(df['持有股數'], errors='coerce').fillna(0.0)
        
        return df[['股票代號', '買入均價', '持有股數', '累積配息']]
    except Exception as e:
        return pd.DataFrame(columns=["股票代號", "買入均價", "持有股數", "累積配息"])

def save_holdings(df):
    """將庫存存回 GitHub"""
    try:
        g = Github(st.secrets["general"]["GITHUB_TOKEN"])
        repo = g.get_repo(HOLDING_REPO)
        contents = repo.get_contents(HOLDINGS_FILE)
        repo.update_file(contents.path, f"V32 Sync {get_taiwan_time_str()}", df.to_csv(index=False), contents.sha)
        st.toast("✅ 資料已同步至雲端", icon="☁️")
    except Exception as e:
        st.error(f"儲存失敗: {e}")

# --- 主程式 ---
def main():
    st.title("💰 V32 真實收益戰情室")
    
    # 初始化 Session State
    if 'inventory' not in st.session_state: 
        st.session_state['inventory'] = load_holdings()
    
    # 載入爬蟲價格 (背景自動抓取你的 v32_dataset.csv)
    crawler_prices = load_v32_crawler_data()

    # --- 第一區：資產總覽 ---
    inv = st.session_state['inventory'].copy()
    if not inv.empty:
        res = []
        for _, r in inv.iterrows():
            code = r['股票代號']
            # 使用爬蟲數據，若無則暫時顯示買入均價
            curr = crawler_prices.get(code, r['買入均價']) 
            
            total_cost = r['買入均價'] * r['持有股數']
            market_val = curr * r['持有股數']
            unrealized_pl = market_val - total_cost
            true_pl = unrealized_pl + r['累積配息']
            
            res.append({
                '代號': code, 
                '持有股數': int(r['持有股數']), 
                '買入均價': r['買入均價'],
                '最新收盤價': curr, 
                '累積配息': r['累積配息'], 
                '真實總損益': true_pl,
                '真實報酬率%': (true_pl / total_cost * 100) if total_cost > 0 else 0
            })
        
        df_res = pd.DataFrame(res)
        
        # 顯示頂部數據卡
        c1, c2, c3 = st.columns(3)
        total_cost_sum = df_res['買入均價'].mul(df_res['持有股數']).sum()
        total_pl_sum = df_res['真實總損益'].sum()
        
        c1.metric("投入總成本", f"${total_cost_sum:,.0f}")
        c2.metric("累積總配息", f"${df_res['累積配息'].sum():,.0f}")
        c3.metric("總真實損益", f"${total_pl_sum:,.0f}", 
                  delta=f"{(total_pl_sum / (total_cost_sum or 1) * 100):.2f}%")

        # 顯示主要表格
        st.dataframe(
            df_res.style.format({
                '買入均價':'{:.2f}', 
                '最新收盤價':'{:.2f}', 
                '累積配息':'{:,.0f}',
                '真實總損益':'{:+,.0f}', 
                '真實報酬率%':'{:+.2f}%'
            }).map(color_surplus, subset=['真實總損益', '真實報酬率%']), 
            use_container_width=True, 
            hide_index=True
        )
    else:
        st.info("目前無庫存資料，請在下方新增。")

    st.divider()

    # --- 第二區：動作計算機 ---
    st.subheader("🛠️ 庫存異動與領息登記")
    with st.expander("點擊展開：加碼買入 / 領取配息小工具", expanded=True):
        col_sel, col_act, col_p, col_q = st.columns([1, 1, 1, 1])
        
        exist_codes = inv['股票代號'].tolist() if not inv.empty else []
        target = col_sel.selectbox("選擇股票", options=["+ 新增標的"] + exist_codes)
        action = col_act.selectbox("執行動作", options=["加碼買入", "登記領息", "手動校正"])
        
        if target == "+ 新增標的":
            new_code = st.text_input("輸入新股票代號 (如: 00919)")
            current_target = new_code.strip().zfill(4) if new_code else ""
        else:
            current_target = target

        if action == "加碼買入":
            price = col_p.number_input("買入單價", min_value=0.0, step=0.01)
            qty = col_q.number_input("買入股數", min_value=0, step=1000)
        elif action == "登記領息":
            div_amt = col_p.number_input("本次領息總額", min_value=0.0, step=1.0)
            st.caption("※ 系統會直接累加至『累積配息』欄位")
        else:
            price = col_p.number_input("校正均價", min_value=0.0, step=0.01)
            qty = col_q.number_input("校正總股數", min_value=0, step=1)

        if st.button("🔥 執行並更新狀態", type="primary"):
            if not current_target:
                st.error("請輸入或選擇股票代號！")
                return

            temp_inv = st.session_state['inventory'].copy()
            
            # 抓取舊有數據，確保無 None 值
            match = temp_inv[temp_inv['股票代號'] == current_target]
            old_q = float(match['持有股數'].values[0]) if not match.empty else 0.0
            old_p = float(match['買入均價'].values[0]) if not match.empty else 0.0
            old_d = float(match['累積配息'].values[0]) if not match.empty else 0.0
            
            if action == "加碼買入":
                new_total_q = old_q + qty
                new_avg_p = ((old_p * old_q) + (price * qty)) / new_total_q if new_total_q > 0 else 0
                new_row = {'股票代號': current_target, '買入均價': round(new_avg_p, 2), '持有股數': new_total_q, '累積配息': old_d}
            elif action == "登記領息":
                new_row = {'股票代號': current_target, '買入均價': old_p, '持有股數': old_q, '累積配息': old_d + div_amt}
            else: # 手動校正
                new_row = {'股票代號': current_target, '買入均價': price, '持有股數': qty, '累積配息': old_d}

            # 移除舊列，新增更新後的列
            temp_inv = temp_inv[temp_inv['股票代號'] != current_target]
            temp_inv = pd.concat([temp_inv, pd.DataFrame([new_row])], ignore_index=True)
            
            # 更新並儲存
            st.session_state['inventory'] = temp_inv
            save_holdings(temp_inv)
            st.success(f"✅ {current_target} 狀態更新成功！")
            time.sleep(1)
            
            # 清除 cache 以便抓取最新資料 (如果需要)
            st.rerun()

    # --- 第三區：強制刷新 ---
    st.divider()
    if st.button("🔄 重新載入爬蟲數據庫"):
        load_v32_crawler_data.clear() # 清除快取，強制重拉 GitHub 爬蟲資料
        st.session_state['inventory'] = load_holdings() # 重讀庫存
        st.rerun()

if __name__ == "__main__":
    main()
