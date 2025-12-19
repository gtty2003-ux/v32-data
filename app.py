import streamlit as st
import pandas as pd
import os
from datetime import datetime
import pytz
import yfinance as yf
from github import Github # 用來寫入資料

# --- 設定頁面資訊 ---
st.set_page_config(
    page_title="V32 戰情室",
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

# --- 全域變數設定 (請依實際情況修改) ---
# 這裡填您的 GitHub 帳號與倉庫名稱
REPO_KEY = "你的GitHub帳號/v32-data" # 例如: gtty2003-ux/v32-data
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

# --- GitHub 存取函數 (核心功能) ---
def load_data_from_github():
    """從 GitHub 下載 holdings.csv"""
    try:
        # 嘗試從 Secrets 拿 Token，如果沒有則報錯提醒
        token = st.secrets["general"]["GITHUB_TOKEN"]
        g = Github(token)
        repo = g.get_repo(REPO_KEY)
        contents = repo.get_contents(FILE_PATH)
        df = pd.read_csv(contents.download_url)
        # 確保欄位格式正確
        df['股票代號'] = df['股票代號'].astype(str).str.strip()
        return df
    except Exception as e:
        # 如果檔案不存在或讀取失敗，回傳空的 DataFrame
        # st.error(f"讀取失敗 (若是第一次使用請忽略): {e}")
        return pd.DataFrame(columns=["股票代號", "股票名稱", "買入均價", "持有股數"])

def save_data_to_github(df):
    """將 DataFrame 寫回 GitHub"""
    try:
        token = st.secrets["general"]["GITHUB_TOKEN"]
        g = Github(token)
        repo = g.get_repo(REPO_KEY)
        
        # 轉成 CSV 字串
        csv_content = df.to_csv(index=False)
        
        try:
            # 嘗試取得檔案 (更新模式)
            contents = repo.get_contents(FILE_PATH)
            repo.update_file(contents.path, f"Update holdings {get_taiwan_time()}", csv_content, contents.sha)
            st.success(f"✅ 資料已成功儲存至雲端！ ({get_taiwan_time()})")
        except:
            # 檔案不存在 (建立模式)
            repo.create_file(FILE_PATH, "Create holdings.csv", csv_content)
            st.success("✅ 已建立新庫存檔並儲存！")
            
    except Exception as e:
        st.error(f"❌ 儲存失敗，請檢查 Token 或 Repo 設定。\n錯誤訊息: {e}")

# --- 抓取股價邏輯 ---
def get_current_price(symbol, v32_df):
    # 1. 查 V32 榜單
    if not v32_df.empty:
        match = v32_df[v32_df['代號'] == str(symbol)]
        if not match.empty:
            for col in ['收盤', '現價', 'Price', 'Close']:
                if col in match.columns:
                    return float(match.iloc[0][col]), True
    # 2. 查 Yahoo
    try:
        ticker = f"{symbol}.TW"
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1], False 
    except:
        pass
    return 0, False

# --- 讀取 V32 掃描檔 ---
@st.cache_data(ttl=60)
def load_v32_data():
    # 這裡還是讀原本的掃描檔
    url = f"https://raw.githubusercontent.com/{REPO_KEY}/main/v32_recommend.csv"
    try:
        df = pd.read_csv(url)
        code_col = next((c for c in ['代碼', '代號', 'Code', 'Symbol'] if c in df.columns), None)
        if code_col:
            df[code_col] = df[code_col].astype(str).str.strip()
            df = df.rename(columns={code_col: '代號'})
        if '總分' in df.columns:
            df['總分'] = pd.to_numeric(df['總分'], errors='coerce').fillna(0)
        return df, None
    except:
        return pd.DataFrame(), "無法讀取 V32 資料"

# --- 主程式 ---
def main():
    st.title("📈 V32 戰情室")
    
    # 讀取 V32 資料
    v32_df, err = load_v32_data()

    tab_scan, tab_holdings = st.tabs(["🚀 Top 10 掃描", "💼 庫存管理與損益"])

    # === Tab 1: 掃描 (保持精簡) ===
    with tab_scan:
        if not v32_df.empty:
            def get_cat(row):
                c = str(row['代號'])
                n = str(row.get('名稱', row.get('Name', row.get('股票名稱', ''))))
                if '債' in n or 'KY' in n or c.startswith('00') or c.startswith('91') or c[-1].isalpha() or (len(c)>4 and c.isdigit()):
                    return 'Special'
                return 'General'
            v32_df['cat'] = v32_df.apply(get_cat, axis=1)
            t1, t2 = st.tabs(["🏢 一般個股", "📊 ETF/特殊"])
            excludes = ['Unnamed: 0', 'cat']
            with t1: st.dataframe(v32_df[v32_df['cat']=='General'].head(10).drop(columns=excludes, errors='ignore'), use_container_width=True, hide_index=True)
            with t2: st.dataframe(v32_df[v32_df['cat']=='Special'].head(10).drop(columns=excludes, errors='ignore'), use_container_width=True, hide_index=True)
        else:
            st.warning("暫無掃描資料")

    # === Tab 2: 庫存管理 (雲端版) ===
    with tab_holdings:
        st.subheader("📝 庫存編輯器 (直接修改)")
        st.caption("在此處新增、修改或刪除持股，完成後請點擊「儲存變更」以寫入雲端。")
        
        # 1. 讀取目前的雲端資料
        if 'editor_data' not in st.session_state:
            st.session_state['editor_data'] = load_data_from_github()

        # 2. 顯示編輯器 (Data Editor)
        # num_rows="dynamic" 讓您可以新增和刪除行
        edited_df = st.data_editor(
            st.session_state['editor_data'],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "股票代號": st.column_config.TextColumn("代號", help="請輸入代號", required=True),
                "股票名稱": st.column_config.TextColumn("名稱", required=True),
                "買入均價": st.column_config.NumberColumn("均價", min_value=0, format="%.2f"),
                "持有股數": st.column_config.NumberColumn("股數", min_value=0, step=1000),
            },
            key="holdings_editor" 
        )

        # 3. 儲存按鈕
        if st.button("💾 儲存變更至雲端"):
            save_data_to_github(edited_df)
            # 重新讀取確保同步
            st.session_state['editor_data'] = edited_df
            st.rerun()

        st.divider()

        # 4. 戰情儀表板 (唯讀，自動計算)
        st.subheader("📊 即時損益戰情室")
        
        if not edited_df.empty:
            display_data = []
            
            # 使用進度條
            p_bar = st.progress(0)
            total = len(edited_df)
            
            for i, row in edited_df.iterrows():
                # 跳過空行
                if not row['股票代號']: continue
                
                code = str(row['股票代號'])
                name = str(row['股票名稱'])
                cost_p = float(row['買入均價']) if row['買入均價'] else 0
                qty = float(row['持有股數']) if row['持有股數'] else 0
                
                curr_price, is_v32 = get_current_price(code, v32_df)
                
                # 計算損益 (防呆: 價為0則不計損益)
                if curr_price > 0:
                    val = curr_price * qty
                    cost = cost_p * qty
                    pl = val - cost
                    roi = (pl / cost * 100) if cost > 0 else 0
                else:
                    val = 0
                    cost = cost_p * qty
                    pl = 0
                    roi = 0
                
                health = "⚠️ 榜外"
                if is_v32 and not v32_df.empty:
                    match = v32_df[v32_df['代號'] == code]
                    if not match.empty:
                        health = f"{float(match.iloc[0]['總分']):.1f} 分"

                display_data.append({
                    "代號": code,
                    "名稱": name,
                    "現價": curr_price,
                    "成本": cost_p,
                    "股數": qty,
                    "損益": pl,
                    "報酬率%": roi,
                    "V32分數": health
                })
                if total > 0: p_bar.progress((i+1)/total)
            
            p_bar.empty()
            
            if display_data:
                res_df = pd.DataFrame(display_data)
                
                # 總計
                t_cost = (res_df['成本'] * res_df['股數']).sum()
                t_pl = res_df['損益'].sum()
                # 修正：市值只算有抓到價格的，避免低估，但損益是準的
                # 若要嚴謹，這裡的市值僅供參考
                t_val = (res_df['現價'] * res_df['股數']).sum() 
                t_roi = (t_pl / t_cost * 100) if t_cost > 0 else 0
                
                c1, c2, c3 = st.columns(3)
                c1.metric("總成本", f"${t_cost:,.0f}")
                c2.metric("總損益", f"${t_pl:,.0f}", f"{t_roi:.2f}%")
                c3.metric("總市值(參考)", f"${t_val:,.0f}")
                
                st.dataframe(
                    res_df.style.map(color_surplus, subset=['損益', '報酬率%'])
                    .format({
                        "現價": "{:.2f}",
                        "成本": "{:.2f}",
                        "股數": "{:,.0f}",
                        "損益": "{:+,.0f}",
                        "報酬率%": "{:+.2f}%"
                    }),
                    use_container_width=True, 
                    height=400,
                    hide_index=True
                )
        else:
            st.info("目前無持股，請在上方編輯器新增資料。")

if __name__ == "__main__":
    main()
