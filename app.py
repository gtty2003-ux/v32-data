import streamlit as st
import pandas as pd
import os
from datetime import datetime
import pytz
import yfinance as yf
from github import Github 

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

# --- 全域變數設定 ---
# ⚠️ 這裡只放公開的倉庫名稱，Token 請放在 Streamlit Secrets
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

# --- 智慧名稱獲取 (Cache 加速) ---
@st.cache_data(ttl=86400) # 名字通常不會變，快取 24 小時
def fetch_name_from_web(symbol):
    try:
        # 嘗試用 yfinance 抓取名稱 (針對榜外股票)
        t = yf.Ticker(f"{symbol}.TW")
        # 優先抓短名，沒有則抓長名，再沒有就回傳代號
        return t.info.get('shortName') or t.info.get('longName') or symbol
    except:
        return symbol

# --- 整合式股價與資訊獲取 ---
def get_stock_info(symbol, v32_df):
    """回傳: (現價, 名稱, 是否為V32榜內)"""
    symbol_str = str(symbol)
    
    # 1. 先查 V32 榜單 (速度最快)
    if not v32_df.empty:
        match = v32_df[v32_df['代號'] == symbol_str]
        if not match.empty:
            # 抓價格
            price = 0
            for col in ['收盤', '現價', 'Price', 'Close']:
                if col in match.columns:
                    price = float(match.iloc[0][col])
                    break
            
            # 抓名稱 (優先用 V32 表裡的名稱)
            name = str(match.iloc[0].get('名稱', match.iloc[0].get('Name', symbol_str)))
            return price, name, True

    # 2. 榜內沒有，用 yfinance 抓即時 (榜外)
    try:
        stock = yf.Ticker(f"{symbol_str}.TW")
        data = stock.history(period="1d")
        
        # 抓價格
        price = 0
        if not data.empty:
            price = data['Close'].iloc[-1]
            
        # 抓名稱 (因為不在 V32 表裡，所以去網路上補抓)
        name = fetch_name_from_web(symbol_str)
        
        return price, name, False
    except:
        pass
    
    return 0, symbol_str, False

# --- GitHub 存取函數 ---
def load_data_from_github():
    try:
        token = st.secrets["general"]["GITHUB_TOKEN"]
        g = Github(token)
        repo = g.get_repo(REPO_KEY)
        contents = repo.get_contents(FILE_PATH)
        df = pd.read_csv(contents.download_url)
        df['股票代號'] = df['股票代號'].astype(str).str.strip()
        
        # 為了介面乾淨，我們載入時只取這三個欄位
        # (舊的資料如果有名稱欄位，這裡會自動過濾掉，達成瘦身目的)
        required_cols = ["股票代號", "買入均價", "持有股數"]
        # 確保欄位存在，若無則補齊
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
        
        # 儲存時只存 3 個欄位，不存名稱
        csv_content = df.to_csv(index=False)
        
        try:
            contents = repo.get_contents(FILE_PATH)
            repo.update_file(contents.path, f"Update holdings {get_taiwan_time()}", csv_content, contents.sha)
            st.success(f"✅ 資料已成功儲存！ ({get_taiwan_time()})")
        except:
            repo.create_file(FILE_PATH, "Create holdings.csv", csv_content)
            st.success("✅ 已建立新庫存檔並儲存！")
            
    except Exception as e:
        st.error(f"❌ 儲存失敗: {e}")

# --- 讀取 V32 掃描檔 ---
@st.cache_data(ttl=60)
def load_v32_data():
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
    
    v32_df, err = load_v32_data()

    tab_scan, tab_holdings = st.tabs(["🚀 Top 10 掃描", "💼 庫存管理與損益"])

    # === Tab 1: 掃描 ===
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
            with t1: 
                df_gen = v32_df[v32_df['cat']=='General'].head(10)
                if not df_gen.empty:
                    st.dataframe(df_gen.drop(columns=excludes, errors='ignore'), use_container_width=True, hide_index=True)
            with t2: 
                df_spec = v32_df[v32_df['cat']=='Special'].head(10)
                if not df_spec.empty:
                    st.dataframe(df_spec.drop(columns=excludes, errors='ignore'), use_container_width=True, hide_index=True)
        else:
            st.warning("暫無掃描資料")

    # === Tab 2: 庫存管理 (智慧補名版) ===
    with tab_holdings:
        st.subheader("📝 庫存編輯器")
        st.caption("輸入代號、成本與股數即可，名稱會自動帶入。")
        
        if 'editor_data' not in st.session_state:
            st.session_state['editor_data'] = load_data_from_github()

        # 編輯器只顯示三個欄位，乾淨俐落
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

        if st.button("💾 儲存變更至雲端"):
            save_data_to_github(edited_df)
            st.session_state['editor_data'] = edited_df
            st.rerun()

        st.divider()

        # 戰情儀表板
        st.subheader("📊 即時損益")
        
        if not edited_df.empty:
            display_data = []
            p_bar = st.progress(0)
            total = len(edited_df)
            
            for i, row in edited_df.iterrows():
                if not row['股票代號']: continue
                
                code = str(row['股票代號'])
                cost_p = float(row['買入均價']) if pd.notnull(row['買入均價']) else 0
                qty = float(row['持有股數']) if pd.notnull(row['持有股數']) else 0
                
                # 自動獲取現價與名稱 (Key Change Here)
                curr_price, stock_name, is_v32 = get_stock_info(code, v32_df)
                
                # 計算
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
                if is_v32:
                    match = v32_df[v32_df['代號'] == code]
                    if not match.empty:
                        health = f"{float(match.iloc[0]['總分']):.1f} 分"

                display_data.append({
                    "代號": code,
                    "名稱": stock_name, # 這裡會自動顯示抓到的名稱
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
                
                t_cost = (res_df['成本'] * res_df['股數']).sum()
                t_pl = res_df['損益'].sum()
                t_val = (res_df['現價'] * res_df['股數']).sum()
                t_roi = (t_pl / t_cost * 100) if t_cost > 0 else 0
                
                c1, c2, c3 = st.columns(3)
                c1.metric("總成本", f"${t_cost:,.0f}")
                c2.metric("總損益", f"${t_pl:,.0f}", f"{t_roi:.2f}%")
                c3.metric("總市值 (有效)", f"${t_val:,.0f}")
                
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
