import streamlit as st
import pandas as pd
import os
from datetime import datetime
import pytz
import yfinance as yf

# --- 設定頁面資訊 ---
st.set_page_config(
    page_title="V32 戰情室",
    layout="wide",
    page_icon="📈"
)

# --- 樣式設定 ---
st.markdown("""
    <style>
    /* 表頭顏色: 淺綠色 */
    .stDataFrame thead tr th {
        background-color: #C8E6C9 !important;
        color: #000000 !important;
    }
    /* 指標數值放大 */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 工具函數 ---
def get_taiwan_time():
    utc_now = datetime.utcnow()
    tw_time = utc_now.replace(tzinfo=pytz.utc).astimezone(pytz.timezone('Asia/Taipei'))
    return tw_time.strftime("%Y-%m-%d %H:%M:%S")

def color_surplus(val):
    """台股慣例：漲紅(>0)、跌綠(<0)、平黑(0)"""
    if val > 0: return 'color: red'
    elif val < 0: return 'color: green'
    return 'color: black'

# 獲取股價 (優先查 V32 表，沒有則查 Yahoo Finance)
def get_current_price(symbol, v32_df):
    # 1. 先看 V32 掃描結果有沒有 (最快，且代表還在榜內)
    if not v32_df.empty:
        # 嘗試對應代號
        match = v32_df[v32_df['代號'] == str(symbol)]
        if not match.empty:
            # 嘗試抓取可能的價格欄位
            for col in ['收盤', '現價', 'Price', 'Close']:
                if col in match.columns:
                    # 回傳: (價格, 是否在榜內)
                    return float(match.iloc[0][col]), True
    
    # 2. 榜內沒有，用 yfinance 抓即時 (代表榜外)
    try:
        ticker_symbol = f"{symbol}.TW"
        stock = yf.Ticker(ticker_symbol)
        data = stock.history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1], False 
    except:
        pass
    
    return 0, False

# --- 資料讀取 ---
@st.cache_data(ttl=60)
def load_v32_data():
    file_path = 'v32_recommend.csv'
    if not os.path.exists(file_path): return pd.DataFrame(), "找不到 V32 資料"
    try:
        df = pd.read_csv(file_path)
        # 欄位正規化：統一找 '代號'
        code_col = next((c for c in ['代碼', '代號', 'Code', 'Symbol', '股票代號'] if c in df.columns), None)
        if code_col:
            df[code_col] = df[code_col].astype(str).str.strip()
            df = df.rename(columns={code_col: '代號'})
        
        # 確保總分是數字
        if '總分' in df.columns:
            df['總分'] = pd.to_numeric(df['總分'], errors='coerce').fillna(0)
            
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)

@st.cache_data(ttl=60)
def load_csv_holdings():
    """嘗試讀取 holdings.csv，如果沒有檔案就回傳空清單，不報錯"""
    file_path = 'holdings.csv'
    if not os.path.exists(file_path): return []
    try:
        df = pd.read_csv(file_path)
        # 轉成 List of Dict 方便 Session State 操作
        return df.to_dict('records')
    except:
        return []

# --- 主程式 ---
def main():
    st.title("📈 V32 戰情室")
    st.caption(f"最後更新: {get_taiwan_time()}")

    # 初始化 Session State 
    # (如果沒有 holdings.csv，這裡就會是空的，等待使用者手動輸入)
    if 'holdings' not in st.session_state:
        st.session_state['holdings'] = load_csv_holdings()

    tab_scan, tab_monitor = st.tabs(["🚀 Top 10 掃描", "💼 庫存/損益試算"])

    # === Tab 1: 掃描 (分類邏輯強化版) ===
    with tab_scan:
        v32_df, error = load_v32_data()
        
        if not v32_df.empty:
            # 定義嚴格的分類邏輯
            def get_cat(row):
                c = str(row['代號']) # 代號
                n = str(row.get('名稱', row.get('Name', row.get('股票名稱', '')))) # 名稱
                
                # 1. 關鍵字過濾
                if '債' in n: return 'Special' # 債券
                if 'KY' in n: return 'Special' # KY股
                
                # 2. 代號規則過濾
                if c.startswith('00'): return 'Special' # ETF
                if c.startswith('91'): return 'Special' # DR
                
                # 3. 後綴英文過濾 (通殺: 特別股A/B, ETF槓桿L/反向R/債券B/期貨U)
                if c[-1].isalpha(): return 'Special'
                
                # 4. 長度過濾 (排除可轉債等5碼純數字)
                if len(c) > 4 and c.isdigit(): return 'Special'
                
                # 剩下的才是純種一般個股
                return 'General'

            v32_df['cat'] = v32_df.apply(get_cat, axis=1)
            
            # 分流顯示
            t1, t2 = st.tabs(["🏢 一般個股", "📊 ETF/特殊"])
            excludes = ['Unnamed: 0', 'cat']
            
            with t1: 
                df_gen = v32_df[v32_df['cat']=='General'].head(10)
                if not df_gen.empty:
                    st.dataframe(df_gen.drop(columns=excludes, errors='ignore'), use_container_width=True, hide_index=True)
                    st.caption(f"✅ 純一般個股 (排除 ETF, KY, DR, 特別股, 債券)。")
                else:
                    st.info("無符合的一般個股。")

            with t2: 
                df_spec = v32_df[v32_df['cat']=='Special'].head(10)
                if not df_spec.empty:
                    st.dataframe(df_spec.drop(columns=excludes, errors='ignore'), use_container_width=True, hide_index=True)
                    st.caption(f"📋 特殊類別 (包含 ETF, KY, 特別股, 債券等)。")
                else:
                    st.info("無符合的特殊類股。")
        else:
            if error: st.error(error)
            st.warning("暫無掃描資料，請確認 GitHub 是否有 v32_recommend.csv")

    # === Tab 2: 庫存管理 (手動輸入版) ===
    with tab_monitor:
        st.markdown("### 📝 持股輸入與試算 (模擬交易)")
        
        # 輸入區塊
        with st.expander("➕ 新增/試算持股 (點擊展開)", expanded=True):
            c1, c2, c3, c4, c5, c6 = st.columns([1.5, 2, 1.5, 1.5, 1.5, 1])
            with c1: input_code = st.text_input("代號", placeholder="如 2330")
            with c2: input_name = st.text_input("名稱 (選填)", placeholder="如 台積電")
            with c3: input_cost = st.number_input("買入均價", min_value=0.0, step=0.1)
            with c4: input_qty = st.number_input("股數 (張x1000)", min_value=0, step=1000, value=1000)
            with c5: 
                input_sl = st.number_input("停損價", min_value=0.0)
                input_tp = st.number_input("停利價", min_value=0.0)
            with c6:
                st.write("") # 排版佔位
                st.write("") 
                if st.button("加入"):
                    if input_code and input_qty > 0:
                        new_stock = {
                            "股票代號": input_code,
                            "股票名稱": input_name if input_name else input_code,
                            "買入均價": input_cost,
                            "持有股數": input_qty,
                            "停損價格": input_sl,
                            "停利價格": input_tp
                        }
                        st.session_state['holdings'].append(new_stock)
                        st.success(f"已加入 {input_code}")
                        st.rerun()
                    else:
                        st.error("請輸入代號與股數")

        st.divider()

        # 計算與顯示區塊
        if st.session_state['holdings']:
            display_data = []
            
            # 進度條 (提升體驗)
            p_bar = st.progress(0)
            total_items = len(st.session_state['holdings'])
            
            for i, item in enumerate(st.session_state['holdings']):
                code = str(item['股票代號'])
                qty = float(item['持有股數'])
                cost_p = float(item['買入均價'])
                
                # 抓價 (V32榜內 -> 即時)
                curr_price, is_v32 = get_current_price(code, v32_df)
                
                # 損益計算
                cost_total = cost_p * qty
                mv_total = curr_price * qty
                pl = mv_total - cost_total
                roi = (pl / cost_total * 100) if cost_total > 0 else 0
                
                # 操作建議
                action = "續抱"
                if curr_price > 0:
                    if float(item['停損價格']) > 0 and curr_price <= float(item['停損價格']): 
                        action = "⚠️ 破停損"
                    elif float(item['停利價格']) > 0 and curr_price >= float(item['停利價格']): 
                        action = "🎯 達停利"
                
                # V32 健康度
                health = "⚠️ 榜外"
                if is_v32 and not v32_df.empty:
                    match = v32_df[v32_df['代號'] == code]
                    if not match.empty:
                        health = f"{float(match.iloc[0]['總分']):.1f} 分"
                
                display_data.append({
                    "代號": code,
                    "名稱": item['股票名稱'],
                    "現價": curr_price,
                    "成本": cost_p,
                    "股數": qty,
                    "損益": pl,
                    "報酬率%": roi,
                    "V32分數": health,
                    "建議": action
                })
                p_bar.progress((i + 1) / total_items)
            
            p_bar.empty()
            
            # --- 儀表板總覽 ---
            st.subheader("📊 資產總覽 (模擬)")
            df_res = pd.DataFrame(display_data)
            t_cost = (df_res['成本'] * df_res['股數']).sum()
            t_val = (df_res['現價'] * df_res['股數']).sum()
            t_pl = t_val - t_cost
            t_roi = (t_pl / t_cost * 100) if t_cost > 0 else 0
            
            c1, c2, c3 = st.columns(3)
            c1.metric("總市值", f"${t_val:,.0f}")
            c2.metric("總成本", f"${t_cost:,.0f}")
            c3.metric("總損益", f"${t_pl:,.0f}", f"{t_roi:.2f}%")
            
            # --- 明細表格 ---
            st.subheader("📋 持股明細")
            
            # 清空按鈕
            if st.button("🗑️ 清空試算資料"):
                st.session_state['holdings'] = []
                st.rerun()

            st.dataframe(
                df_res.style.map(color_surplus, subset=['損益', '報酬率%'])
                .format({
                    "現價": "{:.2f}",
                    "成本": "{:.2f}",
                    "股數": "{:,.0f}",
                    "損益": "{:+,.0f}", # 顯示正負號
                    "報酬率%": "{:+.2f}%"
                }),
                use_container_width=True,
                height=500,
                hide_index=True
            )
            st.caption("🔴 紅色：獲利 | 🟢 綠色：虧損 | 💡 網頁重新整理後資料將重置")
            
        else:
            st.info("目前無持股資料，請在上方輸入加入。")

if __name__ == "__main__":
    main()
