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

# 獲取股價
def get_current_price(symbol, v32_df):
    # 1. 先看 V32 掃描結果有沒有
    if not v32_df.empty:
        match = v32_df[v32_df['代號'] == str(symbol)]
        if not match.empty:
            for col in ['收盤', '現價', 'Price', 'Close']:
                if col in match.columns:
                    return float(match.iloc[0][col]), True
    
    # 2. 榜內沒有，用 yfinance 抓即時
    try:
        ticker_symbol = f"{symbol}.TW"
        stock = yf.Ticker(ticker_symbol)
        # 使用 fast_info 或是 history 抓取最新價
        # 嘗試抓取 history (較穩定)
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
        code_col = next((c for c in ['代碼', '代號', 'Code', 'Symbol', '股票代號'] if c in df.columns), None)
        if code_col:
            df[code_col] = df[code_col].astype(str).str.strip()
            df = df.rename(columns={code_col: '代號'})
        if '總分' in df.columns:
            df['總分'] = pd.to_numeric(df['總分'], errors='coerce').fillna(0)
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)

@st.cache_data(ttl=60)
def load_csv_holdings():
    file_path = 'holdings.csv'
    if not os.path.exists(file_path): return []
    try:
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    except:
        return []

# --- 主程式 ---
def main():
    st.title("📈 V32 戰情室")
    st.caption(f"最後更新: {get_taiwan_time()}")

    if 'holdings' not in st.session_state:
        st.session_state['holdings'] = load_csv_holdings()

    tab_scan, tab_monitor = st.tabs(["🚀 Top 10 掃描", "💼 庫存/損益試算"])

    # === Tab 1: 掃描 ===
    with tab_scan:
        v32_df, error = load_v32_data()
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
                st.dataframe(v32_df[v32_df['cat']=='General'].head(10).drop(columns=excludes, errors='ignore'), use_container_width=True, hide_index=True)
            with t2: 
                st.dataframe(v32_df[v32_df['cat']=='Special'].head(10).drop(columns=excludes, errors='ignore'), use_container_width=True, hide_index=True)
        else:
            if error: st.error(error)
            st.warning("暫無掃描資料")

    # === Tab 2: 庫存管理 (精簡版) ===
    with tab_monitor:
        st.markdown("### 📝 持股輸入與試算 (模擬交易)")
        
        # 輸入區塊 (移除停損停利)
        with st.expander("➕ 新增/試算持股 (點擊展開)", expanded=True):
            # 調整欄位比例
            c1, c2, c3, c4, c5 = st.columns([1.5, 2, 1.5, 1.5, 1])
            with c1: input_code = st.text_input("代號", placeholder="如 2330")
            with c2: input_name = st.text_input("名稱 (選填)", placeholder="如 台積電")
            with c3: input_cost = st.number_input("買入均價", min_value=0.0, step=0.1)
            with c4: input_qty = st.number_input("股數 (張x1000)", min_value=0, step=1000, value=1000)
            with c5:
                st.write("") 
                st.write("") 
                if st.button("加入"):
                    if input_code and input_qty > 0:
                        new_stock = {
                            "股票代號": input_code,
                            "股票名稱": input_name if input_name else input_code,
                            "買入均價": input_cost,
                            "持有股數": input_qty
                        }
                        st.session_state['holdings'].append(new_stock)
                        st.success(f"已加入 {input_code}")
                        st.rerun()
                    else:
                        st.error("請輸入代號與股數")

        st.divider()

        # 計算與顯示
        if st.session_state['holdings']:
            display_data = []
            
            p_bar = st.progress(0)
            total_items = len(st.session_state['holdings'])
            
            for i, item in enumerate(st.session_state['holdings']):
                code = str(item['股票代號'])
                qty = float(item['持有股數'])
                cost_p = float(item['買入均價'])
                
                # 抓價
                curr_price, is_v32 = get_current_price(code, v32_df)
                
                # 損益計算邏輯修正：若現價為 0，則不計算虧損
                if curr_price > 0:
                    cost_total = cost_p * qty
                    mv_total = curr_price * qty
                    pl = mv_total - cost_total
                    roi = (pl / cost_total * 100) if cost_total > 0 else 0
                else:
                    # 現價為 0 (抓不到數據)，將損益歸零，避免顯示 -100%
                    cost_total = cost_p * qty
                    mv_total = 0 
                    pl = 0 
                    roi = 0
                
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
                    "V32分數": health
                })
                p_bar.progress((i + 1) / total_items)
            
            p_bar.empty()
            
            # 總覽
            st.subheader("📊 資產總覽 (模擬)")
            df_res = pd.DataFrame(display_data)
            
            # 計算總合時，要注意現價為 0 的項目不應計入「錯誤的虧損」，但市值會少算
            # 這裡維持簡單加總，但因為上面已經把 PL 設為 0，所以總損益不會莫名大虧
            t_cost = (df_res['成本'] * df_res['股數']).sum()
            t_val = (df_res['現價'] * df_res['股數']).sum()
            
            # 修正總損益顯示：只計算有抓到價格的股票的損益，或者是簡單加總 (因上面已歸零，加總即為 0)
            t_pl = df_res['損益'].sum()
            t_roi = (t_pl / t_cost * 100) if t_cost > 0 else 0
            
            c1, c2, c3 = st.columns(3)
            c1.metric("總市值", f"${t_val:,.0f}")
            c2.metric("總成本", f"${t_cost:,.0f}")
            c3.metric("總損益", f"${t_pl:,.0f}", f"{t_roi:.2f}%")
            
            # 明細表
            st.subheader("📋 持股明細")
            
            if st.button("🗑️ 清空試算資料"):
                st.session_state['holdings'] = []
                st.rerun()

            st.dataframe(
                df_res.style.map(color_surplus, subset=['損益', '報酬率%'])
                .format({
                    "現價": "{:.2f}",
                    "成本": "{:.2f}",
                    "股數": "{:,.0f}",
                    "損益": "{:+,.0f}",
                    "報酬率%": "{:+.2f}%"
                }),
                use_container_width=True,
                height=500,
                hide_index=True
            )
            st.caption("🔴 紅色：獲利 | 🟢 綠色：虧損 | ⚠️ 若現價為 0.00，表示暫時無法取得數據，損益將顯示為 0。")
            
        else:
            st.info("目前無持股資料，請在上方輸入加入。")

if __name__ == "__main__":
    main()
