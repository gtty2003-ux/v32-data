import streamlit as st
import pandas as pd
import random
import time
from datetime import datetime

# ==========================================
# 1. 系統配置與樣式 (System Config)
# ==========================================
st.set_page_config(
    page_title="V32 戰情室 (Attack Focus)",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義 CSS：優化按鈕、表格與台股紅綠色系
st.markdown("""
    <style>
    /* 全局字體與表格優化 */
    .stDataFrame { font-size: 1.1rem; }
    
    /* 按鈕樣式 */
    .stButton>button {
        width: 100%;
        font-weight: bold;
        border-radius: 8px;
    }
    
    /* 台股漲跌色系修正：Streamlit 預設是綠漲紅跌，這邊強制調整 Metrics */
    [data-testid="stMetricDelta"] svg { display: none; }
    
    /* 庫存輸入區塊背景 */
    .input-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. V32 核心邏輯與數據模擬 (Data Logic)
# ==========================================

# 初始化 Session State
if 'inventory' not in st.session_state:
    # 預設庫存結構
    st.session_state['inventory'] = pd.DataFrame(columns=['代號', '股數', '均價'])
    # 預填兩筆範例資料
    st.session_state['inventory'] = pd.DataFrame([
        {'代號': '2915', '股數': 4000, '均價': 52.5},
        {'代號': '1528', '股數': 1000, '均價': 14.8},
    ])

if 'input_key_counter' not in st.session_state:
    st.session_state['input_key_counter'] = 0

# --- [核心] V32 選股數據產生器 ---
# 註：實際運用時，請在此函數內串接你的爬蟲 (TWSE/Yahoo Finance)
# 目前使用模擬數據，符合你的條件：股價 < 80, 依照攻擊分排序
@st.cache_data(ttl=600)  # 數據緩存 10 分鐘
def get_v32_market_data():
    # 模擬台股清單
    stock_list = [
        {'id': '2915', 'name': '潤泰全', 'price': 55.0, 'chg': 1.5, 'vol': 5000},
        {'id': '1528', 'name': '安得勝', 'price': 15.9, 'chg': 0.2, 'vol': 1200},
        {'id': '3028', 'name': '增你強', 'price': 38.0, 'chg': -0.5, 'vol': 3000},
        {'id': '1210', 'name': '大成',   'price': 51.0, 'chg': 0.0, 'vol': 800},
        {'id': '2603', 'name': '長榮',   'price': 155.0, 'chg': 2.0, 'vol': 15000}, # 超過 80 元，應被過濾
        {'id': '2317', 'name': '鴻海',   'price': 102.0, 'chg': 1.0, 'vol': 20000},
        {'id': '2303', 'name': '聯電',   'price': 48.5, 'chg': 0.3, 'vol': 45000},
        {'id': '2884', 'name': '玉山金', 'price': 25.4, 'chg': -0.1, 'vol': 12000},
        {'id': '6269', 'name': '台郡',   'price': 78.2, 'chg': 1.2, 'vol': 2500},
        {'id': '8069', 'name': '元太',   'price': 180.0, 'chg': -5.0, 'vol': 6000},
    ]
    
    # 增加更多隨機數據以模擬選股池
    for i in range(20):
        price = round(random.uniform(10, 90), 2)
        stock_list.append({
            'id': f'99{i:02d}', 
            'name': f'模擬股{i}', 
            'price': price, 
            'chg': round(random.uniform(-2, 2), 2), 
            'vol': random.randint(500, 5000)
        })

    v32_data = []
    
    for s in stock_list:
        # V32 篩選條件 1: 股價 < 80
        if s['price'] > 80:
            continue
            
        # V32 評分邏輯 (模擬)：技術面(70%) + 籌碼面(30%)
        # 這裡用隨機數模擬計算結果
        tech_score = random.randint(50, 95)
        chip_score = random.randint(40, 90)
        total_score = int(tech_score * 0.7 + chip_score * 0.3)
        
        # 訊號判定
        if total_score >= 85: signal = "🔴 強勢"
        elif total_score >= 70: signal = "🟡 轉強"
        else: signal = "⚪ 觀察"
        
        v32_data.append({
            '代號': s['id'],
            '名稱': s['name'],
            '現價': s['price'],
            '漲跌': s['chg'],
            '成交量': s['vol'],
            '攻擊分': total_score,
            '訊號': signal
        })
    
    # 轉為 DataFrame 並排序 (Top 20)
    df = pd.DataFrame(v32_data)
    df = df.sort_values(by='攻擊分', ascending=False).reset_index(drop=True)
    return df.head(20) # 只取前 20 名

# 載入市場數據
market_df = get_v32_market_data()

# 輔助函數：從市場數據中撈取特定股票資訊
def get_stock_info(code):
    row = market_df[market_df['代號'] == code]
    if not row.empty:
        return row.iloc[0].to_dict()
    else:
        # 若選股池沒有，則回傳模擬數據 (防止報錯)
        return {
            '代號': code, '名稱': code, '現價': 0, '漲跌': 0, 
            '攻擊分': 0, '訊號': '無數據'
        }

# ==========================================
# 3. 頁面佈局 (Layout)
# ==========================================

# 側邊欄
with st.sidebar:
    st.header("V32 控制台")
    st.write(f"今日日期: {datetime.now().strftime('%Y-%m-%d')}")
    st.info("💡 篩選條件：\n1. 股價 < 80元\n2. 攻擊分 Top 20\n3. 權重: 技術70%/量能30%")
    if st.button("🔄 強制刷新數據"):
        st.cache_data.clear()
        st.rerun()

# 主頁面 Tab 分頁
tab1, tab2 = st.tabs(["📊 庫存管理 (Inventory)", "🚀 V32 選股排行 (Screener)"])

# ==========================================
# Tab 1: 庫存管理 (你的新需求)
# ==========================================
with tab1:
    st.subheader("我的持股戰情")
    
    # --- A. 交易輸入區 (分離式設計) ---
    with st.expander("📝 交易登錄 (點擊展開)", expanded=True):
        c1, c2 = st.columns(2)
        key_idx = st.session_state['input_key_counter']
        
        with c1:
            st.markdown("##### 📥 新增買入")
            df_buy_in = pd.DataFrame([{"代號": "", "股數": 1000, "成交均價": 0.0}])
            edited_buy = st.data_editor(df_buy_in, num_rows="dynamic", key=f"buy_{key_idx}", use_container_width=True, hide_index=True)
        
        with c2:
            st.markdown("##### 📤 賣出調節")
            df_sell_in = pd.DataFrame([{"代號": "", "股數": 1000, "成交均價": 0.0}])
            edited_sell = st.data_editor(df_sell_in, num_rows="dynamic", key=f"sell_{key_idx}", use_container_width=True, hide_index=True)
        
        # 儲存按鈕
        if st.button("💾 儲存交易變更", type="primary"):
            current_inv = st.session_state['inventory'].copy()
            updated = False
            
            # 處理買入
            for _, row in edited_buy.iterrows():
                code = str(row['代號']).strip()
                if code and row['股數'] > 0:
                    updated = True
                    shares = int(row['股數'])
                    price = float(row['成交均價'])
                    
                    if code in current_inv['代號'].values:
                        idx = current_inv[current_inv['代號'] == code].index[0]
                        old_s = current_inv.at[idx, '股數']
                        old_p = current_inv.at[idx, '均價']
                        new_avg = ((old_s * old_p) + (shares * price)) / (old_s + shares)
                        current_inv.at[idx, '股數'] = old_s + shares
                        current_inv.at[idx, '均價'] = new_avg
                    else:
                        new_row = pd.DataFrame([{'代號': code, '股數': shares, '均價': price}])
                        current_inv = pd.concat([current_inv, new_row], ignore_index=True)
            
            # 處理賣出
            for _, row in edited_sell.iterrows():
                code = str(row['代號']).strip()
                if code and row['股數'] > 0:
                    updated = True
                    shares = int(row['股數'])
                    if code in current_inv['代號'].values:
                        idx = current_inv[current_inv['代號'] == code].index[0]
                        cur_s = current_inv.at[idx, '股數']
                        if cur_s > shares:
                            current_inv.at[idx, '股數'] = cur_s - shares
                        else:
                            current_inv = current_inv.drop(idx)

            if updated:
                st.session_state['inventory'] = current_inv
                st.session_state['input_key_counter'] += 1
                st.success("交易已更新！")
                time.sleep(0.5)
                st.rerun()

    st.markdown("---")

    # --- B. 庫存監控表格 ---
    if not st.session_state['inventory'].empty:
        inv_df = st.session_state['inventory'].copy()
        
        # 計算即時損益
        report_data = []
        total_cost_sum = 0
        total_mkt_sum = 0
        
        for _, row in inv_df.iterrows():
            code = str(row['代號'])
            shares = int(row['股數'])
            cost_p = float(row['均價'])
            
            # 連結 V32 市場數據
            info = get_stock_info(code)
            curr_p = info.get('現價', cost_p) # 若無報價則用成本價暫代
            
            mkt_val = shares * curr_p
            cost_val = shares * cost_p
            profit = mkt_val - cost_val
            roi = (profit / cost_val) if cost_val > 0 else 0
            
            total_cost_sum += cost_val
            total_mkt_sum += mkt_val
            
            report_data.append({
                "代號": code,
                "名稱": info.get('名稱', code),
                "現價": curr_p,
                "漲跌": info.get('漲跌', 0),
                "持有成本": cost_p,
                "股數": shares,
                "損益金額": int(profit),
                "報酬率": roi,
                "攻擊分": info.get('攻擊分', 0),
                "訊號": info.get('訊號', '無')
            })
            
        final_inv_df = pd.DataFrame(report_data)
        
        # 頂部大數據
        tot_profit = total_mkt_sum - total_cost_sum
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("總市值", f"${total_mkt_sum:,.0f}")
        col_m2.metric("總成本", f"${total_cost_sum:,.0f}")
        col_m3.metric("總損益", f"${tot_profit:,.0f}", delta=f"{tot_profit:,.0f}", delta_color="normal")
        
        # 顯示詳細表格
        st.dataframe(
            final_inv_df,
            use_container_width=True,
            column_config={
                "現價": st.column_config.NumberColumn(format="$%.2f"),
                "持有成本": st.column_config.NumberColumn(format="$%.2f"),
                "報酬率": st.column_config.NumberColumn(format="%.2f%%"),
                "攻擊分": st.column_config.ProgressColumn(format="%d", min_value=0, max_value=100),
            },
            hide_index=True
        )
    else:
        st.info("目前無庫存，請上方新增交易。")

# ==========================================
# Tab 2: V32 選股排行 (Screener)
# ==========================================
with tab2:
    st.subheader(f"🔥 V32 強勢選股 Top 20 (股價 < 80)")
    
    # 這裡顯示的是經過篩選後的市場數據
    st.dataframe(
        market_df,
        use_container_width=True,
        column_config={
            "現價": st.column_config.NumberColumn(format="$%.2f"),
            "漲跌": st.column_config.NumberColumn(format="%.2f"),
            "攻擊分": st.column_config.ProgressColumn(
                format="%d", 
                min_value=0, 
                max_value=100,
                help="技術面(70%) + 籌碼面(30%) 綜合評分"
            ),
            "成交量": st.column_config.NumberColumn(format="%d 張"),
        },
        hide_index=True
    )
    
    st.caption("※ 數據來源：V32 模擬爬蟲 (實際部署時請連結 TWSE API)")
