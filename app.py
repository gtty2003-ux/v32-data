import streamlit as st
import pandas as pd

# --- 模擬用：初始化 Session State (若已存在則不重置) ---
if 'inventory' not in st.session_state:
    # 這是實際儲存庫存的地方 [代號, 持有股數, 平均成本]
    st.session_state['inventory'] = pd.DataFrame(columns=['代號', '股數', '均價'])

# --- 模擬用：取得即時股價函數 (請替換成你原本 V32 的抓取邏輯) ---
def get_realtime_data(code):
    # 這裡只是模擬數據，請接上你原本的 crawler 或 API
    mock_data = {
        '2915': {'price': 55.0, 'attack': 89, 'signal': '🔴 強勢'},
        '1528': {'price': 15.9, 'attack': 89, 'signal': '🔴 強勢'},
        '3028': {'price': 38.0, 'attack': 85, 'signal': '🔴 強勢'},
        '1210': {'price': 51.0, 'attack': 67, 'signal': '⚪ 觀察'},
    }
    return mock_data.get(code, {'price': 0, 'attack': 0, 'signal': '無訊號'})

# ==========================================
# UI 區塊開始
# ==========================================

st.title("V32 庫存管理系統")

# --- 第一部分：交易輸入區 (買入 & 賣出) ---
st.subheader("1. 交易輸入")

col_buy, col_sell = st.columns(2)

with col_buy:
    st.markdown("##### 📥 新增買入 (Buy)")
    # 建立一個空的範本供使用者輸入
    input_buy_df = pd.DataFrame([{"代號": "", "股數": 1000, "均價": 0.0}])
    edited_buy = st.data_editor(
        input_buy_df, 
        num_rows="dynamic", 
        key="editor_buy",
        use_container_width=True,
        hide_index=True
    )

with col_sell:
    st.markdown("##### 📤 賣出調節 (Sell)")
    # 建立一個空的範本供使用者輸入
    input_sell_df = pd.DataFrame([{"代號": "", "股數": 1000, "賣出價": 0.0}])
    edited_sell = st.data_editor(
        input_sell_df, 
        num_rows="dynamic", 
        key="editor_sell",
        use_container_width=True,
        hide_index=True
    )

# --- 第二部分：儲存按鈕與邏輯處理 ---
st.write("") # Spacer
if st.button("💾 儲存變更 (更新庫存)", type="primary", use_container_width=True):
    current_inv = st.session_state['inventory'].copy()
    
    # 1. 處理買入 (邏輯：計算加權平均)
    for index, row in edited_buy.iterrows():
        if row['代號'] and row['股數'] > 0:
            code = str(row['代號'])
            new_shares = int(row['股數'])
            new_price = float(row['均價'])
            
            # 檢查是否已在庫存
            if code in current_inv['代號'].values:
                idx = current_inv[current_inv['代號'] == code].index[0]
                old_shares = current_inv.at[idx, '股數']
                old_price = current_inv.at[idx, '均價']
                
                # 加權平均公式：(舊股數*舊價 + 新股數*新價) / 總股數
                total_shares = old_shares + new_shares
                avg_cost = ((old_shares * old_price) + (new_shares * new_price)) / total_shares
                
                current_inv.at[idx, '股數'] = total_shares
                current_inv.at[idx, '均價'] = round(avg_cost, 2)
            else:
                # 新增一筆
                new_row = pd.DataFrame([{'代號': code, '股數': new_shares, '均價': new_price}])
                current_inv = pd.concat([current_inv, new_row], ignore_index=True)

    # 2. 處理賣出 (邏輯：減少股數，若歸零則刪除)
    for index, row in edited_sell.iterrows():
        if row['代號'] and row['股數'] > 0:
            code = str(row['代號'])
            sell_shares = int(row['股數'])
            
            if code in current_inv['代號'].values:
                idx = current_inv[current_inv['代號'] == code].index[0]
                current_shares = current_inv.at[idx, '股數']
                
                if current_shares > sell_shares:
                    current_inv.at[idx, '股數'] = current_shares - sell_shares
                    # 賣出通常不影響剩餘股票的單位成本，故不更新均價
                else:
                    # 全部賣光，移除該行
                    current_inv = current_inv.drop(idx)
    
    # 更新回 Session State
    st.session_state['inventory'] = current_inv
    st.success("庫存已更新！")
    st.rerun() # 重新執行以刷新下方顯示

# --- 第三部分：庫存監控儀表板 (下方顯示) ---
st.divider()
st.subheader("2. 庫存即時監控")

if not st.session_state['inventory'].empty:
    inventory_df = st.session_state['inventory'].copy()
    
    # 準備計算欄位
    display_rows = []
    total_cost = 0
    total_value = 0
    total_profit = 0
    
    for idx, row in inventory_df.iterrows():
        code = str(row['代號'])
        cost_price = float(row['均價'])
        shares = int(row['股數'])
        
        # 取得即時報價
        realtime = get_realtime_data(code)
        current_price = realtime['price']
        
        # 計算個別數據
        market_val = current_price * shares
        cost_val = cost_price * shares
        profit = market_val - cost_val
        roi = (profit / cost_val * 100) if cost_val > 0 else 0
        
        # 累加總計
        total_cost += cost_val
        total_value += market_val
        total_profit += profit
        
        display_rows.append({
            "代號": code,
            "即時價": current_price,
            "成本均價": cost_price,
            "股數": shares,
            "損益": int(profit),
            "報酬率%": f"{roi:.2f}%",
            "攻擊分": realtime['attack'],
            "訊號": realtime['signal']
        })
    
    final_df = pd.DataFrame(display_rows)
    
    # 顯示上方大指標 (Metrics)
    m1, m2, m3 = st.columns(3)
    m1.metric("總成本", f"${total_cost:,.0f}")
    m2.metric("總損益", f"${total_profit:,.0f}", delta=f"{total_profit:,.0f}")
    m3.metric("總市值", f"${total_value:,.0f}")
    
    # 顯示詳細表格 (依照攻擊分排序)
    st.dataframe(
        final_df.sort_values(by="攻擊分", ascending=False),
        use_container_width=True,
        column_config={
            "攻擊分": st.column_config.ProgressColumn(
                "攻擊分",
                help="V32 攻擊分數",
                format="%d",
                min_value=0,
                max_value=100,
            ),
        },
        hide_index=True
    )
else:
    st.info("目前無庫存，請在上方新增買入資料。")
