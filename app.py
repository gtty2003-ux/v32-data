import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import pytz
import yfinance as yf
from github import Github 
import time
from FinMind.data import DataLoader 

# --- 設定頁面資訊 ---
st.set_page_config(
    page_title="V32 戰情室 (Attack Focus)",
    layout="wide",
    page_icon="⚔️"
)

# --- 樣式設定 ---
st.markdown("""
    <style>
    .stDataFrame thead tr th {
        background-color: #ffebee !important; 
        color: #b71c1c !important;
        font-weight: bold;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 全域變數 ---
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

def color_stability(val):
    if not isinstance(val, str): return ''
    try:
        score = int(val.split('/')[0])
        if score <= 2: return 'color: #E65100; font-weight: bold;'
        elif score >= 3: return 'color: #2E7D32; font-weight: bold;'
    except: pass
    return ''

# --- 籌碼分析函數 ---
def get_chip_analysis(symbol_list):
    chip_data = []
    dl = DataLoader()
    p_bar = st.progress(0)
    status = st.empty()
    total = len(symbol_list)
    start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
    
    for i, symbol in enumerate(symbol_list):
        status.text(f"🔍 分析籌碼結構: {symbol} ({i+1}/{total})...")
        p_bar.progress((i + 1) / total)
        try:
            df = dl.taiwan_stock_institutional_investors(stock_id=symbol, start_date=start_date)
            if df.empty:
                chip_data.append({'代號': symbol, '投信(張)': 0, '外資(張)': 0, '主力動向': '⚪ 資料不足'})
                continue
            latest_date = df['date'].iloc[-1]
            day_data = df[df['date'] == latest_date]
            foreign_net = day_data[day_data['name'].str.contains('Foreign')]['buy'].sum() - \
                          day_data[day_data['name'].str.contains('Foreign')]['sell'].sum()
            foreign_buy = int(foreign_net // 1000)
            trust_net = day_data[day_data['name'] == 'Investment_Trust']['buy'].sum() - \
                        day_data[day_data['name'] == 'Investment_Trust']['sell'].sum()
            trust_buy = int(trust_net // 1000)
            
            status_str = ""
            if trust_buy > 0: status_str += "🔴 投信買 "
            elif trust_buy < 0: status_str += "🟢 投信賣 "
            if foreign_buy > 1000: status_str += "🔥 外資大買 "
            elif foreign_buy < -1000: status_str += "🧊 外資倒貨 "
            
            if trust_buy > 0 and foreign_buy > 0: final_tag = "🚀 土洋合買"
            elif trust_buy > 0 and foreign_buy < 0: final_tag = "⚔️ 土洋對作(信)"
            elif trust_buy < 0 and foreign_buy > 0: final_tag = "⚔️ 土洋對作(外)"
            elif trust_buy < 0 and foreign_buy < 0: final_tag = "☠️ 主力棄守"
            elif trust_buy == 0 and abs(foreign_buy) < 50: final_tag = "⚪ 籌碼觀望"
            else: final_tag = "🟡 一般輪動"
                
            chip_data.append({
                '代號': symbol, '投信(張)': trust_buy, '外資(張)': foreign_buy,
                '主力動向': f"{final_tag} | {status_str}"
            })
            time.sleep(0.05) 
        except Exception as e:
            chip_data.append({'代號': symbol, '投信(張)': 0, '外資(張)': 0, '主力動向': f'❌ {str(e)}'})
            
    p_bar.empty()
    status.empty()
    return pd.DataFrame(chip_data)

# --- V32 指標運算 ---
def calculate_indicators(hist):
    if len(hist) < 65: return 0, 0, 0, "0/5"
    close = hist['Close']
    vol = hist['Volume']
    high = hist['High']
    open_p = hist['Open']
    
    ma5_s = close.rolling(5).mean()
    ma20_s = close.rolling(20).mean()
    ma60_s = close.rolling(60).mean()
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi_s = 100 - (100 / (1 + rs))

    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd_s = exp1 - exp2
    signal_s = macd_s.ewm(span=9, adjust=False).mean()

    vol_ma20_s = vol.rolling(20).mean()
    vol_ma5_s = vol.rolling(5).mean()
    high_20_s = high.rolling(20).max()

    raw_scores = [] 
    lookback_indices = range(-7, 0)

    for i in lookback_indices:
        c_now = close.iloc[i]
        ma5 = ma5_s.iloc[i]
        ma20 = ma20_s.iloc[i]
        ma20_prev = ma20_s.iloc[i-1] 
        ma60 = ma60_s.iloc[i]
        rsi_now = rsi_s.iloc[i]
        macd_now = macd_s.iloc[i]
        sig_now = signal_s.iloc[i]
        high_20_prev = high_20_s.iloc[i-1] 
        v_now = vol.iloc[i]
        v_prev = vol.iloc[i-1]
        v_ma20 = vol_ma20_s.iloc[i]
        v_ma5 = vol_ma5_s.iloc[i]
        o_now = open_p.iloc[i]

        t_score = 60
        if not np.isnan(ma20) and c_now > ma20: t_score += 5          
        if not np.isnan(ma20) and not np.isnan(ma20_prev) and ma20 > ma20_prev: t_score += 5      
        if not np.isnan(ma5) and not np.isnan(ma20) and not np.isnan(ma60):
            if ma5 > ma20 and ma20 > ma60: t_score += 10 
        if not np.isnan(rsi_now) and rsi_now > 50: t_score += 5          
        if not np.isnan(rsi_now) and rsi_now > 70: t_score += 5          
        if not np.isnan(macd_now) and not np.isnan(sig_now) and macd_now > sig_now: t_score += 5    
        if not np.isnan(high_20_prev) and c_now > high_20_prev: t_score += 10 

        v_score = 60
        if not np.isnan(v_ma20) and v_now > v_ma20: v_score += 10      
        if not np.isnan(v_ma5) and v_now > v_ma5: v_score += 10        
        is_red = c_now > o_now
        vol_increase = v_now > v_prev
        if is_red and vol_increase: v_score += 15 
        if not np.isnan(v_ma20) and v_now > v_ma20 * 1.5: v_score += 5 

        t_score = min(100, t_score)
        v_score = min(100, v_score)
        daily_total = (t_score * 0.7) + (v_score * 0.3)
        raw_scores.append(daily_total)

    raw_scores = [0 if np.isnan(x) else x for x in raw_scores]
    if len(raw_scores) < 2: return 0, 0, 0, "0/5"
    raw_today = raw_scores[-1]
    raw_yesterday = raw_scores[-2]
    attack_score = (raw_today * 0.7) + (raw_yesterday * 0.3)
    last_5_days = raw_scores[-5:]
    stability_count = sum(1 for s in last_5_days if s >= 70)
    stability_str = f"{stability_count}/5"

    return t_score, v_score, attack_score, stability_str

# --- 運算引擎 ---
@st.cache_data(ttl=3600)
def run_v32_engine(ticker_list):
    results = []
    p_bar = st.progress(0)
    status = st.empty()
    total = len(ticker_list)
    
    for i, row in enumerate(ticker_list):
        symbol = str(row['代號'])
        name = str(row.get('名稱', ''))
        status.text(f"正在掃描: {symbol} {name} ({i+1}/{total})...")
        p_bar.progress((i + 1) / total)
        try:
            stock = yf.Ticker(f"{symbol}.TW")
            hist = stock.history(period="6mo")
            if len(hist) < 65: continue 
            t_s, v_s, atk_s, stab = calculate_indicators(hist)
            results.append({
                '代號': symbol, '名稱': name,
                '收盤': hist['Close'].iloc[-1],
                '技術分': t_s, '量能分': v_s, '攻擊分': atk_s, '穩定度': stab   
            })
        except: continue
            
    p_bar.empty()
    status.empty()
    return pd.DataFrame(results)

# --- 資料載入 ---
def load_and_process_data():
    url = f"https://raw.githubusercontent.com/{REPO_KEY}/main/v32_recommend.csv"
    try:
        df = pd.read_csv(url)
        code_col = next((c for c in ['代碼', '代號', 'Code', 'Symbol'] if c in df.columns), None)
        if code_col:
            df[code_col] = df[code_col].astype(str).str.strip()
            df = df.rename(columns={code_col: '代號'})
            processed = run_v32_engine(df[['代號', '名稱']].to_dict('records'))
            return processed, None
    except Exception as e:
        return pd.DataFrame(), str(e)

# --- GitHub 庫存存取 ---
def load_holdings():
    try:
        token = st.secrets["general"]["GITHUB_TOKEN"]
        g = Github(token)
        repo = g.get_repo(REPO_KEY)
        contents = repo.get_contents(FILE_PATH)
        df = pd.read_csv(contents.download_url)
        df['股票代號'] = df['股票代號'].astype(str).str.strip()
        for c in ["股票代號", "買入均價", "持有股數"]:
            if c not in df.columns: df[c] = 0 if c != "股票代號" else ""
        return df
    except:
        return pd.DataFrame(columns=["股票代號", "買入均價", "持有股數"])

def save_holdings(df):
    try:
        token = st.secrets["general"]["GITHUB_TOKEN"]
        g = Github(token)
        repo = g.get_repo(REPO_KEY)
        csv_content = df.to_csv(index=False)
        try:
            contents = repo.get_contents(FILE_PATH)
            repo.update_file(contents.path, f"Update {get_taiwan_time()}", csv_content, contents.sha)
            st.success("✅ 交易已儲存至雲端！")
        except:
            repo.create_file(FILE_PATH, "Create holdings.csv", csv_content)
            st.success("✅ 建立並儲存成功！")
    except Exception as e:
        st.error(f"❌ 儲存失敗: {e}")

# --- 庫存更新邏輯 ---
def update_inventory(buy_data, sell_data):
    df = load_holdings()
    # 處理買入
    if buy_data and buy_data['code']:
        code = buy_data['code']
        qty_add = buy_data['zhang'] * 1000
        price_in = buy_data['price']
        if code in df['股票代號'].values:
            idx = df[df['股票代號'] == code].index[0]
            old_qty = df.at[idx, '持有股數']
            old_cost = df.at[idx, '買入均價']
            new_qty = old_qty + qty_add
            if new_qty > 0:
                new_cost = ((old_qty * old_cost) + (qty_add * price_in)) / new_qty
            else:
                new_cost = price_in
            df.at[idx, '持有股數'] = new_qty
            df.at[idx, '買入均價'] = new_cost
        else:
            new_row = pd.DataFrame({'股票代號': [code], '買入均價': [price_in], '持有股數': [qty_add]})
            df = pd.concat([df, new_row], ignore_index=True)
    # 處理賣出
    if sell_data and sell_data['code']:
        code = sell_data['code']
        qty_sell = sell_data['zhang'] * 1000
        if code in df['股票代號'].values:
            idx = df[df['股票代號'] == code].index[0]
            current_qty = df.at[idx, '持有股數']
            new_qty = current_qty - qty_sell
            if new_qty <= 0:
                df = df.drop(idx)
            else:
                df.at[idx, '持有股數'] = new_qty
    save_holdings(df)

# --- 篩選與排序邏輯 ---
def get_stratified_selection(df):
    if df.empty: return df, []
    cols = ['攻擊分', '技術分', '量能分']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    mask = (df['技術分'] >= 80) & (df['量能分'] >= 60) & (df['攻擊分'] >= 86) & (df['攻擊分'] <= 92)
    filtered = df[mask].copy()
    if filtered.empty: return pd.DataFrame(), ["無符合條件標的"]
    b_a = filtered[(filtered['攻擊分'] >= 90) & (filtered['攻擊分'] <= 92)].sort_values('攻擊分', ascending=False).head(5)
    b_b = filtered[(filtered['攻擊分'] >= 88) & (filtered['攻擊分'] < 90)].sort_values('攻擊分', ascending=False).head(5)
    b_c = filtered[(filtered['攻擊分'] >= 86) & (filtered['攻擊分'] < 88)].sort_values('攻擊分', ascending=False).head(5)
    final = pd.concat([b_a, b_b, b_c])
    stats = [f"90-92: {len(b_a)}", f"88-90: {len(b_b)}", f"86-88: {len(b_c)}"]
    return final, stats

def get_raw_top10(df):
    if df.empty: return df
    df['攻擊分'] = pd.to_numeric(df['攻擊分'], errors='coerce').fillna(0)
    return df.sort_values(by='攻擊分', ascending=False).head(10)

# --- 主程式 ---
def main():
    st.title("⚔️ V32 戰情室 (Attack Focus)")
    st.caption(f"最後更新: {get_taiwan_time()} | 核心邏輯：攻擊力優先 + FinMind 籌碼輔助")
    
    v32_df, err = load_and_process_data()
    if err: st.error(err)

    if not v32_df.empty:
        v32_df['cat'] = v32_df.apply(lambda r: 'Special' if ('債' in str(r.get('名稱')) or 'KY' in str(r.get('名稱')) or str(r['代號']).startswith(('00','91')) or str(r['代號'])[-1].isalpha() or (len(str(r['代號']))>4 and str(r['代號']).isdigit())) else 'General', axis=1)
        v32_df = v32_df[v32_df['cat'] == 'General']

    tab_strat, tab_raw, tab_inv = st.tabs(["🎯 今日攻擊力 Top 15", "🏆 原始攻擊分 Top 10", "💼 庫存管理"])
    fmt_score = {'收盤':'{:.2f}', '技術分':'{:.0f}', '量能分':'{:.0f}', '攻擊分':'{:.1f}', '外資(張)': '{:,.0f}', '投信(張)': '{:,.0f}'}

    # === Tab 1: 分層精選 ===
    with tab_strat:
        if not v32_df.empty:
            final_df, stats = get_stratified_selection(v32_df)
            st.info(f"🎯 分層結構：{' | '.join(stats)} (排序依據：攻擊分)")
            if not final_df.empty:
                st.markdown("#### 🕵️ 籌碼結構偵測")
                if st.button("🚀 啟動籌碼掃描 (查詢三大法人動向)", key="btn_strat_scan"):
                    with st.spinner("正在連線 FinMind 歷史資料庫..."):
                        chip_df = get_chip_analysis(final_df['代號'].tolist())
                        if not chip_df.empty:
                            final_df = pd.merge(final_df, chip_df, on='代號', how='left')
                
                cols_to_show = ['代號','名稱','收盤','攻擊分','穩定度','技術分','量能分']
                if '主力動向' in final_df.columns: cols_to_show += ['主力動向', '投信(張)', '外資(張)']
                
                # --- [修復關鍵] 嘗試繪製顏色，失敗則略過，防止當機 ---
                styler = final_df[cols_to_show].style.format(fmt_score).map(color_stability, subset=['穩定度'])
                try:
                    styler = styler.background_gradient(subset=['攻擊分'], cmap='Reds')
                except Exception:
                    pass # 忽略 matplotlib 錯誤
                
                st.dataframe(styler, hide_index=True, use_container_width=True)
            else: st.warning("無符合條件的一般個股。")
        else: st.warning("暫無資料")

    # === Tab 2: Top 10 ===
    with tab_raw:
        st.markdown("### 🏆 全市場攻擊力排行 (Top 10)")
        if not v32_df.empty:
            raw_df = get_raw_top10(v32_df)
            if not raw_df.empty:
                st.markdown("#### 🕵️ 籌碼結構偵測")
                if st.button("🚀 啟動籌碼掃描 (Top 10)", key="btn_raw_scan"):
                    with st.spinner("正在連線 FinMind 歷史資料庫..."):
                        chip_df = get_chip_analysis(raw_df['代號'].tolist())
                        if not chip_df.empty:
                            raw_df = pd.merge(raw_df, chip_df, on='代號', how='left')
                
                cols_to_show = ['代號','名稱','收盤','攻擊分','穩定度','技術分','量能分']
                if '主力動向' in raw_df.columns: cols_to_show += ['主力動向', '投信(張)', '外資(張)']

                # --- [修復關鍵] 嘗試繪製顏色，失敗則略過 ---
                styler = raw_df[cols_to_show].style.format(fmt_score).map(color_stability, subset=['穩定度'])
                try:
                    styler = styler.background_gradient(subset=['攻擊分'], cmap='Reds')
                except Exception:
                    pass
                
                st.dataframe(styler, hide_index=True, use_container_width=True)
            else: st.info("無資料")
        else: st.warning("暫無資料")

    # === Tab 3: 庫存管理 ===
    with tab_inv:
        st.subheader("📝 交易登錄")
        with st.form("trade_form", clear_on_submit=True):
            col_buy, col_sell = st.columns(2)
            with col_buy:
                st.markdown("### 🔴 買入")
                b_code = st.text_input("代號", key="b_code", placeholder="例如: 2330")
                b_zhang = st.number_input("張數", min_value=0.0, step=1.0, key="b_zhang")
                b_price = st.number_input("成交均價", min_value=0.0, step=0.1, key="b_price")
            with col_sell:
                st.markdown("### 🟢 賣出")
                s_code = st.text_input("代號", key="s_code", placeholder="例如: 2330")
                s_zhang = st.number_input("張數", min_value=0.0, step=1.0, key="s_zhang")
                s_price = st.number_input("成交均價", min_value=0.0, step=0.1, key="s_price")
            st.markdown("---")
            submitted = st.form_submit_button("💾 執行交易並儲存", type="primary")
            
            if submitted:
                buy_data = {'code': b_code, 'zhang': b_zhang, 'price': b_price} if b_code and b_zhang > 0 else None
                sell_data = {'code': s_code, 'zhang': s_zhang, 'price': s_price} if s_code and s_zhang > 0 else None
                if buy_data or sell_data:
                    with st.spinner("正在更新雲端庫存..."):
                        update_inventory(buy_data, sell_data)
                    time.sleep(1)
                    st.rerun()
                else: st.warning("⚠️ 請至少輸入買入或賣出的資料")

        st.divider()
        st.subheader("💼 我的庫存")
        current_holdings = load_holdings()
        if not current_holdings.empty:
            res = []
            score_map = {}
            if not v32_df.empty:
                score_map = v32_df.set_index('代號')['攻擊分'].to_dict()
            progress_bar = st.progress(0)
            total_rows = len(current_holdings)
            for idx, r in current_holdings.iterrows():
                progress_bar.progress((idx + 1) / total_rows)
                if not r['股票代號']: continue
                code = str(r['股票代號'])
                qty = float(r['持有股數'] or 0)
                cost = float(r['買入均價'] or 0)
                curr = 0; nm = code; sc = 0; signal = "⚪ 資料不足"
                try:
                    stock = yf.Ticker(f"{code}.TW")
                    h = stock.history(period="1mo") 
                    if not h.empty:
                        curr = h['Close'].iloc[-1]
                        if code in score_map:
                            match = v32_df[v32_df['代號'] == code].iloc[0]
                            nm = match['名稱']; sc = match['攻擊分']
                        else:
                            nm = stock.info.get('shortName', code); sc = 0 
                        ma20 = h['Close'].rolling(20).mean().iloc[-1]
                        if not np.isnan(ma20) and curr < ma20: signal = "🔴 破線(停損)"
                        elif sc > 0 and sc < 60: signal = "🟡 熄火(停利)"
                        elif sc == 0:
                            if curr >= ma20: signal = "⚪ 榜外(觀察)"
                            else: signal = "🔴 破線(榜外)"
                        else: signal = "🟢 續抱"
                except Exception as e: pass
                val = curr * qty; c_tot = cost * qty; pl = val - c_tot
                roi = (pl/c_tot*100) if c_tot>0 else 0
                score_display = f"{sc:.1f}" if sc > 0 else "N/A"
                res.append({'代號': code, '名稱': nm, '現價': curr, '成本': cost, '股數': qty, '損益': pl, '報酬率%': roi, '攻擊分': score_display, '建議': signal})
            progress_bar.empty()
            if res:
                df_res = pd.DataFrame(res)
                c1, c2, c3 = st.columns(3)
                c1.metric("總成本", f"${(df_res['成本']*df_res['股數']).sum():,.0f}")
                total_pl = df_res['損益'].sum()
                c2.metric("總損益", f"${total_pl:,.0f}", delta=f"{total_pl:,.0f}")
                c3.metric("總市值", f"${(df_res['現價']*df_res['股數']).sum():,.0f}")
                def color_signal(val):
                    if "🔴" in val: return 'color: white; background-color: #d32f2f; font-weight: bold;'
                    if "🟡" in val: return 'color: black; background-color: #fbc02d; font-weight: bold;'
                    if "🟢" in val: return 'color: white; background-color: #388e3c; font-weight: bold;'
                    return ''
                st.dataframe(df_res.style.map(color_surplus, subset=['損益','報酬率%']).map(color_signal, subset=['建議']).format({'現價':'{:.2f}','損益':'{:+,.0f}','報酬率%':'{:+.2f}%', '股數':'{:.0f}'}), use_container_width=True, hide_index=True)
        else: st.info("目前無庫存資料，請在上方新增交易。")

if __name__ == "__main__":
    main()
