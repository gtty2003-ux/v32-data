import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import pytz
import yfinance as yf
from github import Github 
import time
import twstock  # <--- 新增這個套件

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

# --- 新增：籌碼分析函數 (使用 twstock) ---
def get_chip_analysis(symbol_list):
    """
    針對篩選後的清單抓取三大法人資料
    """
    chip_data = []
    
    # 進度條
    p_bar = st.progress(0)
    status = st.empty()
    total = len(symbol_list)
    
    for i, symbol in enumerate(symbol_list):
        status.text(f"🔍 分析籌碼結構: {symbol} ({i+1}/{total})...")
        p_bar.progress((i + 1) / total)
        
        try:
            stock = twstock.Stock(symbol)
            # 抓取最近 5 日的三大法人資料
            # twstock 的 institutional 屬性會回傳列表，最新在後
            inst = stock.institutional 
            
            if not inst or len(inst) < 1:
                chip_data.append({'代號': symbol, '投信': '無資料', '外資': '無資料', '主力動向': '⚪ 資料不足'})
                continue
                
            # 取得最近一日與累積數據
            last_day = inst[-1] # [日期, 外資買賣超, 投信買賣超, 自營商買賣超, 合計]
            prev_day = inst[-2] if len(inst) > 1 else last_day
            
            # 數據清洗 (twstock 有時回傳 None)
            foreign_buy = int(last_day[1]) if last_day[1] else 0
            trust_buy = int(last_day[2]) if last_day[2] else 0
            dealer_buy = int(last_day[3]) if last_day[3] else 0
            
            # --- 簡易籌碼邏輯判定 ---
            status_str = ""
            
            # 1. 投信判定 (權重最高)
            if trust_buy > 0:
                status_str += "🔴 投信買進 "
            elif trust_buy < 0:
                status_str += "🟢 投信賣出 "
                
            # 2. 外資判定
            if foreign_buy > 1000: # 外資買超大於 1000 張
                status_str += "🔥 外資大買 "
            elif foreign_buy < -1000:
                status_str += "🧊 外資倒貨 "
            
            # 3. 土洋對作/合作
            if trust_buy > 0 and foreign_buy > 0:
                final_tag = "🚀 土洋合買"
            elif trust_buy > 0 and foreign_buy < 0:
                final_tag = "⚔️ 土洋對作(信)" # 投信買，外資賣
            elif trust_buy < 0 and foreign_buy > 0:
                final_tag = "⚔️ 土洋對作(外)" # 外資買，投信賣
            elif trust_buy < 0 and foreign_buy < 0:
                final_tag = "☠️ 主力棄守"
            elif trust_buy == 0 and abs(foreign_buy) < 50:
                final_tag = "⚪ 籌碼觀望"
            else:
                final_tag = "🟡 一般輪動"
                
            chip_data.append({
                '代號': symbol,
                '投信(張)': trust_buy,
                '外資(張)': foreign_buy,
                '主力動向': f"{final_tag} | {status_str}"
            })
            
            time.sleep(0.5) # 避免太快被證交所擋
            
        except Exception as e:
            chip_data.append({'代號': symbol, '投信(張)': 0, '外資(張)': 0, '主力動向': '❌ 讀取失敗'})
            
    p_bar.empty()
    status.empty()
    return pd.DataFrame(chip_data)

# --- 核心：V32 指標運算 (維持原樣) ---
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

# --- 運算引擎 (Engine) ---
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
        return df[["股票代號", "買入均價", "持有股數"]]
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
            st.success("✅ 儲存成功！")
        except:
            repo.create_file(FILE_PATH, "Create holdings.csv", csv_content)
            st.success("✅ 建立並儲存成功！")
    except Exception as e:
        st.error(f"❌ 儲存失敗: {e}")

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
    st.caption(f"最後更新: {get_taiwan_time()} | 核心邏輯：攻擊力優先 + 籌碼輔助")
    
    v32_df, err = load_and_process_data()
    if err: st.error(err)

    if not v32_df.empty:
        v32_df['cat'] = v32_df.apply(lambda r: 'Special' if ('債' in str(r.get('名稱')) or 'KY' in str(r.get('名稱')) or str(r['代號']).startswith(('00','91')) or str(r['代號'])[-1].isalpha() or (len(str(r['代號']))>4 and str(r['代號']).isdigit())) else 'General', axis=1)
        v32_df = v32_df[v32_df['cat'] == 'General']

    tab_strat, tab_raw, tab_inv = st.tabs(["🎯 今日攻擊力 Top 15", "🏆 原始攻擊分 Top 10", "💼 庫存管理"])
    fmt_score = {'收盤':'{:.2f}', '技術分':'{:.0f}', '量能分':'{:.0f}', '攻擊分':'{:.1f}', '外資(張)': '{:,.0f}', '投信(張)': '{:,.0f}'}

    # === Tab 1: 分層精選 + 籌碼分析 ===
    with tab_strat:
        if not v32_df.empty:
            final_df, stats = get_stratified_selection(v32_df)
            st.info(f"🎯 分層結構：{' | '.join(stats)} (排序依據：攻擊分)")
            
            if not final_df.empty:
                # --- 新增功能區塊 ---
                st.markdown("#### 🕵️ 籌碼結構偵測")
                if st.button("🚀 啟動籌碼掃描 (查詢三大法人動向)"):
                    with st.spinner("正在連線證交所抓取資料，請稍候..."):
                        chip_df = get_chip_analysis(final_df['代號'].tolist())
                        # 合併資料
                        final_df = pd.merge(final_df, chip_df, on='代號', how='left')
                
                # 顯示表格
                cols_to_show = ['代號','名稱','收盤','攻擊分','穩定度','技術分','量能分']
                if '主力動向' in final_df.columns:
                    cols_to_show += ['主力動向', '投信(張)', '外資(張)']
                
                st.dataframe(
                    final_df[cols_to_show]
                    .style
                    .format(fmt_score)
                    .background_gradient(subset=['攻擊分'], cmap='Reds')
                    .map(color_stability, subset=['穩定度']), 
                    hide_index=True, 
                    use_container_width=True
                )
            else:
                st.warning("無符合條件的一般個股。")
        else:
            st.warning("暫無資料")

    # === Tab 2: Top 10 (維持原樣) ===
    with tab_raw:
        st.markdown("### 🏆 全市場攻擊力排行 (Top 10)")
        if not v32_df.empty:
            raw_df = get_raw_top10(v32_df)
            if not raw_df.empty:
                st.dataframe(
                    raw_df[['代號','名稱','收盤','攻擊分','穩定度','技術分','量能分']]
                    .style
                    .format(fmt_score)
                    .background_gradient(subset=['攻擊分'], cmap='Reds')
                    .map(color_stability, subset=['穩定度']),
                    hide_index=True, 
                    use_container_width=True
                )

    # === Tab 3: 庫存管理 (維持原樣) ===
    with tab_inv:
        st.subheader("📝 庫存編輯器")
        if 'editor_data' not in st.session_state:
            st.session_state['editor_data'] = load_holdings()
            
        edited = st.data_editor(
            st.session_state['editor_data'],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "股票代號": st.column_config.TextColumn("代號", required=True),
                "買入均價": st.column_config.NumberColumn("均價", format="%.2f"),
                "持有股數": st.column_config.NumberColumn("股數", step=1000)
            }, key="inv_editor"
        )
        if st.button("💾 儲存變更"):
            save_holdings(edited)
            st.rerun()
        
        st.divider()
        
        if not edited.empty:
            # (此處為庫存診斷邏輯，為節省篇幅省略，若需完整版請告知，通常這段不需修改)
            # 這裡只要貼上你原本程式碼 Tab 3 的後半段即可
            res = []
            score_map = {}
            if not v32_df.empty:
                score_map = v32_df.set_index('代號')['攻擊分'].to_dict()

            progress_text = st.empty()
            
            for idx, r in edited.iterrows():
                if not r['股票代號']: continue
                code = str(r['股票代號'])
                qty = float(r['持有股數'] or 0)
                cost = float(r['買入均價'] or 0)
                
                curr = 0
                nm = code
                sc = 0
                signal = "⚪ 資料不足"
                
                try:
                    stock = yf.Ticker(f"{code}.TW")
                    h = stock.history(period="1mo") 
                    if not h.empty:
                        curr = h['Close'].iloc[-1]
                        if code in score_map:
                            match = v32_df[v32_df['代號'] == code].iloc[0]
                            nm = match['名稱']
                            sc = match['攻擊分']
                        else:
                            nm = stock.info.get('shortName', code)
                            sc = 0 
                        
                        ma20 = h['Close'].rolling(20).mean().iloc[-1]
                        if not np.isnan(ma20) and curr < ma20: signal = "🔴 破線(停損)"
                        elif sc > 0 and sc < 60: signal = "🟡 熄火(停利)"
                        elif sc == 0:
                            if curr >= ma20: signal = "⚪ 榜外(觀察)"
                            else: signal = "🔴 破線(榜外)"
                        else: signal = "🟢 續抱"
                except Exception as e: pass
                
                val = curr * qty
                c_tot = cost * qty
                pl = val - c_tot
                roi = (pl/c_tot*100) if c_tot>0 else 0
                score_display = f"{sc:.1f}" if sc > 0 else "N/A"
                
                res.append({'代號': code, '名稱': nm, '現價': curr, '成本': cost, '股數': qty, '損益': pl, '報酬率%': roi, '攻擊分': score_display, '建議': signal})
            
            progress_text.empty()
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

                st.dataframe(
                    df_res.style
                    .map(color_surplus, subset=['損益','報酬率%'])
                    .map(color_signal, subset=['建議'])
                    .format({'現價':'{:.2f}','損益':'{:+,.0f}','報酬率%':'{:+.2f}%'}), 
                    use_container_width=True, 
                    hide_index=True
                )

if __name__ == "__main__":
    main()
