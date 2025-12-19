# --- 批次運算引擎 (修正版：擴大抓取範圍至 6mo) ---
@st.cache_data(ttl=3600)
def run_v32_engine(ticker_list):
    results = []
    p_bar = st.progress(0)
    status = st.empty()
    total = len(ticker_list)
    
    for i, row in enumerate(ticker_list):
        symbol = str(row['代號'])
        name = str(row.get('名稱', ''))
        
        status.text(f"正在分析: {symbol} {name} ({i+1}/{total})...")
        p_bar.progress((i + 1) / total)
        
        try:
            stock = yf.Ticker(f"{symbol}.TW")
            
            # 🔥【關鍵修正】改成 "6mo" (6個月)
            # 3個月只有約60天，會被下面的 <65 過濾掉
            # 6個月約120天，這才夠算 MA60 + 回溯
            hist = stock.history(period="6mo")
            
            # 資料不足 65 天者，直接剔除
            if len(hist) < 65:
                continue 
            
            # 資料充足才進行運算
            t_s, v_s, final_s, stab = calculate_indicators(hist)
            
            results.append({
                '代號': symbol, '名稱': name,
                '收盤': hist['Close'].iloc[-1],
                '成交量': hist['Volume'].iloc[-1],
                '技術分': t_s,   
                '量能分': v_s,   
                'V32總分': final_s,
                '穩定度': stab   
            })
                
        except Exception as e:
            continue
            
    p_bar.empty()
    status.empty()
    return pd.DataFrame(results)
