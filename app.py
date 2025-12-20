import pandas as pd
import zipfile
import io
import os
from google.colab import drive

# 1. 掛載雲端硬碟
drive.mount('/content/drive')

# 2. 定義檔案路徑 (請確認 ZIP 檔案在您雲端硬碟中的實際路徑)
# 根據您的連結，檔案名稱為 V32_Standard_Data.zip
ZIP_PATH = '/content/drive/MyDrive/V32_Standard_Data.zip' 

def process_v32_selection(zip_path):
    v32_results = []
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # 獲取壓縮檔內所有 CSV 檔案清單
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            print(f"📦 偵測到 {len(csv_files)} 檔股票數據，開始執行 V32 標準計算...")
            
            for csv_file in csv_files:
                with z.open(csv_file) as f:
                    # 讀取個別股票的歷史數據 
                    df = pd.read_csv(f)
                    if df.empty: continue
                    
                    # 獲取最新一筆成交資訊
                    latest = df.iloc[-1]
                    price = float(latest['收盤價'])
                    volume = float(latest['成交股數'])
                    
                    # --- V32 篩選門檻 ---
                    # 門檻 1：最近成交價必須低於 $80 元
                    if price < 80:
                        v32_results.append({
                            '股票代碼': latest['股票代碼'],
                            '股票名稱': latest['股票名稱'],
                            '收盤價': price,
                            '成交股數': volume,
                            '日期': latest['日期']
                        })
        
        # 3. 執行權重計算與排名
        result_df = pd.DataFrame(v32_results)
        if not result_df.empty:
            # 評分權重：技術分 (價格位階) 7:3 量能分 (成交量位階)
            result_df['A_Score'] = result_df['收盤價'].rank(pct=True) * 100
            result_df['B_Score'] = result_df['成交股數'].rank(pct=True) * 100
            result_df['Total_Score'] = (result_df['A_Score'] * 0.7) + (result_df['B_Score'] * 0.3)
            
            # 數量保證：顯示總分最高的 Top 20
            top_20 = result_df.sort_values(by='Total_Score', ascending=False).head(20)
            
            # 視覺化顯示 (模擬 Tab 1 需求)
            print(f"✅ V32 標準掃描完成！Top 20 標的已產出：")
            return top_20
        else:
            print("⚠️ 未發現符合低價門檻 (<80) 的標的。")
            return None

    except Exception as e:
        print(f"❌ 程式執行失敗: {e}")
        return None

# 執行選股
v32_final_list = process_v32_selection(ZIP_PATH)
if v32_final_list is not None:
    display(v32_final_list)
