#!/usr/bin/env python3
import akshare as ak
import pandas as pd
import time
import os
from datetime import datetime
from tqdm import tqdm  # 进度条

def get_stock_data_10years(stock_code, stock_name=None):
    """
    获取单个股票或ETF过去10年的日线数据（不复权）。
    如果本地已存在CSV文件，则直接读取，否则通过API获取并保存。
    """
    # 如果未提供股票名称，默认为股票代码
    if stock_name is None:
        stock_name = stock_code
        
    # 创建子目录保存股票数据
    data_dir = "stock_data_bfq"
    os.makedirs(data_dir, exist_ok=True)
    
    csv_filename = os.path.join(data_dir, f'{stock_code}.csv')
    
    try:
        # 1. 尝试从本地CSV文件读取数据（缓存机制）
        if os.path.exists(csv_filename):
            stock_df = pd.read_csv(csv_filename)
            print(f"✅ 已从本地文件 {csv_filename} 加载 {stock_code} ({stock_name}) 的数据。")
            # 确保日期列是datetime类型，便于后续处理
            stock_df['date'] = pd.to_datetime(stock_df['date'])
            return stock_df
        
        # 2. 如果文件不存在，通过akshare获取数据
        print(f"⏳ 开始通过API获取 {stock_code} ({stock_name}) 过去10年的数据...")
        
        # 设置获取数据的起止日期
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - pd.DateOffset(years=10)).strftime('%Y%m%d')
        
        # 使用akshare获取历史数据，adjust="" 表示不复权
        stock_df = ak.stock_zh_a_hist(
            symbol=stock_code, 
            period="daily", 
            start_date=start_date, 
            end_date=end_date, 
            adjust=""  # "" 表示不复权, "qfq" 前复权, "hfq" 后复权
        )
        
        # 检查返回的数据是否为空
        if stock_df is None or stock_df.empty:
            print(f"⚠️ 警告: {stock_code} ({stock_name}) 没有返回任何数据，可能代码有误或该时段无数据。")
            return None
        
        # 3. 数据清洗和整理
        # akshare返回的列名是中文，统一重命名为英文，方便处理
        stock_df = stock_df.rename(columns={
            '日期': 'date', 
            '开盘': 'open', 
            '收盘': 'close', 
            '最高': 'high', 
            '最低': 'low', 
            '成交量': 'volume',
            '成交额': 'turnover',
            '振幅': 'amplitude',
            '涨跌幅': 'pct_chg',
            '涨跌额': 'change',
            '换手率': 'turnover_rate'
        })
        
        # 将日期列转换为datetime对象
        stock_df['date'] = pd.to_datetime(stock_df['date'])
        
        # 4. 保存数据到CSV文件
        stock_df.to_csv(csv_filename, index=False)
        print(f"💾 数据已成功保存到 {csv_filename}。共获取 {len(stock_df)} 条记录。")
        
        return stock_df
            
    except Exception as e:
        print(f"❌ 获取或处理 {stock_code} ({stock_name}) 数据时发生严重错误: {e}")
        # 记录获取失败的股票
        with open("error_log.txt", "a", encoding='utf-8') as error_file:
            error_file.write(f"{datetime.now()}: {stock_code},{stock_name},{e}\n")
        return None

# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == "__main__":
    # 1. 定义需要获取数据的股票/ETF列表
    # 使用字典，key为股票代码，value为名称
    stocks_to_fetch = {
        '600900': '长江电力',
        '518880': '黄金ETF',
        '513100': '纳指ETF',
        '600030': '中信证券'
    }
    
    # 用于存储所有获取到的数据的字典
    all_data = {}

    print("========================================")
    print("🚀 开始执行数据获取任务...")
    print("========================================")

    # 2. 遍历列表，获取每只股票/ETF的数据
    # 使用tqdm来显示进度条
    for code, name in tqdm(stocks_to_fetch.items(), desc="总体进度"):
        print(f"\n--- 正在处理: {name} ({code}) ---")
        
        # 调用函数获取数据
        df = get_stock_data_10years(stock_code=code, stock_name=name)
        
        if df is not None:
            all_data[code] = df
            print(f"✔️ {name} ({code}) 数据处理完成。")
        else:
            print(f"❌ {name} ({code}) 数据获取失败，请检查错误日志。")
        
        # 在每次API调用后稍微暂停一下，避免因请求过于频繁而被服务器限制
        time.sleep(1) 

    print("\n========================================")
    print("🎉 所有任务执行完毕！")
    print("========================================")

    # 3. 简单展示获取到的数据信息
    if all_data:
        print("\n已获取的数据摘要:")
        for code, df in all_data.items():
            stock_name = stocks_to_fetch[code]
            # 检查DataFrame是否不为空再访问数据
            if not df.empty:
                print(f"  - {stock_name} ({code}): {len(df)} 条数据, "
                      f"时间范围从 {df['date'].min().strftime('%Y-%m-%d')} 到 {df['date'].max().strftime('%Y-%m-%d')}")
            else:
                print(f"  - {stock_name} ({code}): 数据框为空。")
    else:
        print("本次运行未能获取到任何数据。")


