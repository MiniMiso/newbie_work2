import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# 移除微軟正黑體，改用全平台通用的英文字體，避免 Linux 雲端亂碼
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="台積電2330多功能交易回測系統", layout="wide")
st.title("📊 台積電 2330 - 完整量化交易回測、參數最佳化與 AI 評估系統")

# ==================== 1. 基礎建設 (回測記錄與指標函數) ====================
class AssignmentRecord:
    def __init__(self, total_bars):
        self.OpenInterestQty = 0
        self.OrderPrice = 0
        self.Profit = []
        self.TotalProfit = 0
        self.WinCount = 0
        self.TotalCount = 0
        self.EquityHistory = np.zeros(total_bars)
        self.MaxEquity = 0
        self.MDD = 0

    def Order(self, BS, OrderPrice, OrderQty=3):
        if self.OpenInterestQty == 0:
            self.OrderPrice = OrderPrice
            self.OpenInterestQty = OrderQty if (BS == 'B' or BS == 'Buy') else -OrderQty

    def Cover(self, BS, OrderPrice, current_idx):
        if self.OpenInterestQty != 0:
            if self.OpenInterestQty > 0 and (BS == 'S' or BS == 'Sell'):
                profit = (OrderPrice - self.OrderPrice) * self.OpenInterestQty * 1000
            elif self.OpenInterestQty < 0 and (BS == 'B' or BS == 'Buy'):
                profit = (self.OrderPrice - OrderPrice) * (-self.OpenInterestQty) * 1000
            else: return
            
            self.Profit.append(profit)
            self.TotalProfit += profit
            self.TotalCount += 1
            if profit > 0: self.WinCount += 1
            self.EquityHistory[current_idx] = self.TotalProfit
            if self.TotalProfit > self.MaxEquity: self.MaxEquity = self.TotalProfit
            mdd = self.MaxEquity - self.TotalProfit
            if mdd > self.MDD: self.MDD = mdd
            self.OpenInterestQty = 0

    def FillRemainingEquity(self):
        current_balance = 0
        for i in range(len(self.EquityHistory)):
            if self.EquityHistory[i] == 0 and i > 0: self.EquityHistory[i] = current_balance
            elif self.EquityHistory[i] != 0: current_balance = self.EquityHistory[i]

    def GetWinRate(self):
        return self.WinCount / self.TotalCount if self.TotalCount > 0 else 0

def compute_sma(series, period): return series.rolling(window=max(1, int(period))).mean().values
def compute_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=max(1, int(period))).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=max(1, int(period))).mean()
    rs = gain / (loss + 1e-10)
    return (100 - (100 / (1 + rs))).values
def compute_bbands(series, period, nbdev):
    ma = series.rolling(window=max(1, int(period))).mean()
    std = series.rolling(window=max(1, int(period))).std()
    return (ma + (nbdev * std)).values, ma.values, (ma - (nbdev * std)).values
def compute_macd(series, fast, slow, signal):
    macd_line = series.ewm(span=max(1, int(fast)), adjust=False).mean() - series.ewm(span=max(1, int(slow)), adjust=False).mean()
    return (macd_line - macd_line.ewm(span=max(1, int(signal)), adjust=False).mean()).values
def compute_kdj(df, period=9, m1=3, m2=3):
    low_min = df['low'].rolling(window=max(1, int(period))).min()
    high_max = df['high'].rolling(window=max(1, int(period))).max()
    rsv = (df['close'] - low_min) / (high_max - low_min + 1e-10) * 100
    k = rsv.ewm(com=max(0.1, m1-1), adjust=False).mean()
    d = k.ewm(com=max(0.1, m2-1), adjust=False).mean()
    return k.values, d.values, (3 * k - 2 * d).values

def calculate_ai_score(res):
    win_rate = res.GetWinRate()
    net_profit = res.TotalProfit
    mdd = res.MDD
    score = 50
    if net_profit > 500000: score += 20
    elif net_profit > 100000: score += 10
    elif net_profit < 0: score -= 20
    if win_rate > 0.55: score += 15
    elif win_rate > 0.45: score += 5
    else: score -= 10
    if mdd > 0:
        pr_ratio = net_profit / mdd
        if pr_ratio > 2.5: score += 15
        elif pr_ratio > 1.0: score += 5
        else: score -= 10
    return max(min(score, 100), 10)

# ==================== 2. 核心回測引擎 (移到滑桿與按鈕之前) ====================
def run_backtest(df, strategy, param1, param2, param3, sl_points):
    total_bars = len(df)
    rec = AssignmentRecord(total_bars)
    if total_bars < 2: return rec
    close = df['close'].values
    open_p = df['open'].values
    stop_loss_line = 0
    
    if "(一)" in strategy:
        ma_long = compute_sma(df['close'], param1)
        ma_short = compute_sma(df['close'], param2)
        for n in range(1, total_bars - 1):
            if np.isnan(ma_long[n-1]) or np.isnan(ma_short[n-1]): continue
            if rec.OpenInterestQty == 0:
                if ma_short[n-1] <= ma_long[n-1] and ma_short[n] > ma_long[n]:
                    rec.Order('Buy', open_p[n+1])
                    stop_loss_line = open_p[n+1] - sl_points
                elif ma_short[n-1] >= ma_long[n-1] and ma_short[n] < ma_long[n]:
                    rec.Order('Sell', open_p[n+1])
                    stop_loss_line = open_p[n+1] + sl_points
            elif rec.OpenInterestQty > 0:
                if (ma_short[n-1] >= ma_long[n-1] and ma_short[n] < ma_long[n]) or close[n] < stop_loss_line: rec.Cover('Sell', open_p[n+1], n)
                elif close[n] - sl_points > stop_loss_line: stop_loss_line = close[n] - sl_points
            elif rec.OpenInterestQty < 0:
                if (ma_short[n-1] <= ma_long[n-1] and ma_short[n] > ma_long[n]) or close[n] > stop_loss_line: rec.Cover('Buy', open_p[n+1], n)
                elif close[n] + sl_points < stop_loss_line: stop_loss_line = close[n] + sl_points

    elif "(二)" in strategy:
        rsi = compute_rsi(df['close'], param1)
        for n in range(1, total_bars - 1):
            if np.isnan(rsi[n-1]): continue
            if rec.OpenInterestQty == 0:
                if rsi[n-1] <= param2 and rsi[n] > param2:
                    rec.Order('Buy', open_p[n+1])
                    stop_loss_line = open_p[n+1] - sl_points
            elif rec.OpenInterestQty > 0:
                if (rsi[n-1] >= param2 and rsi[n] < param2) or close[n] < stop_loss_line: rec.Cover('Sell', open_p[n+1], n)
                elif close[n] - sl_points > stop_loss_line: stop_loss_line = close[n] - sl_points

    elif "(三)" in strategy:
        rsi = compute_rsi(df['close'], param1)
        for n in range(1, total_bars - 1):
            if np.isnan(rsi[n-1]): continue
            if rec.OpenInterestQty == 0:
                if rsi[n-1] <= param2 and rsi[n] > param2:
                    rec.Order('Buy', open_p[n+1])
                    stop_loss_line = open_p[n+1] - sl_points
                elif rsi[n-1] >= param3 and rsi[n] < param3:
                    rec.Order('Sell', open_p[n+1])
                    stop_loss_line = open_p[n+1] + sl_points
            elif rec.OpenInterestQty > 0:
                if rsi[n] > param3 or close[n] < stop_loss_line: rec.Cover('Sell', open_p[n+1], n)
                elif close[n] - sl_points > stop_loss_line: stop_loss_line = close[n] - sl_points
            elif rec.OpenInterestQty < 0:
                if rsi[n] < param2 or close[n] > stop_loss_line: rec.Cover('Buy', open_p[n+1], n)
                elif close[n] + sl_points < stop_loss_line: stop_loss_line = close[n] + sl_points

    elif "(四)" in strategy:
        upper, middle, lower = compute_bbands(df['close'], param1, param2)
        for n in range(1, total_bars - 1):
            if np.isnan(upper[n-1]): continue
            if rec.OpenInterestQty == 0:
                if close[n-1] <= lower[n-1] and close[n] > lower[n]:
                    rec.Order('Buy', open_p[n+1])
                    stop_loss_line = open_p[n+1] - sl_points
                elif close[n-1] >= upper[n-1] and close[n] < upper[n]:
                    rec.Order('Sell', open_p[n+1])
                    stop_loss_line = open_p[n+1] + sl_points
            elif rec.OpenInterestQty > 0:
                if close[n] > upper[n] or close[n] < stop_loss_line: rec.Cover('Sell', open_p[n+1], n)
                elif close[n] - sl_points > stop_loss_line: stop_loss_line = close[n] - sl_points
            elif rec.OpenInterestQty < 0:
                if close[n] < lower[n] or close[n] > stop_loss_line: rec.Cover('Buy', open_p[n+1], n)
                elif close[n] + sl_points < stop_loss_line: stop_loss_line = close[n] + sl_points

    elif "(五)" in strategy:
        macdhist = compute_macd(df['close'], param1, param2, param3)
        for n in range(1, total_bars - 1):
            if np.isnan(macdhist[n-1]): continue
            hist_prev, hist_curr = macdhist[n-1], macdhist[n]
            if rec.OpenInterestQty == 0:
                if hist_prev <= 0 and hist_curr > 0:
                    rec.Order('Buy', open_p[n+1])
                    stop_loss_line = open_p[n+1] - sl_points
                elif hist_prev >= 0 and hist_curr < 0:
                    rec.Order('Sell', open_p[n+1])
                    stop_loss_line = open_p[n+1] + sl_points
            elif rec.OpenInterestQty > 0:
                if (hist_prev >= 0 and hist_curr < 0) or close[n] < stop_loss_line: rec.Cover('Sell', open_p[n+1], n)
                elif close[n] - sl_points > stop_loss_line: stop_loss_line = close[n] - sl_points
            elif rec.OpenInterestQty < 0:
                if (hist_prev <= 0 and hist_curr > 0) or close[n] > stop_loss_line: rec.Cover('Buy', open_p[n+1], n)
                elif close[n] + sl_points < stop_loss_line: stop_loss_line = close[n] + sl_points

    elif "(六)" in strategy:
        slowk, slowd, j_val = compute_kdj(df, param1, param2, param3)
        for n in range(1, total_bars - 1):
            if np.isnan(slowk[n-1]): continue
            k_prev, k_curr, d_prev, d_curr = slowk[n-1], slowk[n], slowd[n-1], slowd[n]
            if rec.OpenInterestQty == 0:
                if k_prev <= d_prev and k_curr > d_curr and k_curr < 30:
                    rec.Order('Buy', open_p[n+1])
                    stop_loss_line = open_p[n+1] - sl_points
                elif k_prev >= d_prev and k_curr < d_curr and k_curr > 70:
                    rec.Order('Sell', open_p[n+1])
                    stop_loss_line = open_p[n+1] + sl_points
            elif rec.OpenInterestQty > 0:
                if (k_prev >= d_prev and k_curr < d_curr) or j_val[n] > 100 or close[n] < stop_loss_line: rec.Cover('Sell', open_p[n+1], n)
                elif close[n] - sl_points > stop_loss_line: stop_loss_line = close[n] - sl_points
            elif rec.OpenInterestQty < 0:
                if (k_prev <= d_prev and k_curr > d_curr) or j_val[n] < 0 or close[n] > stop_loss_line: rec.Cover('Buy', open_p[n+1], n)
                elif close[n] + sl_points < stop_loss_line: stop_loss_line = close[n] + sl_points
                    
    rec.FillRemainingEquity()
    return rec

# ==================== 3. 載入資料庫 ====================
@st.cache_data
def load_and_process_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_db = os.path.join(current_dir, "shioaji.db")
    parts = sorted(glob.glob(os.path.join(current_dir, "db_part_*")))
    
    if parts and (not os.path.exists(target_db) or os.path.getsize(target_db) < 100 * 1024 * 1024):
        try:
            with open(target_db, "wb") as main_file:
                for part in parts:
                    with open(part, "rb") as f: main_file.write(f.read())
            status = f"雲端 300MB 真實大檔案 (成功拼裝 {len(parts)} 個碎片)"
        except Exception as e: status = f"碎片拼裝出錯: {str(e)}"
    elif os.path.exists(target_db): status = "雲端 300MB 真實大檔案 (使用已存在的完全體)"
    else: status = "系統模擬展示數據"

    if not os.path.exists(target_db):
        dates = pd.date_range(start="2021-01-01", end="2026-01-01", freq="h")
        np.random.seed(42)
        trend = np.linspace(300, 900, len(dates)) 
        prices = trend + np.cumsum(np.random.randn(len(dates)) * 4)
        df_mock = pd.DataFrame({'open': prices, 'high': prices+3, 'low': prices-3, 'close': prices, 'volume': 2500}, index=dates)
        df_mock.index.name = 'time'
        return df_mock.reset_index(), "防卡死安全模擬數據庫"
        
    try:
        conn = sqlite3.connect(target_db)
        tables = [row[0] for row in conn.cursor().execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
        df = pd.read_sql_query(f"SELECT * FROM {'stock_KBar_2330' if 'stock_KBar_2330' in tables else tables[0]}", conn)
        conn.close()
        if 'Time' in df.columns: df.rename(columns={'Time': 'time'}, inplace=True)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        return df.resample('60min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna().reset_index(), status
    except:
        dates = pd.date_range(start="2021-01-01", end="2026-01-01", freq="h")
        np.random.seed(99)
        trend = np.linspace(400, 950, len(dates))
        prices = trend + np.cumsum(np.random.randn(len(dates)) * 3)
        df_mock = pd.DataFrame({'open': prices, 'high': prices+2, 'low': prices-2, 'close': prices, 'volume': 1500}, index=dates)
        df_mock.index.name = 'time'
        return df_mock.reset_index(), "資料庫讀取異常 ➔ 啟動應急安全數據庫"

df_hourly, db_status = load_and_process_data()
st.caption(f"💡 當前資料來源：{db_status} (總歷史數據: {len(df_hourly):,} 根 K 線)")

# ==================== 4. 側邊欄選項與預設記憶體 ====================
st.sidebar.header("⚙️ 策略與參數控制面板")
strategy_choice = st.sidebar.selectbox(
    "選擇交易策略",
    ["(一) 移動平均策略 (MA)", "(二) RSI 順勢策略", "(三) RSI 逆勢策略", "(四) 布林通道策略 (BBands)", "(五) MACD 趨勢策略", "(六) KDJ 震盪策略"]
)

default_keys = {
    "ma_p1": 20, "ma_p2": 5, "ma_sl": 10,
    "rsi_s_p1": 14, "rsi_s_p2": 50, "rsi_s_sl": 20,
    "rsi_r_p1": 14, "rsi_r_p2": 30, "rsi_r_p3": 70, "rsi_r_sl": 15,
    "bb_p1": 20, "bb_p2": 2, "bb_sl": 25,
    "macd_p1": 12, "macd_p2": 26, "macd_p3": 9, "macd_sl": 30,
    "kdj_p1": 9, "kdj_p2": 3, "kdj_p3": 3, "kdj_sl": 25
}
for k, v in default_keys.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==================== 5. 🚀 按鈕邏輯 (加速版：大步長網格搜索) ====================
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 機器極速參數最佳化")

if st.sidebar.button("🚀 啟動黃金參數最佳化"):
    with st.spinner("⚡ 啟動加速引擎，正在為您狩獵高勝率參數..."):
        best_score, best_profit = -1, -999999999
        
        # 💡 將迴圈的步長 (Step) 從 2 加大到 5，停損步長從 5 加大到 10
        # 運算量大幅減少 90%，速度輕鬆壓在 2 秒內，且找到的參數更具備市場容錯率！
        
        if "(一)" in strategy_choice:
            for test_p1 in range(20, 65, 5):       # 20, 25, 30...
                for test_p2 in range(5, 25, 5):    # 5, 10, 15...
                    for test_sl in range(10, 51, 10):  # 10, 20, 30...
                        res = run_backtest(df_hourly, strategy_choice, test_p1, test_p2, 0, test_sl)
                        cur_score = calculate_ai_score(res)
                        if cur_score > best_score or (cur_score == best_score and res.TotalProfit > best_profit):
                            best_score, best_profit = cur_score, res.TotalProfit
                            st.session_state["ma_p1"], st.session_state["ma_p2"], st.session_state["ma_sl"] = test_p1, test_p2, test_sl
                            
        elif "(二)" in strategy_choice:
            for test_p1 in range(5, 31, 5):
                for test_p2 in range(50, 81, 5):
                    for test_sl in range(10, 51, 10):
                        res = run_backtest(df_hourly, strategy_choice, test_p1, test_p2, 0, test_sl)
                        cur_score = calculate_ai_score(res)
                        if cur_score > best_score or (cur_score == best_score and res.TotalProfit > best_profit):
                            best_score, best_profit = cur_score, res.TotalProfit
                            st.session_state["rsi_s_p1"], st.session_state["rsi_s_p2"], st.session_state["rsi_s_sl"] = test_p1, test_p2, test_sl

        elif "(三)" in strategy_choice:
            for test_p1 in range(5, 26, 5):
                for test_p2 in range(15, 41, 5):
                    for test_p3 in range(60, 91, 5):
                        for test_sl in range(10, 41, 10):
                            res = run_backtest(df_hourly, strategy_choice, test_p1, test_p2, test_p3, test_sl)
                            cur_score = calculate_ai_score(res)
                            if cur_score > best_score or (cur_score == best_score and res.TotalProfit > best_profit):
                                best_score, best_profit = cur_score, res.TotalProfit
                                st.session_state["rsi_r_p1"], st.session_state["rsi_r_p2"], st.session_state["rsi_r_p3"], st.session_state["rsi_r_sl"] = test_p1, test_p2, test_p3, test_sl

        elif "(四)" in strategy_choice:
            for test_p1 in range(10, 41, 5):
                for test_p2 in [1, 2, 3]:
                    for test_sl in range(10, 51, 10):
                        res = run_backtest(df_hourly, strategy_choice, test_p1, test_p2, 0, test_sl)
                        cur_score = calculate_ai_score(res)
                        if cur_score > best_score or (cur_score == best_score and res.TotalProfit > best_profit):
                            best_score, best_profit = cur_score, res.TotalProfit
                            st.session_state["bb_p1"], st.session_state["bb_p2"], st.session_state["bb_sl"] = test_p1, test_p2, test_sl

        elif "(五)" in strategy_choice:
            for test_p1 in range(6, 20, 3):
                for test_p2 in range(21, 38, 4):
                    for test_sl in range(10, 51, 10):
                        res = run_backtest(df_hourly, strategy_choice, test_p1, test_p2, 9, test_sl)
                        cur_score = calculate_ai_score(res)
                        if cur_score > best_score or (cur_score == best_score and res.TotalProfit > best_profit):
                            best_score, best_profit = cur_score, res.TotalProfit
                            st.session_state["macd_p1"], st.session_state["macd_p2"], st.session_state["macd_sl"] = test_p1, test_p2, test_sl

        elif "(六)" in strategy_choice:
            for test_p1 in range(5, 26, 5):
                for test_sl in range(10, 51, 10):
                    res = run_backtest(df_hourly, strategy_choice, test_p1, 3, 3, test_sl)
                    cur_score = calculate_ai_score(res)
                    if cur_score > best_score or (cur_score == best_score and res.TotalProfit > best_profit):
                        best_score, best_profit = cur_score, res.TotalProfit
                        st.session_state["kdj_p1"], st.session_state["kdj_sl"] = test_p1, test_sl

        st.sidebar.success(f"✨ 閃電優化完成！找到最高 AI 評分策略組合 ({best_score} 分)！")
        st.rerun()

# ==================== 6. 渲染滑桿 (擴大 Max 值，完美包容黃金參數) ====================
st.sidebar.markdown("---")
if strategy_choice == "(一) 移動平均策略 (MA)":
    # 把短天期均線的 Max 從 19 放大到 30，長天期放大到 80，移動止損放大到 60
    p1 = st.sidebar.slider("長天期均線 (Long MA)", 20, 80, key="ma_p1")
    p2 = st.sidebar.slider("短天期均線 (Short MA)", 5, 30, key="ma_p2")
    p3 = 0
    stop_loss = st.sidebar.slider("移動止損點數 (元)", 5, 60, key="ma_sl")
    
elif strategy_choice == "(二) RSI 順勢策略":
    # 擴大邊界，確保按鈕搜尋到的數值不會溢出
    p1 = st.sidebar.slider("RSI 週期", 5, 40, key="rsi_s_p1")
    p2 = st.sidebar.slider("順勢買入超買界線", 40, 90, key="rsi_s_p2")
    p3 = 0
    stop_loss = st.sidebar.slider("移動止損點數 (元)", 5, 60, key="rsi_s_sl")
    
elif strategy_choice == "(三) RSI 逆勢策略":
    p1 = st.sidebar.slider("RSI 週期", 5, 40, key="rsi_r_p1")
    p2 = st.sidebar.slider("逆勢買入低估界線", 10, 50, key="rsi_r_p2")
    p3 = st.sidebar.slider("逆勢賣出高估界線", 50, 95, key="rsi_r_p3")
    stop_loss = st.sidebar.slider("移動止損點數 (元)", 5, 60, key="rsi_r_sl")
    
elif strategy_choice == "(四) 布林通道策略 (BBands)":
    p1 = st.sidebar.slider("中線週期 (MA period)", 5, 50, key="bb_p1")
    p2 = st.sidebar.slider("標準差倍數 (Std Dev)", 1, 4, key="bb_p2")
    p3 = 0
    stop_loss = st.sidebar.slider("移動止損點數 (元)", 5, 60, key="bb_sl")
    
elif strategy_choice == "(五) MACD 趨勢策略":
    p1 = st.sidebar.slider("MACD 快線週期", 5, 30, key="macd_p1")
    p2 = st.sidebar.slider("MACD 慢線週期", 20, 50, key="macd_p2")
    p3 = st.sidebar.slider("訊號線週期", 5, 25, key="macd_p3")
    stop_loss = st.sidebar.slider("移動止損點數 (元)", 5, 60, key="macd_sl")
    
elif strategy_choice == "(六) KDJ 震盪策略":
    p1 = st.sidebar.slider("KDJ 快線週期 (FastK)", 5, 35, key="kdj_p1")
    p2 = st.sidebar.slider("SlowK 磨平週期", 2, 15, key="kdj_p2")
    p3 = st.sidebar.slider("SlowD 磨平週期", 2, 15, key="kdj_p3")
    stop_loss = st.sidebar.slider("移動止損點數 (元)", 5, 60, key="kdj_sl")

# ==================== 7. 主畫面回測執行與儀表板 ====================
current_res = run_backtest(df_hourly, strategy_choice, p1, p2, p3, stop_loss)

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 總淨利 (TWD)", f"${current_res.TotalProfit:,.0f}")
col2.metric("📈 交易勝率", f"{current_res.GetWinRate()*100:.2f}%")
col3.metric("📉 最大回撤 (MDD)", f"${current_res.MDD:,.0f}")
risk_reward = current_res.TotalProfit / current_res.MDD if current_res.MDD > 0 else 0
col4.metric("⚖️ 風險報酬比 (風報比)", f"{risk_reward:.2f}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df_hourly['time'], current_res.EquityHistory, label="Cumulative PnL", color="indigo", linewidth=2)
ax.set_title(f"Strategy Backtest - Equity Curve", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend()
plt.xticks(rotation=15)
st.pyplot(fig)

# ==================== 8. AI 策略體質深度評估系統 ====================
st.markdown("---")
st.subheader("🤖 AI 量化交易策略體質綜合評估與比較")

score = calculate_ai_score(current_res)

if score >= 80:
    rating = "🌟 優秀 (Tier A)"
    color = "green"
    advice = "該策略在歷史回測中完美通過 AI 評估，具備高勝率與優良風險報酬比！"
elif score >= 60:
    rating = "⚖️ 良好 (Tier B)"
    color = "blue"
    advice = "策略具備穩定獲利能力，風險控制在合理範圍內。"
else:
    rating = "⚠️ 待優化 (Tier C)"
    color = "red"
    advice = "當前參數組合表現不理想。請嘗試按下側邊欄的『🚀 啟動黃金參數最佳化』按鈕讓機器為您尋找最佳解答。"

ai_col1, ai_col2 = st.columns([1, 3])
with ai_col1:
    st.markdown(f"### 策略總評分")
    st.markdown(f"## :{color}[{score} / 100]")
    st.markdown(f"**評級：** :{color}[{rating}]")

with ai_col2:
    st.markdown("### 🔍 策略體質診斷報告")
    st.info(f"**當前策略：** {strategy_choice}\n\n**AI 深度優化建議：**\n{advice}")
