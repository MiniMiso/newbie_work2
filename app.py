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

# ==================== 1. 核心回測引擎 ====================
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
            else:
                return
            
            self.Profit.append(profit)
            self.TotalProfit += profit
            self.TotalCount += 1
            if profit > 0:
                self.WinCount += 1
                
            self.EquityHistory[current_idx] = self.TotalProfit
            
            if self.TotalProfit > self.MaxEquity:
                self.MaxEquity = self.TotalProfit
            mdd = self.MaxEquity - self.TotalProfit
            if mdd > self.MDD:
                self.MDD = mdd
                
            self.OpenInterestQty = 0

    def FillRemainingEquity(self):
        current_balance = 0
        for i in range(len(self.EquityHistory)):
            if self.EquityHistory[i] == 0 and i > 0:
                self.EquityHistory[i] = current_balance
            elif self.EquityHistory[i] != 0:
                current_balance = self.EquityHistory[i]

    def GetWinRate(self):
        return self.WinCount / self.TotalCount if self.TotalCount > 0 else 0

# ==================== 2. 用 Pandas 純手寫還原技術指標 ====================
def compute_sma(series, period):
    return series.rolling(window=max(1, int(period))).mean().values

def compute_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=max(1, int(period))).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=max(1, int(period))).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def compute_bbands(series, period, nbdev):
    ma = series.rolling(window=max(1, int(period))).mean()
    std = series.rolling(window=max(1, int(period))).std()
    upper = ma + (nbdev * std)
    lower = ma - (nbdev * std)
    return upper.values, ma.values, lower.values

def compute_macd(series, fast, slow, signal):
    exp1 = series.ewm(span=max(1, int(fast)), adjust=False).mean()
    exp2 = series.ewm(span=max(1, int(slow)), adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=max(1, int(signal)), adjust=False).mean()
    hist = macd_line - signal_line
    return hist.values

def compute_kdj(df, period=9, m1=3, m2=3):
    low_min = df['low'].rolling(window=max(1, int(period))).min()
    high_max = df['high'].rolling(window=max(1, int(period))).max()
    rsv = (df['close'] - low_min) / (high_max - low_min + 1e-10) * 100
    k = rsv.ewm(com=max(0.1, m1-1), adjust=False).mean()
    d = k.ewm(com=max(0.1, m2-1), adjust=False).mean()
    j = 3 * k - 2 * d
    return k.values, d.values, j.values

# ==================== 3. 雲端自動碎片拼接與資料庫讀取 ====================
@st.cache_data
def load_and_process_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_db = os.path.join(current_dir, "shioaji.db")
    
    parts = sorted(glob.glob(os.path.join(current_dir, "db_part_*")))
    if not os.path.exists(target_db) and parts:
        with open(target_db, "wb") as main_file:
            for part in parts:
                with open(part, "rb") as f:
                    main_file.write(f.read())
                    
    if os.path.exists(target_db):
        db_path = target_db
        status = "雲端 300MB 真實大檔案 (碎片自動還原)"
    elif os.path.exists(r"C:\Users\Minir\OneDrive\桌面\量化交易期末報告\shioaji.db"):
        db_path = r"C:\Users\Minir\OneDrive\桌面\量化交易期末報告\shioaji.db"
        status = "本地 300MB 真實台積電完整歷史資料庫"
    else:
        st.warning("⚠️ 找不到任何資料庫或碎片，切換至系統模擬展示數據")
        dates = pd.date_range(start="2020-01-01", end="2025-01-01", freq="h")
        np.random.seed(42)
        prices = 300 + np.cumsum(np.random.randn(len(dates)) * 2)
        df_mock = pd.DataFrame({'open': prices, 'high': prices+2, 'low': prices-2, 'close': prices, 'volume': 1000}, index=dates)
        df_mock.index.name = 'time'
        return df_mock.reset_index(), "模擬環境"
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    target_table = "stock_KBar_2330" if "stock_KBar_2330" in tables else tables[0]
    
    df = pd.read_sql_query(f"SELECT * FROM {target_table}", conn)
    conn.close()
    
    if 'Time' in df.columns:
        df.rename(columns={'Time': 'time'}, inplace=True)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    df_hourly = df.resample('60min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna().reset_index()
    
    return df_hourly, status

df_hourly, db_status = load_and_process_data()
st.caption(f"💡 當前資料來源：{db_status} (總歷史數據: {len(df_hourly):,} 根 K 線)")

# ==================== 4. 側邊欄控制面板 ====================
st.sidebar.header("⚙️ 策略與參數控制面板")
strategy_choice = st.sidebar.selectbox(
    "選擇交易策略",
    ["(一) 移動平均策略 (MA)", "(二) RSI 順勢策略", "(三) RSI 逆勢策略", "(四) 布林通道策略 (BBands)", "(五) MACD 趨勢策略", "(六) KDJ 震盪策略"]
)

# 💡 初始化各策略的預設初始參數字典（確保不受殘留干擾）
default_presets = {
    "(一) 移動平均策略 (MA)": {"p1": 20, "p2": 5, "p3": 0, "sl": 10},
    "(二) RSI 順勢策略": {"p1": 14, "p2": 50, "p3": 0, "sl": 20},
    "(三) RSI 逆勢策略": {"p1": 14, "p2": 30, "p3": 70, "sl": 15},
    "(四) 布林通道策略 (BBands)": {"p1": 20, "p2": 2, "p3": 0, "sl": 25},
    "(五) MACD 趨勢策略": {"p1": 12, "p2": 26, "p3": 9, "sl": 30},
    "(六) KDJ 震盪策略": {"p1": 9, "p2": 3, "p3": 3, "sl": 25}
}

if "strat_params" not in st.session_state:
    st.session_state.strat_params = dict(default_presets)

# 當使用者「手動切換不同策略」時，自動洗掉上一檔策略殘留的暫存狀態
if "last_strategy" not in st.session_state:
    st.session_state.last_strategy = strategy_choice
elif st.session_state.last_strategy != strategy_choice:
    st.session_state.strat_params[strategy_choice] = dict(default_presets[strategy_choice])
    st.session_state.last_strategy = strategy_choice

current_params = st.session_state.strat_params[strategy_choice]

# 渲染滑桿組件
if strategy_choice == "(一) 移動平均策略 (MA)":
    p1 = st.sidebar.slider("長天期均線 (Long MA)", 20, 60, int(current_params["p1"]), key="ma_p1")
    p2 = st.sidebar.slider("短天期均線 (Short MA)", 5, 19, int(current_params["p2"]), key="ma_p2")
    p3 = 0
    stop_loss = st.sidebar.slider("移動止損點數 (元)", 5, 50, int(current_params["sl"]), key="ma_sl")
elif strategy_choice == "(二) RSI 順勢策略":
    p1 = st.sidebar.slider("RSI 週期", 5, 30, int(current_params["p1"]), key="rsi_s_p1")
    p2 = st.sidebar.slider("順勢買入超買界線", 50, 80, int(current_params["p2"]), key="rsi_s_p2")
    p3 = 0
    stop_loss = st.sidebar.slider("移動止損點數 (元)", 5, 50, int(current_params["sl"]), key="rsi_s_sl")
elif strategy_choice == "(三) RSI 逆勢策略":
    p1 = st.sidebar.slider("RSI 週期", 5, 30, int(current_params["p1"]), key="rsi_r_p1")
    p2 = st.sidebar.slider("逆勢買入低估界線", 10, 45, int(current_params["p2"]), key="rsi_r_p2")
    p3 = st.sidebar.slider("逆勢賣出高估界線", 55, 90, int(current_params["p3"]), key="rsi_r_p3")
    stop_loss = st.sidebar.slider("移動止損點數 (元)", 5, 50, int(current_params["sl"]), key="rsi_r_sl")
elif strategy_choice == "(四) 布林通道策略 (BBands)":
    p1 = st.sidebar.slider("中線週期 (MA period)", 5, 40, int(current_params["p1"]), key="bb_p1")
    p2 = st.sidebar.slider("標準差倍數 (Std Dev)", 1, 3, int(current_params["p2"]), key="bb_p2")
    p3 = 0
    stop_loss = st.sidebar.slider("移動止損點數 (元)", 5, 50, int(current_params["sl"]), key="bb_sl")
elif strategy_choice == "(五) MACD 趨勢策略":
    p1 = st.sidebar.slider("MACD 快線週期", 5, 20, int(current_params["p1"]), key="macd_p1")
    p2 = st.sidebar.slider("MACD 慢線週期", 21, 40, int(current_params["p2"]), key="macd_p2")
    p3 = st.sidebar.slider("訊號線週期", 5, 15, int(current_params["p3"]), key="macd_p3")
    stop_loss = st.sidebar.slider("移動止損點數 (元)", 5, 50, int(current_params["sl"]), key="macd_sl")
elif strategy_choice == "(六) KDJ 震盪策略":
    p1 = st.sidebar.slider("KDJ 快線週期 (FastK)", 5, 25, int(current_params["p1"]), key="kdj_p1")
    p2 = st.sidebar.slider("SlowK 磨平週期", 2, 10, int(current_params["p2"]), key="kdj_p2")
    p3 = st.sidebar.slider("SlowD 磨平週期", 2, 10, int(current_params["p3"]), key="kdj_p3")
    stop_loss = st.sidebar.slider("移動止損點數 (元)", 5, 50, int(current_params["sl"]), key="kdj_sl")

# 當前使用者拉動滑桿時，即時寫入暫存字典中
st.session_state.strat_params[strategy_choice]["p1"] = p1
st.session_state.strat_params[strategy_choice]["p2"] = p2
st.session_state.strat_params[strategy_choice]["p3"] = p3
st.session_state.strat_params[strategy_choice]["sl"] = stop_loss

# ==================== 5. 核心回測實作 ====================
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
                if (ma_short[n-1] >= ma_long[n-1] and ma_short[n] < ma_long[n]) or close[n] < stop_loss_line:
                    rec.Cover('Sell', open_p[n+1], n)
                elif close[n] - sl_points > stop_loss_line: stop_loss_line = close[n] - sl_points
            elif rec.OpenInterestQty < 0:
                if (ma_short[n-1] <= ma_long[n-1] and ma_short[n] > ma_long[n]) or close[n] > stop_loss_line:
                    rec.Cover('Buy', open_p[n+1], n)
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
                if (rsi[n-1] >= param2 and rsi[n] < param2) or close[n] < stop_loss_line:
                    rec.Cover('Sell', open_p[n+1], n)
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
                if rsi[n] > param3 or close[n] < stop_loss_line:
                    rec.Cover('Sell', open_p[n+1], n)
                elif close[n] - sl_points > stop_loss_line: stop_loss_line = close[n] - sl_points
            elif rec.OpenInterestQty < 0:
                if rsi[n] < param2 or close[n] > stop_loss_line:
                    rec.Cover('Buy', open_p[n+1], n)
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
                if close[n] > upper[n] or close[n] < stop_loss_line:
                    rec.Cover('Sell', open_p[n+1], n)
                elif close[n] - sl_points > stop_loss_line: stop_loss_line = close[n] - sl_points
            elif rec.OpenInterestQty < 0:
                if close[n] < lower[n] or close[n] > stop_loss_line:
                    rec.Cover('Buy', open_p[n+1], n)
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
                if (hist_prev >= 0 and hist_curr < 0) or close[n] < stop_loss_line:
                    rec.Cover('Sell', open_p[n+1], n)
                elif close[n] - sl_points > stop_loss_line: stop_loss_line = close[n] - sl_points
            elif rec.OpenInterestQty < 0:
                if (hist_prev <= 0 and hist_curr > 0) or close[n] > stop_loss_line:
                    rec.Cover('Buy', open_p[n+1], n)
                elif close[n] + sl_points < stop_loss_line: stop_loss_line = close[n] + sl_points

    elif "(六)" in strategy:
        slowk, slowd, j_val = compute_kdj(df, param1, param2, param3)
        for n in range(1, total_bars - 1):
            if np.isnan(slowk[n-1]): continue
            k_prev, k_curr = slowk[n-1], slowk[n]
            d_prev, d_curr = slowd[n-1], slowd[n]
            if rec.OpenInterestQty == 0:
                if k_prev <= d_prev and k_curr > d_curr and k_curr < 30:
                    rec.Order('Buy', open_p[n+1])
                    stop_loss_line = open_p[n+1] - sl_points
                elif k_prev >= d_prev and k_curr < d_curr and k_curr > 70:
                    rec.Order('Sell', open_p[n+1])
                    stop_loss_line = open_p[n+1] + sl_points
            elif rec.OpenInterestQty > 0:
                if (k_prev >= d_prev and k_curr < d_curr) or j_val[n] > 100 or close[n] < stop_loss_line:
                    rec.Cover('Sell', open_p[n+1], n)
                elif close[n] - sl_points > stop_loss_line: stop_loss_line = close[n] - sl_points
            elif rec.OpenInterestQty < 0:
                if (k_prev <= d_prev and k_curr > d_curr) or j_val[n] < 0 or close[n] > stop_loss_line:
                    rec.Cover('Buy', open_p[n+1], n)
                elif close[n] + sl_points < stop_loss_line: stop_loss_line = close[n] + sl_points
                    
    rec.FillRemainingEquity()
    return rec

# ==================== 6. 黃金參數最佳化計算觸發 ====================
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 機器極速參數最佳化")
st.sidebar.caption("同時計算『風險與報酬』，尋找最高風險報酬比的參數組合")

if st.sidebar.button("🚀 啟動黃金參數最佳化"):
    with st.spinner("🤖 AI 正在高速回溯參數組合，尋找最佳風報比..."):
        # 💡 解法核心一：強制清空快取，確保最佳化迴圈拿到的是原汁原味的資料
        st.cache_data.clear()
        
        best_ratio = -999
        best_p1, best_p2, best_p3, best_sl = p1, p2, p3, stop_loss
        
        if "(一)" in strategy_choice:
            for test_p1 in range(20, 61, 10):
                for test_p2 in range(5, 20, 5):
                    for test_sl in range(10, 41, 10):
                        res = run_backtest(df_hourly, strategy_choice, test_p1, test_p2, 0, test_sl)
                        ratio = res.TotalProfit / res.MDD if res.MDD > 0 else 0
                        if ratio > best_ratio and res.TotalProfit > 0:
                            best_ratio, best_p1, best_p2, best_sl = ratio, test_p1, test_p2, test_sl
            best_p3 = 0
            
        elif "(二)" in strategy_choice:
            for test_p1 in range(6, 25, 4):
                for test_p2 in range(50, 76, 10):
                    for test_sl in range(10, 41, 10):
                        res = run_backtest(df_hourly, strategy_choice, test_p1, test_p2, 0, test_sl)
                        ratio = res.TotalProfit / res.MDD if res.MDD > 0 else 0
                        if ratio > best_ratio and res.TotalProfit > 0:
                            best_ratio, best_p1, best_p2, best_sl = ratio, test_p1, test_p2, test_sl
            best_p3 = 0

        elif "(三)" in strategy_choice:
            for test_p1 in range(10, 21, 5):
                for test_p2 in range(20, 41, 10):
                    for test_p3 in range(65, 86, 10):
                        for test_sl in range(15, 36, 10):
                            res = run_backtest(df_hourly, strategy_choice, test_p1, test_p2, test_p3, test_sl)
                            ratio = res.TotalProfit / res.MDD if res.MDD > 0 else 0
                            if ratio > best_ratio and res.TotalProfit > 0:
                                best_ratio, best_p1, best_p2, best_p3, best_sl = ratio, test_p1, test_p2, test_p3, test_sl

        elif "(四)" in strategy_choice:
            for test_p1 in range(10, 31, 10):
                for test_p2 in [1, 2, 3]:
                    for test_sl in range(10, 41, 10):
                        res = run_backtest(df_hourly, strategy_choice, test_p1, test_p2, 0, test_sl)
                        ratio = res.TotalProfit / res.MDD if res.MDD > 0 else 0
                        if ratio > best_ratio and res.TotalProfit > 0:
                            best_ratio, best_p1, best_p2, best_sl = ratio, test_p1, test_p2, test_sl
            best_p3 = 0

        elif "(五)" in strategy_choice:
            for test_p1 in range(8, 17, 4):
                for test_p2 in range(22, 35, 6):
                    for test_sl in range(20, 41, 10):
                        res = run_backtest(df_hourly, strategy_choice, test_p1, test_p2, 9, test_sl)
                        ratio = res.TotalProfit / res.MDD if res.MDD > 0 else 0
                        if ratio > best_ratio and res.TotalProfit > 0:
                            best_ratio, best_p1, best_p2, best_sl = ratio, test_p1, test_p2, test_sl
            best_p3 = 9

        elif "(六)" in strategy_choice:
            for test_p1 in range(9, 19, 5):
                for test_sl in range(15, 36, 10):
                    res = run_backtest(df_hourly, strategy_choice, test_p1, 3, 3, test_sl)
                    ratio = res.TotalProfit / res.MDD if res.MDD > 0 else 0
                    if ratio > best_ratio and res.TotalProfit > 0:
                        best_ratio, best_p1, best_sl = ratio, test_p1, test_sl
            best_p2, best_p3 = 3, 3

        # 💡 解法核心二：將最優解覆蓋進暫存字典，強制洗掉使用者剛剛亂拉的橫桿狀態
        st.session_state.strat_params[strategy_choice] = {
            "p1": best_p1, "p2": best_p2, "p3": best_p3, "sl": best_sl
        }

        st.sidebar.success(f"✨ 最佳化完成！最高風報比：{best_ratio:.2f}")
        st.rerun()

# 最終計算當前滑桿參數的回測結果
激活動態結果 = run_backtest(df_hourly, strategy_choice, p1, p2, p3, stop_loss)

# ==================== 7. 數據呈現儀表板 ====================
col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 總淨利 (TWD)", f"${激活動態結果.TotalProfit:,.0f}")
col2.metric("📈 交易勝率", f"{激活動態結果.GetWinRate()*100:.2f}%")
col3.metric("📉 最大回撤 (MDD)", f"${激活動態結果.MDD:,.0f}")
risk_reward = 激活動態結果.TotalProfit / 激活動態結果.MDD if 激活動態結果.MDD > 0 else 0
col4.metric("⚖️ 風險報酬比 (風報比)", f"{risk_reward:.2f}")

# 繪製主圖表
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df_hourly['time'], 激活動態結果.EquityHistory, label="Cumulative PnL", color="indigo", linewidth=2)
ax.set_title(f"Strategy Backtest - Equity Curve", fontsize=12)
ax.set_xlabel("Timeline")
ax.set_ylabel("Profit / Loss (TWD)")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend()
plt.xticks(rotation=15)
st.pyplot(fig)

# ==================== 8. AI 策略體質深度評估系統 ====================
st.markdown("---")
st.subheader("🤖 AI 量化交易策略體質綜合評估與比較")

win_rate = 激活動態結果.GetWinRate()
net_profit = 激活動態結果.TotalProfit
mdd = 激活動態結果.MDD

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
score = max(min(score, 100), 10)

if score >= 80:
    rating = "🌟 優秀 (Tier A)"
    color = "green"
    advice = "該策略在台積電歷史單邊趨勢中展現極強的獲利爆發力，風險報酬比健康。建議可在實際交易中作為核心策略，但需注意在未來市場進入極端盤整時的潛在假突破風險。"
elif score >= 60:
    rating = "⚖️ 良好 (Tier B)"
    color = "blue"
    advice = "策略具備基本的獲利能力，且勝率維持在合理範圍。然而最大回撤（MDD）偏高，說明在行情反轉時移動止損的回控速度不夠敏銳。建議進一步優化止損點數，或加入濾網以汰除震盪盤整行情。"
else:
    rating = "⚠️ 待優化 (Tier C)"
    color = "red"
    advice = "當前參數在台積電五年歷史中表現掙扎，高昂的 MDD 侵蝕了大幅度的利潤，勝率偏低。這主要是因為台積電在非 AI 爆發期有較長的時間處於箱型震盪，導致趨勢策略被反覆雙巴止損。強烈建議重新調整長短均線跨度，或改採逆勢通道策略。"

ai_col1, ai_col2 = st.columns([1, 3])
with ai_col1:
    st.markdown(f"### 策略總評分")
    st.markdown(f"## :{color}[{score} / 100]")
    st.markdown(f"**評級：** :{color}[{rating}]")

with ai_col2:
    st.markdown("### 🔍 策略體質診斷報告")
    st.info(f"**當前策略：** {strategy_choice}\n\n**AI 深度優化建議：**\n{advice}")
    
    st.markdown("**📊 策略多維度診斷指標：**")
    st.text(f"  - Trend Tracking: {'★' * min(5, int(score/18))} {'☆' * (5 - min(5, int(score/18)))}")
    st.text(f"  - Drawdown Control: {'★' * min(5, int(10 - mdd/100000 if mdd < 500000 else 2))} {'☆' * (5 - min(5, int(10 - mdd/100000 if mdd < 500000 else 2)))}")
    st.text(f"  - Signal Stability: {'★' * min(5, int(win_rate*8))} {'☆' * (5 - min(5, int(win_rate*8)))}")
