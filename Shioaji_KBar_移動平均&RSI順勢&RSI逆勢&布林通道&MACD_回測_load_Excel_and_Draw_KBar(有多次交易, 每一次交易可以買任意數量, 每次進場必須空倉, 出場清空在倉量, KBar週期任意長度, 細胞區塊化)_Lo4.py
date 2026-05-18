# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 00:14:26 2026

@author: user
"""

#%%
###### 載入必要模組
from order_Lo8 import Record
import numpy as np
from talib.abstract import SMA, EMA, WMA, RSI, BBANDS, MACD, STOCH
import datetime, indicator
import pandas as pd


#%%
###### 資料讀入與前處理
df = pd.read_excel("kbars_2330_2022-01-01-2024-04-09.xlsx", index_col=0)
df.columns
df.head()


#%%
###### 畫 KBar 圖
df.set_index("time", inplace=True)
import mplfinance as mpf
mpf.plot(df, volume=True, addplot=[], type='candle', style='charles')
df['time'] = df.index


#%%
###### 轉化為字典
KBar_dic = df.to_dict()

KBar_open_list = list(KBar_dic['open'].values())
KBar_dic['open'] = np.array(KBar_open_list).astype(np.float64)

KBar_dic['product'] = np.repeat('tsmc', KBar_dic['open'].size)

KBar_time_list = list(KBar_dic['time'].values())
KBar_time_list = [i.to_pydatetime() for i in KBar_time_list]
KBar_dic['time'] = np.array(KBar_time_list)

KBar_low_list = list(KBar_dic['low'].values())
KBar_dic['low'] = np.array(KBar_low_list).astype(np.float64)

KBar_high_list = list(KBar_dic['high'].values())
KBar_dic['high'] = np.array(KBar_high_list).astype(np.float64)

KBar_close_list = list(KBar_dic['close'].values())
KBar_dic['close'] = np.array(KBar_close_list).astype(np.float64)

KBar_volume_list = list(KBar_dic['volume'].values())
KBar_dic['volume'] = np.array(KBar_volume_list)

KBar_amount_list = list(KBar_dic['amount'].values())
KBar_dic['amount'] = np.array(KBar_amount_list)


#%%
######  改變 KBar 時間長度
Date = '20220101'
KBar = indicator.KBar(Date, 2880)  ## 2880分鐘=2天

for i in range(KBar_dic['time'].size):
    time = KBar_dic['time'][i]
    price = KBar_dic['close'][i]
    qty = KBar_dic['volume'][i]
    amount = KBar_dic['amount'][i]
    tag = KBar.AddPrice(time, price, qty)

    if tag != 1:
        continue


#%%
###### 形成變換長度後的 KBar 字典
KBar_dic = {}
KBar_dic['time'] = KBar.TAKBar['time']
KBar_dic['product'] = np.repeat('tsmc', KBar_dic['time'].size)
KBar_dic['open'] = KBar.TAKBar['open']
KBar_dic['high'] = KBar.TAKBar['high']
KBar_dic['low'] = KBar.TAKBar['low']
KBar_dic['close'] = KBar.TAKBar['close']
KBar_dic['volume'] = KBar.TAKBar['volume']


#%%
###### 定義繪製相關圖形之函數
def KbarToDf(KBar_dic):
    Kbar_df = pd.DataFrame(KBar_dic)
    Kbar_df.columns = [i[0].upper() + i[1:] for i in Kbar_df.columns]
    Kbar_df.set_index("Time", inplace=True)
    return Kbar_df


def ChartKBar(KBar_dic, addp=None, volume_enable=True):
    if addp is None:
        addp = []
    Kbar_df = KbarToDf(KBar_dic)
    mpf.plot(Kbar_df, volume=volume_enable, addplot=addp, type='candle', style='charles')


def ChartOrder(KBar_dic, TR, addp=None, volume_enable=True):
    if addp is None:
        addp = []

    Kbar_df = KbarToDf(KBar_dic)

    # 買(多)方下單點位紀錄
    BTR = [i for i in TR if i[0] == 'Buy' or i[0] == 'B']
    BuyOrderPoint = []
    BuyCoverPoint = []

    for date, value in Kbar_df['Close'].items():
        if date in [i[2] for i in BTR]:
            BuyOrderPoint.append(Kbar_df['Low'][date] * 0.999)
        else:
            BuyOrderPoint.append(np.nan)

        if date in [i[4] for i in BTR]:
            BuyCoverPoint.append(Kbar_df['High'][date] * 1.001)
        else:
            BuyCoverPoint.append(np.nan)

    if [i for i in BuyOrderPoint if not np.isnan(i)] != []:
        addp.append(mpf.make_addplot(BuyOrderPoint, scatter=True, markersize=50, marker='^', color='red'))
        addp.append(mpf.make_addplot(BuyCoverPoint, scatter=True, markersize=50, marker='v', color='blue'))

    # 賣(空)方下單點位紀錄
    STR = [i for i in TR if i[0] == 'Sell' or i[0] == 'S']
    SellOrderPoint = []
    SellCoverPoint = []

    for date, value in Kbar_df['Close'].items():
        if date in [i[2] for i in STR]:
            SellOrderPoint.append(Kbar_df['High'][date] * 1.001)
        else:
            SellOrderPoint.append(np.nan)

        if date in [i[4] for i in STR]:
            SellCoverPoint.append(Kbar_df['Low'][date] * 0.999)
        else:
            SellCoverPoint.append(np.nan)

    if [i for i in SellOrderPoint if not np.isnan(i)] != []:
        addp.append(mpf.make_addplot(SellOrderPoint, scatter=True, markersize=50, marker='v', color='green'))
        addp.append(mpf.make_addplot(SellCoverPoint, scatter=True, markersize=50, marker='^', color='pink'))

    ChartKBar(KBar_dic, addp, volume_enable)


def ChartOrder_MA(KBar_dic, TR):
    Kbar_df = KbarToDf(KBar_dic)
    addp = []
    addp.append(mpf.make_addplot(Kbar_df['MA_long'], color='red'))
    addp.append(mpf.make_addplot(Kbar_df['MA_short'], color='yellow'))
    ChartOrder(KBar_dic, TR, addp)


#%%
######  (一) 移動平均線策略

OrderRecord = Record()

LongMAPeriod = 10
ShortMAPeriod = 2
MoveStopLoss = 10

KBar_dic['MA_long'] = SMA(KBar_dic, timeperiod=LongMAPeriod)
KBar_dic['MA_short'] = SMA(KBar_dic, timeperiod=ShortMAPeriod)

Order_Quantity = 3

for n in range(0, len(KBar_dic['time']) - 1):
    if not np.isnan(KBar_dic['MA_long'][n - 1]):

        if OrderRecord.GetOpenInterest() == 0:
            if KBar_dic['MA_short'][n - 1] <= KBar_dic['MA_long'][n - 1] and KBar_dic['MA_short'][n] > KBar_dic['MA_long'][n]:
                OrderRecord.Order('Buy', KBar_dic['product'][n + 1], KBar_dic['time'][n + 1], KBar_dic['open'][n + 1], Order_Quantity)
                OrderPrice = KBar_dic['open'][n + 1]
                StopLossPoint = OrderPrice - MoveStopLoss
                continue

            if KBar_dic['MA_short'][n - 1] >= KBar_dic['MA_long'][n - 1] and KBar_dic['MA_short'][n] < KBar_dic['MA_long'][n]:
                OrderRecord.Order('Sell', KBar_dic['product'][n + 1], KBar_dic['time'][n + 1], KBar_dic['open'][n + 1], Order_Quantity)
                OrderPrice = KBar_dic['open'][n + 1]
                StopLossPoint = OrderPrice + MoveStopLoss
                continue

        elif OrderRecord.GetOpenInterest() > 0:
            if KBar_dic['product'][n + 1] != KBar_dic['product'][n]:
                OrderRecord.Cover('Sell', KBar_dic['product'][n], KBar_dic['time'][n], KBar_dic['close'][n], OrderRecord.GetOpenInterest())
                continue

            if KBar_dic['close'][n] - MoveStopLoss > StopLossPoint:
                StopLossPoint = KBar_dic['close'][n] - MoveStopLoss
            elif KBar_dic['close'][n] < StopLossPoint:
                OrderRecord.Cover('Sell', KBar_dic['product'][n + 1], KBar_dic['time'][n + 1], KBar_dic['open'][n + 1], OrderRecord.GetOpenInterest())
                continue

        elif OrderRecord.GetOpenInterest() < 0:
            if KBar_dic['product'][n + 1] != KBar_dic['product'][n]:
                OrderRecord.Cover('Buy', KBar_dic['product'][n], KBar_dic['time'][n], KBar_dic['close'][n], -OrderRecord.GetOpenInterest())
                continue

            if KBar_dic['close'][n] + MoveStopLoss < StopLossPoint:
                StopLossPoint = KBar_dic['close'][n] + MoveStopLoss
            elif KBar_dic['close'][n] > StopLossPoint:
                OrderRecord.Cover('Buy', KBar_dic['product'][n + 1], KBar_dic['time'][n + 1], KBar_dic['open'][n + 1], -OrderRecord.GetOpenInterest())
                continue

ChartOrder_MA(KBar_dic, OrderRecord.GetTradeRecord())
KBar_dic.keys()

print('交易紀錄: \n', OrderRecord.GetTradeRecord())
print('\n')
print('利潤清單(TWD,點數): \n',OrderRecord.GetProfit())
print('\n')
print('淨利(TWD,點數): ',OrderRecord.GetTotalProfit())
print('\n')
print('勝率: ',OrderRecord.GetWinRate())
print('\n')
print('最大連續虧損(TWD,點數): ',OrderRecord.GetAccLoss())
print('\n')
print('最大累計盈虧(TWD,點數)回落: ',OrderRecord.GetMDD())
print('\n')


#%%
###### (二) RSI 順勢策略

OrderRecord = Record()

LongRSIPeriod = 10
ShortRSIPeriod = 5
MoveStopLoss = 30
Order_Quantity = 3

KBar_dic['RSI_long'] = RSI(KBar_dic, timeperiod=LongRSIPeriod)
KBar_dic['RSI_short'] = RSI(KBar_dic, timeperiod=ShortRSIPeriod)
KBar_dic['Middle'] = np.array([50] * len(KBar_dic['time']))

for n in range(1, len(KBar_dic['time']) - 1):
    if not np.isnan(KBar_dic['RSI_long'][n - 1]):
        if OrderRecord.GetOpenInterest() == 0:
            if KBar_dic['RSI_short'][n - 1] <= KBar_dic['RSI_long'][n - 1] and KBar_dic['RSI_short'][n] > KBar_dic['RSI_long'][n] and KBar_dic['RSI_long'][n] > KBar_dic['Middle'][n]:
                OrderRecord.Order('Buy', KBar_dic['product'][n + 1], KBar_dic['time'][n + 1], KBar_dic['open'][n + 1], Order_Quantity)
                OrderPrice = KBar_dic['open'][n + 1]
                StopLossPoint = OrderPrice - MoveStopLoss
                continue

            if KBar_dic['RSI_short'][n - 1] >= KBar_dic['RSI_long'][n - 1] and KBar_dic['RSI_short'][n] < KBar_dic['RSI_long'][n] and KBar_dic['RSI_long'][n] < KBar_dic['Middle'][n]:
                OrderRecord.Order('Sell', KBar_dic['product'][n + 1], KBar_dic['time'][n + 1], KBar_dic['open'][n + 1], Order_Quantity)
                OrderPrice = KBar_dic['open'][n + 1]
                StopLossPoint = OrderPrice + MoveStopLoss
                continue

        elif OrderRecord.GetOpenInterest() > 0:
            if KBar_dic['product'][n + 1] != KBar_dic['product'][n]:
                OrderRecord.Cover('Sell', KBar_dic['product'][n], KBar_dic['time'][n], KBar_dic['close'][n], OrderRecord.GetOpenInterest())
                continue

            if KBar_dic['close'][n] - MoveStopLoss > StopLossPoint:
                StopLossPoint = KBar_dic['close'][n] - MoveStopLoss
            elif KBar_dic['close'][n] < StopLossPoint:
                OrderRecord.Cover('Sell', KBar_dic['product'][n + 1], KBar_dic['time'][n + 1], KBar_dic['open'][n + 1], OrderRecord.GetOpenInterest())
                continue

        elif OrderRecord.GetOpenInterest() < 0:
            if KBar_dic['product'][n + 1] != KBar_dic['product'][n]:
                OrderRecord.Cover('Buy', KBar_dic['product'][n], KBar_dic['time'][n], KBar_dic['close'][n], -OrderRecord.GetOpenInterest())
                continue

            if KBar_dic['close'][n] + MoveStopLoss < StopLossPoint:
                StopLossPoint = KBar_dic['close'][n] + MoveStopLoss
            elif KBar_dic['close'][n] > StopLossPoint:
                OrderRecord.Cover('Buy', KBar_dic['product'][n + 1], KBar_dic['time'][n + 1], KBar_dic['open'][n + 1], -OrderRecord.GetOpenInterest())
                continue

# OrderRecord.GetTradeRecord()
# OrderRecord.GetProfit()
# OrderRecord.GetTotalProfit()
# OrderRecord.GetWinRate()
# OrderRecord.GetAccLoss()
# OrderRecord.GetMDD()
print('交易紀錄: \n', OrderRecord.GetTradeRecord())
print('\n')
print('利潤清單(TWD,點數): \n',OrderRecord.GetProfit())
print('\n')
print('淨利(TWD,點數): ',OrderRecord.GetTotalProfit())
print('\n')
print('勝率: ',OrderRecord.GetWinRate())
print('\n')
print('最大連續虧損(TWD,點數): ',OrderRecord.GetAccLoss())
print('\n')
print('最大累計盈虧(TWD,點數)回落: ',OrderRecord.GetMDD())
print('\n')
OrderRecord.GeneratorProfitChart(StrategyName='RSI-long_short_cross')


#%%
###### (三) RSI 逆勢策略

OrderRecord = Record()
RSIPeriod = 5
Ceil = 80
Floor = 20
MoveStopLoss = 30
Order_Quantity = 3

KBar_dic['RSI'] = RSI(KBar_dic, timeperiod=RSIPeriod)
KBar_dic['Ceil'] = np.array([Ceil] * len(KBar_dic['time']))
KBar_dic['Floor'] = np.array([Floor] * len(KBar_dic['time']))

for n in range(1, len(KBar_dic['time']) - 1):
    if not np.isnan(KBar_dic['RSI'][n - 1]):
        if OrderRecord.GetOpenInterest() == 0:
            if KBar_dic['RSI'][n - 1] <= KBar_dic['Floor'][n - 1] and KBar_dic['RSI'][n] > KBar_dic['Floor'][n]:
                OrderRecord.Order('Buy', KBar_dic['product'][n + 1], KBar_dic['time'][n + 1], KBar_dic['open'][n + 1], Order_Quantity)
                OrderPrice = KBar_dic['open'][n + 1]
                StopLossPoint = OrderPrice - MoveStopLoss
                continue

            if KBar_dic['RSI'][n - 1] >= KBar_dic['Ceil'][n - 1] and KBar_dic['RSI'][n] < KBar_dic['Ceil'][n]:
                OrderRecord.Order('Sell', KBar_dic['product'][n + 1], KBar_dic['time'][n + 1], KBar_dic['open'][n + 1], Order_Quantity)
                OrderPrice = KBar_dic['open'][n + 1]
                StopLossPoint = OrderPrice + MoveStopLoss
                continue

        elif OrderRecord.GetOpenInterest() > 0:
            if KBar_dic['product'][n + 1] != KBar_dic['product'][n]:
                OrderRecord.Cover('Sell', KBar_dic['product'][n], KBar_dic['time'][n], KBar_dic['close'][n], OrderRecord.GetOpenInterest())
                continue

            if KBar_dic['close'][n] - MoveStopLoss > StopLossPoint:
                StopLossPoint = KBar_dic['close'][n] - MoveStopLoss
            elif KBar_dic['close'][n] < StopLossPoint:
                OrderRecord.Cover('Sell', KBar_dic['product'][n + 1], KBar_dic['time'][n + 1], KBar_dic['open'][n + 1], OrderRecord.GetOpenInterest())
                continue

            if KBar_dic['RSI'][n] > KBar_dic['Ceil'][n]:
                OrderRecord.Cover('Sell', KBar_dic['product'][n + 1], KBar_dic['time'][n + 1], KBar_dic['open'][n + 1], OrderRecord.GetOpenInterest())
                continue

        elif OrderRecord.GetOpenInterest() < 0:
            if KBar_dic['product'][n + 1] != KBar_dic['product'][n]:
                OrderRecord.Cover('Buy', KBar_dic['product'][n], KBar_dic['time'][n], KBar_dic['close'][n], -OrderRecord.GetOpenInterest())
                continue

            if KBar_dic['close'][n] + MoveStopLoss < StopLossPoint:
                StopLossPoint = KBar_dic['close'][n] + MoveStopLoss
            elif KBar_dic['close'][n] > StopLossPoint:
                OrderRecord.Cover('Buy', KBar_dic['product'][n + 1], KBar_dic['time'][n + 1], KBar_dic['open'][n + 1], -OrderRecord.GetOpenInterest())
                continue

            if KBar_dic['RSI'][n] < KBar_dic['Floor'][n]:
                OrderRecord.Cover('Buy', KBar_dic['product'][n + 1], KBar_dic['time'][n + 1], KBar_dic['open'][n + 1], -OrderRecord.GetOpenInterest())
                continue

# OrderRecord.GetTradeRecord()
# OrderRecord.GetProfit()
# OrderRecord.GetTotalProfit()
# OrderRecord.GetWinRate()
# OrderRecord.GetAccLoss()
# OrderRecord.GetMDD()
print('交易紀錄: \n', OrderRecord.GetTradeRecord())
print('\n')
print('利潤清單(TWD,點數): \n',OrderRecord.GetProfit())
print('\n')
print('淨利(TWD,點數): ',OrderRecord.GetTotalProfit())
print('\n')
print('勝率: ',OrderRecord.GetWinRate())
print('\n')
print('最大連續虧損(TWD,點數): ',OrderRecord.GetAccLoss())
print('\n')
print('最大累計盈虧(TWD,點數)回落: ',OrderRecord.GetMDD())
print('\n')
OrderRecord.GeneratorProfitChart(StrategyName='RSI_reversal')


#%%
###### (四) 布林通道策略

OrderRecord = Record()
BBANDSPeriod = 60
MoveStopLoss = 30
標準差倍數_上 = 2.0
標準差倍數_下 = 2.0
Order_Quantity = 1

KBar_dic['Upper'], KBar_dic['Middle'], KBar_dic['Lower'] = BBANDS(
    KBar_dic,
    timeperiod=BBANDSPeriod,
    nbdevup=標準差倍數_上,
    nbdevdn=標準差倍數_下,
    matype=0
)

for n in range(1, len(KBar_dic['time']) - 1):
    if not np.isnan(KBar_dic['Middle'][n - 1]):
        if OrderRecord.GetOpenInterest() == 0:
            if KBar_dic['close'][n - 1] <= KBar_dic['Lower'][n - 1] and KBar_dic['close'][n] > KBar_dic['Lower'][n]:
                OrderRecord.Order('Buy', KBar_dic['product'][n + 1], KBar_dic['time'][n + 1], KBar_dic['open'][n + 1], Order_Quantity)
                OrderPrice = KBar_dic['open'][n + 1]
                StopLossPoint = OrderPrice - MoveStopLoss
                continue

            if KBar_dic['close'][n - 1] >= KBar_dic['Upper'][n - 1] and KBar_dic['close'][n] < KBar_dic['Upper'][n]:
                OrderRecord.Order('Sell', KBar_dic['product'][n + 1], KBar_dic['time'][n + 1], KBar_dic['open'][n + 1], Order_Quantity)
                OrderPrice = KBar_dic['open'][n + 1]
                StopLossPoint = OrderPrice + MoveStopLoss
                continue

        elif OrderRecord.GetOpenInterest() > 0:
            if KBar_dic['product'][n + 1] != KBar_dic['product'][n]:
                OrderRecord.Cover('Sell', KBar_dic['product'][n], KBar_dic['time'][n], KBar_dic['close'][n], OrderRecord.GetOpenInterest())
                continue

            if KBar_dic['close'][n] - MoveStopLoss > StopLossPoint:
                StopLossPoint = KBar_dic['close'][n] - MoveStopLoss
            elif KBar_dic['close'][n] < StopLossPoint:
                OrderRecord.Cover('Sell', KBar_dic['product'][n + 1], KBar_dic['time'][n + 1], KBar_dic['open'][n + 1], OrderRecord.GetOpenInterest())
                continue

            if KBar_dic['close'][n] >= KBar_dic['Upper'][n]:
                OrderRecord.Cover('Sell', KBar_dic['product'][n + 1], KBar_dic['time'][n + 1], KBar_dic['open'][n + 1], OrderRecord.GetOpenInterest())
                continue

        elif OrderRecord.GetOpenInterest() < 0:
            if KBar_dic['product'][n + 1] != KBar_dic['product'][n]:
                OrderRecord.Cover('Buy', KBar_dic['product'][n], KBar_dic['time'][n], KBar_dic['close'][n], -OrderRecord.GetOpenInterest())
                continue

            if KBar_dic['close'][n] + MoveStopLoss < StopLossPoint:
                StopLossPoint = KBar_dic['close'][n] + MoveStopLoss
            elif KBar_dic['close'][n] > StopLossPoint:
                OrderRecord.Cover('Buy', KBar_dic['product'][n + 1], KBar_dic['time'][n + 1], KBar_dic['open'][n + 1], -OrderRecord.GetOpenInterest())
                continue

            if KBar_dic['close'][n] <= KBar_dic['Lower'][n]:
                OrderRecord.Cover('Buy', KBar_dic['product'][n + 1], KBar_dic['time'][n + 1], KBar_dic['open'][n + 1], -OrderRecord.GetOpenInterest())
                continue

# OrderRecord.GetTradeRecord()
# OrderRecord.GetProfit()
# OrderRecord.GetTotalProfit()
# OrderRecord.GetWinRate()
# OrderRecord.GetAccLoss()
# OrderRecord.GetMDD()
print('交易紀錄: \n', OrderRecord.GetTradeRecord())
print('\n')
print('利潤清單(TWD,點數): \n',OrderRecord.GetProfit())
print('\n')
print('淨利(TWD,點數): ',OrderRecord.GetTotalProfit())
print('\n')
print('勝率: ',OrderRecord.GetWinRate())
print('\n')
print('最大連續虧損(TWD,點數): ',OrderRecord.GetAccLoss())
print('\n')
print('最大累計盈虧(TWD,點數)回落: ',OrderRecord.GetMDD())
print('\n')
OrderRecord.GeneratorProfitChart(StrategyName='BBands Strategy')


#%%
###### (五) MACD 策略   
## 對於日K, 以下的三個數字時常設定為(依序): 12, 26, 9
Fastperiod = 12    #15
Slowperiod = 26    #20
Signalperiod = 9   #9

## 不同名稱:
'''
1. TA-Lib MACD 回傳值命名：
 macd       = MACD line = DIF = EMA(price,fastperiod) - EMA(price,slowperiod)
 macdsignal = Signal line = DEA = EMA(DIF, signalperiod)
 macdhist   = Histogram = DIF - DEA

2. 注意：
  某些中文教材/軟體會把 DEA 稱為 MACD 線，
  也會把柱狀圖稱為 MACD(老師提供的資料即是此情況)。
  但在 TA-Lib 中，macd 指的是 DIF，不是 DEA，也不是柱狀圖。
3. 老師提供的資料 與 TA-Lib 名稱對照(左邊TA-Lib, 右邊老師資料): macd="DIF", macdsignal="DEA", macdhist="DIF-DEA"="MACD"
'''
KBar_dic['macd'],KBar_dic['macdsignal'],KBar_dic['macdhist']=MACD(KBar_dic,fastperiod=Fastperiod, slowperiod=Slowperiod, signalperiod=Signalperiod)

##### 策略
'''
 1. 多方進場: macdhist>0; 多方出場: macdhist<0
 2. 多方: macdhist>0 且 macdsignal>0
 3. 黃金交叉 (Buy)：快線（DIF）由下往上穿越慢線（DEA），柱狀體由負轉正，視為買進訊號(多方):
    macd 向上突破 macdsignal(黃金交叉))=macdhist從負的變成正的  
 4. 死亡交叉 (Sell)：快線（DIF）由上往下穿越慢線（DEA），柱狀體由正轉負，視為賣出訊號(空方):
    macd 向下突破 macdsignal(死亡交叉))=macdhist從正的變成負的
 5. 多方: macd>0 且 macdsignal>0 且 (macd 向上突破 macdsignal(黃金交叉))=macdhist從負的變成正的
 6. 空方: macd<0 且 macdsignal<0 且 (macd 向下突破 macdsignal(死亡交叉))=macdhist從正的變成負的
 7 背離:
   (1)、MACD多頭背離（底背離）:
       當價格創新低，但 MACD 指標卻沒有跟上，DIF 線與 MACD 線反而向上推移，柱狀體也未能給出相對動能，出現了多頭背離的狀況，表示空頭行情即將或已經到底，有較高的機率出現反轉，被視為「買進訊號」。

   (2)、MACD空頭背離（頂背離）
       當價格創新高，但 MACD 指標卻沒有跟上，DIF 線與 MACD 線反而向下推移，柱狀體也未能給出相對動能，出現了空頭背離的狀況，表示多頭行情即將或已經到頂，有較高的機率出現反轉，被視為「賣出訊號」。  
'''


#%%
###### (六) KDJ 策略 
##### 設定 STOCH 的參數
'''
1. STOCH 的輸出入型式:
slowk, slowd = STOCH(
    high,
    low,
    close,
    fastk_period=5,
    slowk_period=3,
    slowk_matype=0,
    slowd_period=3,
    slowd_matype=0
)


2. TA-Lib STOCH 參數說明：
參數名	        	    預設值
fastk_period	    	      5
slowk_period	          3
slowk_matype	    	      0
slowd_period	    	      3
slowd_matype	    	      0

(1) fastk_period:
    計算 RSV / fast %K 的區間長度。
    RSV = (Close - N期最低價) / (N期最高價 - N期最低價) * 100

(2) slowk_period:
    將 fast %K 平滑成 slow %K 的期數。
    slowk = MA(fastk, slowk_period, slowk_matype)

(3) slowk_matype:
    slowk 使用的移動平均方法。
    0 = SMA, 1 = EMA, 2 = WMA, ...

(4) slowd_period:
    將 slowk 平滑成 slowd 的期數。
    slowd = MA(slowk, slowd_period, slowd_matype)

(5)slowd_matype:
    slowd 使用的移動平均方法。
    0 = SMA, 1 = EMA, 2 = WMA, ...

注意：
    即使 slowk_matype 或 slowd_matype 改成 EMA，
    slowk_period 與 slowd_period 仍然需要設定。
    matype 決定平滑方法，period 決定平滑期數。
'''

params = {
    'fastk_period': 5,
    'slowk_period': 3,
    'slowk_matype': 0,
    'slowd_period': 3,
    'slowd_matype': 0
}

##### 計算 STOCH 指標
stoch = STOCH(KBar_dic, **params)

## STOCH(KBar_dic, **params) 回傳的 stoch 不是 DataFrame，而是 list / tuple，所以不能用字串 'slowk' 當索引。
## 不同 TA-Lib / pandas 版本可能回傳不同型態：
## 1. DataFrame / dict-like：可用欄位名稱 'slowk'、'slowd'
## 2. list / tuple：必須用整數索引 0、1
if isinstance(stoch, (list, tuple)):
    KBar_dic['slowk'] = np.asarray(stoch[0], dtype=np.float64)
    KBar_dic['slowd'] = np.asarray(stoch[1], dtype=np.float64)
else:
    KBar_dic['slowk'] = np.asarray(stoch['slowk'], dtype=np.float64)
    KBar_dic['slowd'] = np.asarray(stoch['slowd'], dtype=np.float64)
KBar_dic['J'] = 3 * KBar_dic['slowk'] - 2 * KBar_dic['slowd']

##### 策略:
'''
隨機指標 KDJ 的設計，首先計算最高價、最低價和收盤價之間的比例關係，再運用均線平滑及乖離的思想，據以捕捉動量及超買超賣等現象，在實務上對快速直觀地研判行情很有助益。
以下介紹 3 種簡單的 KDJ 指標交易策略:

1.在 KDJ 指標的取值上，K 值與 D 值的取值範圍是 0 到 100, 類似RSI。
 依據 K 值與 D 值可以劃分出超買超賣區，一般而言，K 值或者 D 值取值在 80 以上為超買區；K 值或者 D 值取值在 20 以下為超賣區。

2.對於 J 值，當 J 值大於 100 時，可以視為超買區，當 J 值小於 0 時，視為超賣區。

3.K 線、D 線的交叉情況也可以釋放出買入賣出信號:
 (1)當 K 線由下向上穿過 D 線時，即出現所謂「黃金交叉」現象，隱含股票價格上漲的動量較大，釋放出買入信號；
 (2)當 K 線由上向下穿過 D 線時，出現「死亡交叉」現象，股票有下跌的趨勢，釋放出賣出信號。
'''





# #%%
# ### Pandas DataFrame items() Method:
# data = {
#   "firstname": ["Sally", "Mary", "John"],
#   "age": [50, 40, 30]
# }
# df_demo = pd.DataFrame(data)
# df_demo.head()

# for x, y in df_demo["firstname"].items():
#     print(x)
#     print(y)

# for x, y in df_demo["age"].items():
#     print(x)
#     print(y)