# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 23:41:58 2026

@author: user
"""

# 載入必要套件
import matplotlib.pyplot as plt
import numpy as np
import time


# 下單部位管理物件
class Record():
    def __init__(self):   ## 建構子
        # 儲存績效
        self.Profit = []         # 每筆完整交易的總損益
        self.Profit_rate = []    # 每筆完整交易的報酬率
        # 未平倉
        self.OpenInterestQty = 0
        self.OpenInterest = []   # [side, product, order_time, order_price, qty]
        # 交易紀錄總計
        self.TradeRecord = []    # [B/S, product, entry_time, entry_price, exit_time, exit_price, qty]

    # 進場紀錄
    def Order(self, BS, Product, OrderTime, OrderPrice, OrderQty):
        if OrderQty <= 0:
            return

        qty = int(OrderQty)

        if BS == 'B' or BS == 'Buy':
            self.OpenInterest.append([1, Product, OrderTime, OrderPrice, qty])
            self.OpenInterestQty += qty

        elif BS == 'S' or BS == 'Sell':
            self.OpenInterest.append([-1, Product, OrderTime, OrderPrice, qty])
            self.OpenInterestQty -= qty

    # 出場紀錄(買賣別需與進場相反，多單進場則空單出場)
    def Cover(self, BS, Product, CoverTime, CoverPrice, CoverQty):
        if CoverQty <= 0:
            return

        remain_qty = int(CoverQty)

        # 平多單
        if BS == 'S' or BS == 'Sell':
            while remain_qty > 0:
                long_positions = [x for x in self.OpenInterest if x[0] == 1]
                if len(long_positions) == 0:
                    print('尚無進場')
                    return

                pos = long_positions[0]
                pos_qty = pos[4]
                close_qty = min(remain_qty, pos_qty)

                entry_price = pos[3]
                profit_unit = CoverPrice - entry_price
                profit_total = profit_unit * close_qty
                profit_rate = profit_unit / entry_price if abs(entry_price) > 1e-12 else 0

                # 新增交易紀錄
                self.TradeRecord.append([
                    'B',
                    pos[1],
                    pos[2],
                    pos[3],
                    CoverTime,
                    CoverPrice,
                    close_qty
                ])

                # 紀錄績效
                self.Profit.append(profit_total)
                self.Profit_rate.append(profit_rate)

                # 更新未平倉
                if close_qty == pos_qty:
                    self.OpenInterest.remove(pos)
                else:
                    pos[4] -= close_qty

                self.OpenInterestQty -= close_qty
                remain_qty -= close_qty

        # 平空單
        elif BS == 'B' or BS == 'Buy':
            while remain_qty > 0:
                short_positions = [x for x in self.OpenInterest if x[0] == -1]
                if len(short_positions) == 0:
                    print('尚無進場')
                    return

                pos = short_positions[0]
                pos_qty = pos[4]
                close_qty = min(remain_qty, pos_qty)

                entry_price = pos[3]
                profit_unit = entry_price - CoverPrice
                profit_total = profit_unit * close_qty
                profit_rate = profit_unit / entry_price if abs(entry_price) > 1e-12 else 0

                # 新增交易紀錄
                self.TradeRecord.append([
                    'S',
                    pos[1],
                    pos[2],
                    pos[3],
                    CoverTime,
                    CoverPrice,
                    close_qty
                ])

                # 紀錄績效
                self.Profit.append(profit_total)
                self.Profit_rate.append(profit_rate)

                # 更新未平倉
                if close_qty == pos_qty:
                    self.OpenInterest.remove(pos)
                else:
                    pos[4] -= close_qty

                self.OpenInterestQty += close_qty
                remain_qty -= close_qty

    # 取得當前未平倉數量
    def GetOpenInterest(self):
        return self.OpenInterestQty

    # 取得交易紀錄清單(要進場與出場之後, 才會列入至此)
    def GetTradeRecord(self):
        return self.TradeRecord

    # 取得交易盈虧清單
    def GetProfit(self):
        return self.Profit

    # 取得交易投資報酬率清單
    def GetProfitRate(self):
        return self.Profit_rate

    # 取得交易總盈虧
    def GetTotalProfit(self):
        if len(self.Profit) == 0:
            return 0
        return sum(self.Profit)

    # 取得交易次數
    def GetTotalNumber(self):
        return len(self.Profit)

    # 取得平均交易盈虧(每次)
    def GetAverageProfit(self):
        if len(self.Profit) == 0:
            return 0
        return sum(self.Profit) / len(self.Profit)

    # 取得交易 "平均" 投資報酬率
    def GetAverageProfitRate(self):
        if len(self.Profit_rate) == 0:
            return 0
        return sum(self.Profit_rate) / len(self.Profit_rate)

    # 取得勝率 = 賺錢的交易次數 / 總交易次數
    def GetWinRate(self):
        if len(self.Profit) == 0:
            return 0
        WinProfit = [i for i in self.Profit if i > 0]
        return len(WinProfit) / len(self.Profit)

    # 最大連續虧損(TWD,點數)
    def GetAccLoss(self):
        if len(self.Profit) == 0:
            return 0

        AccLoss = 0
        MaxAccLoss = 0
        for p in self.Profit:
            if p <= 0:
                AccLoss += p
                if AccLoss < MaxAccLoss:
                    MaxAccLoss = AccLoss
            else:
                AccLoss = 0
        return MaxAccLoss

    # 最大 "累計盈虧(TWD,點數)" 回落(MDD)
    def GetMDD(self):
        if len(self.Profit) == 0:
            return 0

        MDD, Capital, MaxCapital = 0, 0, 0
        for p in self.Profit:
            Capital += p
            MaxCapital = max(MaxCapital, Capital)
            DD = MaxCapital - Capital
            MDD = max(MDD, DD)
        return MDD

    # 最大 "累計投資報酬率" 回落(MDD_rate)
    def GetMDD_rate(self):
        if len(self.Profit_rate) == 0:
            return 0

        MDD_rate, Capital_rate, MaxCapital_rate = 0, 0, 0
        for p in self.Profit_rate:
            Capital_rate += p
            MaxCapital_rate = max(MaxCapital_rate, Capital_rate)
            DD_rate = MaxCapital_rate - Capital_rate
            MDD_rate = max(MDD_rate, DD_rate)
        return MDD_rate

    # 平均獲利(只看獲利的)
    def GetAverEarn(self):
        WinProfit = [i for i in self.Profit if i > 0]
        if len(WinProfit) == 0:
            return 0
        return sum(WinProfit) / len(WinProfit)

    # 平均虧損(只看虧損的)
    def GetAverLoss(self):
        FailProfit = [i for i in self.Profit if i < 0]
        if len(FailProfit) == 0:
            return 0
        return sum(FailProfit) / len(FailProfit)

    # 累計盈虧(TWD,點數)清單
    def GetCumulativeProfit(self):
        TotalProfit = [0]
        for i in self.Profit:
            TotalProfit.append(TotalProfit[-1] + i)
        return TotalProfit

    # 累計投資報酬率清單
    def GetCumulativeProfit_rate(self):
        TotalProfit_rate = [0]
        for i in self.Profit_rate:
            TotalProfit_rate.append(TotalProfit_rate[-1] + i)
        return TotalProfit_rate

    # 產出交易績效圖(畫出 "累計盈虧(TWD,點數)清單")
    def GeneratorProfitChart(self, StrategyName='Strategy'):
        ax1 = plt.subplot(111)

        TotalProfit = [0]
        for i in self.Profit:
            TotalProfit.append(TotalProfit[-1] + i)

        ax1.plot(TotalProfit, '-', linewidth=1)
        ax1.set_title('Cumulative Profit(TWD,point)')
        plt.show()
        plt.savefig(StrategyName + '.png')
        plt.close()