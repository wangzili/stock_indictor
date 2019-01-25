# -*- coding:utf-8 -*-

"""
股票技术指标接口
Created on 2018/07/26
@author: Wangzili
@group : **
@contact: 446406177@qq.com
"""
import pandas as pd
import numpy as np
from decimal import Decimal
import itertools


def ma(df, n=10, val_name="close"):
    """
    移动平均线 Moving Average
    :param df: pandas.DataFrame
                  通过 get_k_data 取得的股票数据
    :param n: int or list
                  移动平均线时长，时间单位根据df决定
    :param val_name: string
                  计算哪一列的列名，默认为 close 收盘值
    :return:
        pandas.DateFrame
    """
    pv = pd.DataFrame()
    if isinstance(n, int):
        n = [n]
    pv['date'] = df['date']
    for v in n:
        # 方法一
        # pv['v' + str(v)] = _ma(df, v, val_name)
        # 方法二  比方法大约快15倍
        pv['v' + str(v)] = df[val_name].rolling(v).mean()
    return pv


def _ma(df, n, val_name):
    values = []
    _mal = []
    for index, row in df.iterrows():
        values.append(row[val_name])
        if len(values) > n:
            del values[0]
        if len(values) == n:
            # ma.append([row['date'], Decimal(np.average(values)).quantize(Decimal('0.00'))])
            _mal.append(round(Decimal(np.average(values)), 2))
        else:
            _mal.append(0)
    return _mal


def md(df, n=10, val_name="close"):
    """
    移动标准差
    :param df: pandas.DataFrame
                通过ts.get_k_data返回数据
    :param n: int
                时长
    :param val_name: string
                计算哪一列的列名，默认为 close 收盘值
    :return: pandas.DateFrame
    """
    # 方法一
    '''
    values = []
    MD = []
    _md = pd.DataFrame()
    _md['date'] = df['date']
    for index, row in df.iterrows():
        values.append(row[val_name])
        if len(values) > n:
            del values[0]
        if len(values) == n:
            MD.append(round(Decimal(np.std(values)), 2))
        else:
            MD.append(0)
    _md['md' + str(n)] = MD
    '''
    # 方法二
    _md = pd.DataFrame()
    _md['date'] = df['date']
    _md["md" + str(n)] = df[val_name].rolling(n).std(ddof=0)
    return _md


def ema(df, n=12, val_name="close"):
    """
    指数平均数指标 Exponential Moving Average
    :param df: pandas.DataFrame
                通过 get_h_data 取得的股票数据
    :param n: int
                移动平均线时长，时间单位根据df决定
    :param val_name: string
                计算哪一列的列名，默认为 close 收盘值
    :return _ema: pandas.DataFrame

    今日EMA（N）=2/（N+1）×今日收盘价+(N-1)/（N+1）×昨日EMA（N）
    EMA(X,N)=[2×X+(N-1)×EMA(ref(X),N]/(N+1)
    """
    EMA = []
    _ema = pd.DataFrame()
    _ema['date'] = df['date']
    for index, row in enumerate(df.itertuples(index=False)):
        if index == 0:
            today_ema = getattr(row, val_name)
        else:
            # Y=[2*X+(N-1)*Y’]/(N+1)
            today_ema = (2 * getattr(row, val_name) + (n - 1) * today_ema) / (n + 1)
        EMA.append(today_ema)
    _ema['ema' + str(n)] = EMA
    # _ema['ema' + str(n)] = df['close'].ewm(ignore_na=False, span=n, min_periods=0, adjust=False).mean()
    return _ema


def macd(df, quick_n=12, slow_n=26, dem_n=9, val_name="close"):
    """
    平滑异同移动平均线(Moving Average Convergence Divergence)
    今日EMA（N）=2/（N+1）×今日收盘价+(N-1)/（N+1）×昨日EMA（N）
    DIFF= EMA（N1）- EMA（N2）
    DEA(DIF,M)= 2/(M+1)×DIF +[1-2/(M+1)]×DEA(REF(DIF,1),M)
    MACD（BAR）=2×（DIF-DEA）
    :param df:
    :param quick_n: DIFF差离值中快速移动天数
    :param slow_n: DIFF差离值中慢速移动天数
    :param dem_n: DEM讯号线的移动天数
    :param val_name: 计算哪一列的列名，默认为 close 收盘值
    :return:
        _macd: pandas.DataFrame
          date: 日期
          osc: MACD bar / OSC 差值柱形图 DIFF - DEM
          diff: 差离值
          dem: 讯号线
    """
    _macd = pd.DataFrame()
    _macd['date'] = df['date']

    ema_quick = ema(df, quick_n, val_name)
    ema_slow = ema(df, slow_n, val_name)
    _macd['diff'] = ema_quick['ema' + str(quick_n)] - ema_slow['ema' + str(slow_n)]
    dem = ema(_macd, dem_n, "diff")
    _macd['dea'] = dem['ema' + str(dem_n)]
    _macd['macd'] = _macd['diff'] - _macd['dem']

    return _macd


def kdj(df, n=9):
    """
    随机指标KDJ
    :param df: pandas.DataFrame
                通过 get_k_data 取得的股票数据
    :return: _kdj:pandas.DataFrame
                包含 date, k, d, j
    """
    # pd.set_option('display.max_rows', 1000)
    _kdj = pd.DataFrame()
    _kdj['date'] = df['date']
    _k, _d, _j = [], [], []
    last_k = last_d = 50
    for index, row in df.iterrows():
        l = df['low'].loc[index - n + 1: index].min()
        h = df['high'].loc[index - n + 1: index].max()
        c = row["close"]

        rsv = (c - l) / (h - l) * 100
        k = (2 / 3) * last_k + (1 / 3) * rsv
        d = (2 / 3) * last_d + (1 / 3) * k
        j = 3 * k - 2 * d

        _k.append(round(Decimal(k), 2))
        _d.append(round(Decimal(d), 2))
        _j.append(round(Decimal(j), 2))

        last_k, last_d = k, d
    _kdj['k'] = _k
    _kdj['d'] = _d
    _kdj['j'] = _j
    return _kdj


def rsi(df, n=6, val_name="close"):
    """
    相对强弱指标（Relative Strength Index，简称RSI
    :param df: pandas.DataFrame
                    通过 get_k_data 取得的股票数据
    :param n: int
                    统计时长，时间单位根据data决定
    :param val_name:
                    计算哪一列的列名，默认为 close 收盘值
    :return: _rsi: pandas.DataFrame
                    包含 date, rsi
    """
    # pd.set_option('display.max_rows', 1000)
    _rsi = pd.DataFrame()
    _rsi['date'] = df['date']
    '''
    # 方法一
    r = []
    up = 0
    down = 0
    for index, row in enumerate(df.itertuples(index=False)):
        if index == 0:
            past_value = getattr(row, val_name)
            r.append(0)
        else:
            diff = getattr(row, val_name) - past_value
            if diff > 0:
                up = (up * (n - 1) + diff) / n
                down = down * (n - 1) / n
            else:
                up = up * (n - 1) / n
                down = (down * (n - 1) + abs(diff)) / n

            rsit = up / (down + up) * 100
            # r.append(round(Decimal(rsit), 2))
            r.append(rsit)

            past_value = getattr(row, val_name)

    _rsi['rsi'] = r
    '''
    # LC = REF(CLOSE, 1)
    # RSI = SMA(MAX(CLOSE - LC, 0), N, 1) / SMA(ABS(CLOSE - LC), N1, 1)×100
    # 方法二
    px = df['close'] - df['close'].shift(1)
    px[px < 0] = 0
    _rsi['rsi'] = sma(px, n, 1) / sma((df['close'] - df['close'].shift(1)).abs(), n, 1) * 100
    # 方法三
    # def tmax(x):
    #     if x < 0:
    #         x = 0
    #     return x
    # _rsi['rsi'] = sma((df['close'] - df['close'].shift(1)).apply(tmax), n, 1) / sma((df['close'] - df['close'].shift(1)).abs(), n, 1) * 100
    return _rsi


def vrsi(df, n=6):
    """
    量相对强弱指标
    VRSI=SMA（最大值（成交量-REF（成交量，1），0），N,1）/SMA（ABS（（成交量-REF（成交量，1），N，1）×100%
    :param df:
    :param n:
    :return:
    """
    _vrsi = pd.DataFrame()
    _vrsi['date'] = df['date']
    px = df['volume'] - df['volume'].shift(1)
    px[px < 0] = 0
    _vrsi['vrsi'] = sma(px, n, 1) / sma((df['volume'] - df['volume'].shift(1)).abs(), n, 1) * 100
    return _vrsi


def boll(df, n=20, k=2):
    """
    布林线指标BOLL boll(26,2)	MID=MA(N)
    标准差MD=根号[∑（CLOSE-MA(CLOSE，N)）^2/N]
    UPPER=MID＋k×MD
    LOWER=MID－k×MD
    """
    _boll = pd.DataFrame()
    _boll['date'] = df.date
    _boll['mid'] = df.close.rolling(n).mean()
    _md = df.close.rolling(n).std(ddof=0)
    _boll['up'] = _boll.mid + k * _md
    _boll['low'] = _boll.mid - k * _md
    return _boll


def bbiboll(df, n=10, k=3):
    """
    BBI多空布林线	bbiboll(10,3)
    BBI={MA(3)+ MA(6)+ MA(12)+ MA(24)}/4
    标准差MD=根号[∑（BBI-MA(BBI，N)）^2/N]  # 没搞明白什么意思
    UPR= BBI＋k×MD
    DWN= BBI－k×MD
    """
    pd.set_option('display.max_rows', 1000)
    _bbiboll = pd.DataFrame()
    _bbiboll['date'] = df.date
    _bbiboll['bbi'] = (df.close.rolling(3).mean() + df.close.rolling(6).mean() +
                       df.close.rolling(12).mean() + df.close.rolling(24).mean()) / 4
    # _bbiboll['md'] = np.sqrt(np.square(_bbiboll.bbi - _bbiboll.bbi.rolling(n).mean()) / n)
    _bbiboll['md'] = _bbiboll.bbi.rolling(n).std(ddof=0)
    _bbiboll['upr'] = _bbiboll.bbi + k * _bbiboll.md
    _bbiboll['dwn'] = _bbiboll.bbi - k * _bbiboll.md
    return _bbiboll


def wr(df, n=14):
    """
    威廉指标 w&r
    :param df: pandas.DataFrame
                    通过 get_k_data 取得的股票数据
    :param n: int
                    统计时长，时间单位根据data决定
    :return: pandas.DataFrame
            WNR: 威廉指标
    """

    _wnr = pd.DataFrame()
    _wnr['date'] = df['date']
    high_prices = []
    low_prices = []
    WNR = []

    for index, row in data.iterrows():
        high_prices.append(row["high"])
        if len(high_prices) > n:
            del high_prices[0]
        low_prices.append(row["low"])
        if len(low_prices) > n:
            del low_prices[0]

        highest = max(high_prices)
        lowest = min(low_prices)

        wnr = (highest - row["close"]) / (highest - lowest) * 100
        WNR.append(wnr)
    _wnr['wr' + str(n)] = WNR
    return _wnr


def _ma_exact(df, n, val_name="close"):
    values = []
    _mal = []
    for index, row in df.iterrows():
        values.append(row[val_name])
        if len(values) > n:
            del values[0]
        _mal.append(np.average(values))
    return _mal


def _ma_array(arr, n):
    MA = []
    values = []
    for val in arr:
        values.append(val)
        if len(values) > n:
            del values[0]
        MA.append(np.average(values))
    return np.asarray(MA)


def bias(df, n=6):
    """
        乖离率 bias
        Parameters
        ------
          df:pandas.DataFrame
                      通过 get_k_data 取得的股票数据
          n:int
              统计时长，默认6 一般取值 6， 12， 24, 72
        return
        -------
          BIAS:numpy.ndarray<numpy.float64>
              乖离率指标

    """

    _bias = pd.DataFrame()
    _bias['date'] = df['date']
    c = df["close"]
    if isinstance(n, int):
        n = [n]
    for b in n:
        _mav = _ma_exact(df, b)
        # print(np.vectorize(lambda x: round(Decimal(x), 2))(c))
        BIAS = (np.true_divide((c - _mav), _mav)) * 100
        _bias["bias" + str(b)] = np.vectorize(lambda x: round(Decimal(x), 4))(BIAS)
    return _bias


def asi(df, n=5):
    """
        振动升降指标 ASI
        Parameters
        ------
          df:pandas.DataFrame
                      通过 get_k_data 取得的股票数据
          n:int
              统计时长，默认5
        return
        -------
          ASI:numpy.ndarray<numpy.float64>
              振动升降指标

    """
    _si = []
    for index, row in enumerate(df.itertuples(index=False)):
        if index == 0:
            _si.append(0)
            last_row = row
        else:
            a = abs(getattr(row, "high") - getattr(last_row, "close"))
            b = abs(getattr(row, "low") - getattr(last_row, "close"))
            c = abs(getattr(row, "high") - getattr(last_row, "low"))
            d = abs(getattr(last_row, "close") - getattr(last_row, "open"))

            e = getattr(row, 'close') - getattr(last_row, 'close')
            f = getattr(row, 'close') - getattr(row, 'open')
            g = getattr(last_row, 'close') - getattr(last_row, 'open')

            x = e + (1/2) * f + g
            k = max(a, b)

            if max(a, b, c) == a:
                r = a + (1/2) * b + (1/4) * d
            elif max(a, b, c) == b:
                r = b + (1/2) * a + (1/4) * d
            else:
                r = c + (1/4) * d
            l = 3
            SI = 50 * (x/r) * (k/l)
            _si.append(SI)
    _asi = pd.DataFrame()
    _asi['date'] = df['date']
    if isinstance(n, int):
        n = [n]
    for a in n:
        _asi["asi" + str(a)] = _ma_array(_si, a)
    return _asi


def vr_rate(df, n=26):
    """
    成交量变异率 vr or vr_rate
    VR=（AVS+1/2CVS）/（BVS+1/2CVS）×100
    其中：
    AVS：表示N日内股价上涨成交量之和
    BVS：表示N日内股价下跌成交量之和
    CVS：表示N日内股价不涨不跌成交量之和
    """
    '''  # 方法一
    VR = []
    av, bv, cv = list(), list(), list()
    index_v = []
    for index, row in enumerate(df.itertuples(index=False)):
        if index == 0:
            av.append(getattr(row, 'close'))
            index_v.append('av')
        else:
            if getattr(row, 'close') > pre_close:
                av.append(getattr(row, 'volume'))
                index_v.append('av')
            elif getattr(row, 'close') < pre_close:
                bv.append(getattr(row, 'volume'))
                index_v.append("bv")
            else:
                cv.append(getattr(row, 'volume'))
                index_v.append("cv")
        pre_close = getattr(row, 'close')
        if len(index_v) > n:
            (locals()[index_v.pop(0)]).pop(0)
        avs = sum(av)
        bvs = sum(bv)
        cvs = sum(cv)

        if bvs + (1 / 2) * cvs:
            vrd = (avs + (1 / 2) * cvs) / (bvs + (1 / 2) * cvs) * 100
        else:
            vrd = 0

        VR.append(vrd)
    _vr["vrr"] = VR
    '''  # 方法二
    _vr = pd.DataFrame()
    _vr['date'] = df['date']
    _m = pd.DataFrame()
    _m['volume'] = df.volume
    _m['cs'] = df.close - df.close.shift(1)
    _m['avs'] = _m.apply(lambda x: x.volume if x.cs > 0 else 0, axis=1)
    _m['bvs'] = _m.apply(lambda x: x.volume if x.cs < 0 else 0, axis=1)
    _m['cvs'] = _m.apply(lambda x: x.volume if x.cs == 0 else 0, axis=1)
    _vr["vr"] = (_m.avs.rolling(n).sum() + 1 / 2 * _m.cvs.rolling(n).sum()
                 ) / (_m.bvs.rolling(n).sum() + 1 / 2 * _m.cvs.rolling(n).sum()) * 100
    return _vr


def vr(df, n=5):
    """
    开市后平均每分钟的成交量与过去5个交易日平均每分钟成交量之比
    量比:=V/REF(MA(V,5),1);
    涨幅:=(C-REF(C,1))/REF(C,1)*100;
    1)量比大于1.8，涨幅小于2%，现价涨幅在0—2%之间，在盘中选股的
    选股:量比>1.8 AND 涨幅>0 AND 涨幅<2;
    """
    _vr = pd.DataFrame()
    _vr['date'] = df.date
    _vr['vr'] = df.volume / (df.volume.rolling(n).mean().shift(1))
    _vr['rr'] = (df.close - df.close.shift(1)) / df.close.shift(1) * 100
    return _vr


def arbr(df, n=26):
    """
        AR 指标 BR指标
        Parameters
        ------
          df:pandas.DataFrame
                      通过 get_k_data 取得的股票数据
          n:int
              统计时长，默认26
        return
        -------
          AR:
              AR指标
          BR:
              BR指标
    """
    _arbr = pd.DataFrame()
    _arbr['date'] = df['date']
    H, L, O, PC = np.array([0]), np.array([0]), np.array([0]), np.array([0])
    AR, BR = np.array([0]), np.array([0])
    for index, row in enumerate(df.itertuples(index=False)):
        if index == 0:
            last_row = row
        else:
            H = np.append(H, [getattr(row, "high")])
            L = np.append(L, [getattr(row, 'low')])
            O = np.append(O, [getattr(row, 'open')])
            PC = np.append(PC, [getattr(last_row, "close")])
            if len(H) > n:
                H = np.delete(H, 0)
                L = np.delete(L, 0)
                O = np.delete(O, 0)
                PC = np.delete(PC, 0)

            # ar = (np.sum(np.asarray(H) - np.asarray(O)) / sum(np.asarray(O) - np.asarray(L))) * 100
            ar = (np.sum(H - O) / sum(O - L)) * 100
            AR = np.append(AR, [ar])
            br = (np.sum(H - PC) / sum(PC - L)) * 100
            BR = np.append(BR, [br])

            last_row = row
    _arbr['ar'] = AR
    _arbr["br"] = BR

    return _arbr


def dpo(df, n=20, m=6):
    """
        区间震荡线指标 DPO
        Parameters
        ------
          df:pandas.DataFrame
                      通过 get_k_data 取得的股票数据
          n:int
              统计时长，默认20
          m:int
              MADPO的参数M，默认6
        return
        -------
            data
          DPO:numpy.ndarray<numpy.float64>
              DPO指标
          MADPO:numpy.ndarray<numpy.float64>
              MADPO指标

    """
    _dpo = pd.DataFrame()
    _dpo['date'] = df['date']
    DPO = df['close'] - np.asarray(_ma_exact(df, int(n / 2 + 1)))
    MADPO = _ma_array(DPO, m)
    _dpo['dop'] = DPO
    _dpo['dopma'] = MADPO
    return _dpo


def _ema(arr, n=12):
    EMA = []
    for index, a in enumerate(arr):
        if index == 0:
            t_ema = a
        else:
            # Y=[2*X+(N-1)*Y’]/(N+1)
            t_ema = (2 * a + (n - 1) * t_ema) / (n + 1)
        EMA.append(t_ema)
    return EMA


def trix(df, n=12, m=20):
    """ 长期指标
        三重指数平滑平均线 TRIX
        Parameters
        ------
          df:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
          n:int
              统计时长，默认12
          m:int
              TRMA的参数M，默认20
        return
        -------
          TRIX:
              trix指标
          TRMA:
              trix平均指标

    """
    _trix = pd.DataFrame()
    _trix['date'] = df['date']
    # 方法一
    trx = []
    for index, row in enumerate(df.itertuples(index=False)):
        if index == 0:
            ax_last = getattr(row, 'close')
            bx_last = getattr(row, 'close')
            tx_last = getattr(row, 'close')
            trx.append(0)
        else:
            ax = (2 * getattr(row, "close") + (n - 1) * ax_last) / (n + 1)
            bx = (2 * ax + (n - 1) * bx_last) / (n + 1)
            tx = (2 * bx + (n - 1) * tx_last) / (n + 1)
            if tx_last != 0:
                trx.append((tx - tx_last) / tx_last * 100)
            else:
                trx.append(0)
            ax_last = ax
            bx_last = bx
            tx_last = tx
    # 方法二
    '''
    trxi = _ema(_ema(_ema(df["close"], n), n), n)
    trx = []
    for index, t in enumerate(trxi):
        if index in [0, 1]:
            trx.append(0)
        else:
            trx.append((t-trxi[index-1])/trxi[index-1]*100)
    '''
    trma = _ma_array(trx, m)
    _trix["trix"] = trx
    _trix["trma"] = trma
    return _trix


def bbi(df):
    """
        Bull And Bearlndex 多空指标
        Parameters
        ------
          df:pandas.DataFrame
                      通过 get_k_data 取得的股票数据
        return
        -------
          BBI:
              BBI指标

    """
    _bbi = pd.DataFrame()
    _bbi['date'] = df['date']
    CS = []
    BBI = []
    for index, row in df.iterrows():
        CS.append(row["close"])

        if len(CS) < 24:
            BBI.append(row["close"])
        else:
            bbi = np.average([np.average(CS[-3:]), np.average(CS[-6:]), np.average(CS[-12:]), np.average(CS[-24:])])
            BBI.append(bbi)
    _bbi['bbi'] = BBI
    return _bbi


def mtm(df, n=6):
    """
        Momentum Index 动量指标
        Parameters
        ------
          df:pandas.DataFrame
                      通过 get_k_data 取得的股票数据
          n:int
              统计时长，默认6
        return
        -------
          MTM:
              MTM动量指标

        MTM（N日）=C-REF(C,N)式中，C=当日的收盘价，REF(C,N)=N日前的收盘价；N日是只计算交易日期，剔除掉节假日。
        MTMMA（MTM，N1）= MA（MTM，N1）
        N表示间隔天数，N1表示天数
    """
    _mtm = pd.DataFrame()
    _mtm['date'] = df['date']
    MTM = []
    CN = []
    for index, row in enumerate(df.itertuples(index=False)):
        if index < n:
            MTM.append(0.)
        else:
            mtm = getattr(row, 'close') - CN[index - n]
            MTM.append(mtm)
        CN.append(getattr(row, 'close'))
    _mtm["mtm"] = MTM
    _mtm["mtmma"] = _ma_array(MTM, n - 1)
    return _mtm


def obv(df):
    """ # TODO:与同花顺指标数据不匹配
    能量潮  On Balance Volume
    多空比率净额= [（收盘价－最低价）－（最高价-收盘价）] ÷（ 最高价－最低价）×V
    """
    _obv = pd.DataFrame()
    _obv["date"] = df['date']
    tmp = np.true_divide(((df["close"] - df["low"]) - (df["high"] - df["close"])), (df["high"] - df["low"]))
    OBV = tmp * df["volume"]
    _obv["obv"] = OBV.expanding(1).sum() / 100
    return _obv


def cci(df, n=14):
    """
    顺势指标
    :param df: pandas.DataFrame (get_k_data获取的数据)
    :param n: 统计时长，默认14
    :return: pandas.DataFrame

    同花顺算法：TYP:=(HIGH+LOW+CLOSE)/3
              CCI:=(TYP-MA(TYP,N))/(0.015×AVEDEV(TYP,N))
    """
    _cci = pd.DataFrame()
    _cci["date"] = df['date']
    df['typ'] = (df['high'] + df['low'] + df['close']) / 3
    _cci['cci'] = ((df['typ'] - df['typ'].rolling(n).mean()) /
                 (0.015 * df['typ'].rolling(min_periods=1, center=False, window=n).apply(
                    lambda x: np.fabs(x - x.mean()).mean())))
    # _cci['ccci'] = ((df['typ'] - df['typ'].rolling(n).mean()) /
    #              (0.015 * abs(df['typ'] - df['typ'].rolling(n).mean()).rolling(n).mean()))
    df.drop(columns=['typ'], inplace=True)
    return _cci


def priceosc(df, n=12, m=26):
    """
    价格振动指数
    PRICEOSC=(MA(C,12)-MA(C,26))/MA(C,12)
    """
    _c = pd.DataFrame()
    _c['date'] = df['date']
    man = ma(df, n)['v' + str(n)]
    _c['osc'] = (man - ma(df, m)['v' + str(m)]) / man
    return _c


def sma(a, n, m):
    """
    平滑移动指标 Smooth Moving Average
    """
    _sma = []
    for index, value in enumerate(a):
        if index == 0 or pd.isna(value) or np.isnan(value):
            tsma = 0
        else:
            # Y=(M*X+(N-M)*Y')/N
            tsma = (m * value + (n - m) * tsma) / n
        _sma.append(tsma)
    return np.asarray(_sma)


def dbcd(df, n=5, m=16, t=76):
    """
    异同离差乖离率	dbcd(5,16,76)
    BIAS=(C-MA(C,N))/MA(C,N)
    DIF=(BIAS-REF(BIAS,M))
    DBCD=SMA(DIF,T,1) =（1-1/T）×SMA(REF(DIF,1),T,1)+ 1/T×DIF
    MM=MA(DBCD,5)
    :param df:
    :param n:
    :param m:
    :return:
    """
    _dbcd = pd.DataFrame()
    _dbcd['date'] = df['date']
    man = ma(df, n)['v' + str(n)]
    _bias = (df['close'] - man) / man
    _dif = _bias - _bias.shift(m)
    _dbcd['dbcd'] = sma(_dif, t, 1)
    _dbcd['mm'] = _ma_array(_dbcd['dbcd'], n)
    return _dbcd


def roc(df, n=12, m=6):
    """
    :param df:
    :param n:
    :param m:
    :return
    变动速率	roc(12,6)	ROC=(今日收盘价-N日前的收盘价)/ N日前的收盘价×100%
    ROCMA=MA（ROC，M）
    ROC:(CLOSE-REF(CLOSE,N))/REF(CLOSE,N)×100
    ROCMA:MA(ROC,M)
    """
    _roc = pd.DataFrame()
    _roc['date'] = df['date']
    '''
    # 方法一
    r = []
    rn = []
    for index, row in enumerate(df.itertuples(index=False)):
        close = getattr(row, 'close')
        if index < n:
            r.append(0)
        else:
            n_close = rn.pop(0)
            # r.append((getattr(row, 'close') - rn[index - n])/rn[index - n] * 100)
            r.append((close - n_close)/n_close * 100)
        rn.append(close)
    _roc['roc'] = r
    '''
    _roc['roc'] = (df['close'] - df['close'].shift(n))/df['close'].shift(n) * 100
    _roc['rocma'] = _ma_array(_roc['roc'], m)
    return _roc


def vroc(df, n=12):
    """
    量变动速率 VROC=(当日成交量-N日前的成交量)/ N日前的成交量×100%
    :param df:
    :param n:
    :return:
    """
    _vroc = pd.DataFrame()
    _vroc['date'] = df['date']
    _vroc['vroc'] = (df['volume'] - df['volume'].shift(n)) / df['volume'].shift(n) * 100
    return _vroc


def cr(df, n=26):
    """ 能量指标
    CR=∑（H-PM）/∑（PM-L）×100
    PM:上一交易日中价（(最高、最低、收盘价的均值)
    H：当天最高价
    L：当天最低价
    :param df:
    :param n:
    :return:
    """
    _cr = pd.DataFrame()
    _cr['date'] = df['date']
    # pm = ((df['high'] + df['low'] + df['close']) / 3).shift(1)
    pm = (df[['high', 'low', 'close']]).mean(axis=1).shift(1)
    _cr['cr'] = (df['high'] - pm).rolling(n).sum()/(pm - df['low']).rolling(n).sum() * 100
    return _cr


def psy(df, n=12):
    """
    心理指标	PSY(12)
    PSY=N日内上涨天数/N×100
    PSY:COUNT(CLOSE>REF(CLOSE,1),N)/N×100
    MAPSY=PSY的M日简单移动平均
    :param df: pandas.DataFrame get_k_data数据
    :param n:
    :return:
    """
    _psy = pd.DataFrame()
    _psy['date'] = df['date']
    p = df['close'] - df['close'].shift()
    p[p <= 0] = np.nan
    _p = p.rolling(n).count() / n * 100
    _psy['psy'] = _p
    return _psy


def wad(df, n=30):
    """
    威廉聚散指标	WAD(30)
    TRL=昨日收盘价与今日最低价中价格最低者；TRH=昨日收盘价与今日最高价中价格最高者
    如果今日的收盘价>昨日的收盘价，则今日的A/D=今日的收盘价－今日的TRL
    如果今日的收盘价<昨日的收盘价，则今日的A/D=今日的收盘价－今日的TRH
    如果今日的收盘价=昨日的收盘价，则今日的A/D=0
    WAD=今日的A/D+昨日的WAD；MAWAD=WAD的M日简单移动平均
    """
    def dmd(x):
        if x.c > 0:
            y = x.close - x.trl
        elif x.c < 0:
            y = x.close - x.trh
        else:
            y = 0
        return y

    _wad = pd.DataFrame()
    _wad['date'] = df['date']
    ad = pd.DataFrame()
    ad['trl'] = np.minimum(df['low'], df['close'].shift(1))
    ad['trh'] = np.maximum(df['high'], df['close'].shift(1))
    ad['c'] = df['close'] - df['close'].shift()
    ad['close'] = df['close']
    ad['ad'] = ad.apply(dmd, axis=1)
    _wad['wad'] = ad['ad'].expanding(1).sum()
    _wad['mawad'] = _ma_array(_wad['wad'], n)
    return _wad


def mfi(df, n=14):
    """
    资金流向指标	mfi(14)
    MF＝TYP×成交量；TYP:当日中价（(最高、最低、收盘价的均值)
    如果当日TYP>昨日TYP，则将当日的MF值视为当日PMF值。而当日NMF值＝0
    如果当日TYP<=昨日TYP，则将当日的MF值视为当日NMF值。而当日PMF值=0
    MR=∑PMF/∑NMF
    MFI＝100-（100÷(1＋MR)）
    """
    _mfi = pd.DataFrame()
    _mfi['date'] = df['date']
    _m = pd.DataFrame()
    _m['typ'] = df[['high', 'low', 'close']].mean(axis=1)
    _m['mf'] = _m['typ'] * df['volume']
    _m['typ_shift'] = _m.typ - _m.typ.shift(1)
    _m['pmf'] = _m.apply(lambda x: x.mf if x.typ_shift > 0 else 0, axis=1)
    _m['nmf'] = _m.apply(lambda x: x.mf if x.typ_shift <= 0 else 0, axis=1)
    # _mfi['mfi'] = 100 - (100 / (1 + _m.pmf.rolling(n).sum() / _m.nmf.rolling(n).sum()))
    _m['mr'] = _m.pmf.rolling(n).sum() / _m.nmf.rolling(n).sum()
    _mfi['mfi'] = 100 * _m['mr'] / (1 + _m['mr'])  # 同花顺自己给出的公式和实际用的公式不一样，真操蛋，浪费两个小时时间
    return _mfi


def pvt(df):
    """
    pvt	量价趋势指标	pvt
    如果设x=(今日收盘价—昨日收盘价)/昨日收盘价×当日成交量，
    那么当日PVT指标值则为从第一个交易日起每日X值的累加。
    """
    _pvt = pd.DataFrame()
    _pvt['date'] = df.date

    x = (df.close - df.close.shift(1)) / df.close.shift(1) * df.volume
    _pvt['pvt'] = x.expanding(1).sum()
    return _pvt


def wvad(df, n=24, m=6):
    """ # TODO: 与同花顺数据不匹配
    威廉变异离散量	wvad(24,6)
    WVAD=N1日的∑ {(当日收盘价－当日开盘价)/(当日最高价－当日最低价)×成交量}
    MAWVAD=MA（WVAD，N2）
    """
    _wvad = pd.DataFrame()
    _wvad['date'] = df.date
    _wvad['wvad'] = (np.true_divide((df.close - df.open), (df.high - df.low)) * df.volume).rolling(n).sum()
    _wvad['mawvad'] = _ma_array(_wvad['wvad'], m)
    return _wvad


def cdp(df):
    """
    逆势操作	cdp
    CDP=(最高价+最低价+收盘价)/3  # 同花顺实际用的(H+L+2*c)/4
    AH=CDP+(前日最高价-前日最低价)
    NH=CDP×2-最低价
    NL=CDP×2-最高价
    AL=CDP-(前日最高价-前日最低价)
    """
    _cdp = pd.DataFrame()
    _cdp['date'] = df.date
    # _cdp['cdp'] = (df.high + df.low + df.close * 2).shift(1) / 4
    _cdp['cdp'] = df[['high', 'low', 'close', 'close']].shift(1).mean(axis=1)
    _cdp['ah'] = _cdp.cdp + (df.high.shift(1) - df.low.shift(1))
    _cdp['al'] = _cdp.cdp - (df.high.shift(1) - df.low.shift(1))
    _cdp['nh'] = _cdp.cdp * 2 - df.low.shift(1)
    _cdp['nl'] = _cdp.cdp * 2 - df.high.shift(1)
    return _cdp


def env(df, n=14):
    """
    ENV指标	ENV(14)
    Upper=MA(CLOSE，N)×1.06
    LOWER= MA(CLOSE，N)×0.94
    """
    _env = pd.DataFrame()
    _env['date'] = df.date
    _env['up'] = df.close.rolling(n).mean() * 1.06
    _env['low'] = df.close.rolling(n).mean() * 0.94
    return _env


def mike(df, n=12):
    """
    麦克指标	mike(12)
    初始价（TYP）=（当日最高价＋当日最低价＋当日收盘价）/3
    HV=N日内区间最高价
    LV=N日内区间最低价
    初级压力线（WR）=TYP×2-LV
    中级压力线（MR）=TYP+HV-LV
    强力压力线（SR）=2×HV-LV
    初级支撑线（WS）=TYP×2-HV
    中级支撑线（MS）=TYP-HV+LV
    强力支撑线（SS）=2×LV-HV
    """
    _mike = pd.DataFrame()
    _mike['date'] = df.date
    typ = df[['high', 'low', 'close']].mean(axis=1)
    hv = df.high.rolling(n).max()
    lv = df.low.rolling(n).min()
    _mike['wr'] = typ * 2 - lv
    _mike['mr'] = typ + hv - lv
    _mike['sr'] = 2 * hv - lv
    _mike['ws'] = typ * 2 - hv
    _mike['ms'] = typ - hv + lv
    _mike['ss'] = 2 * lv - hv
    return _mike


def vma(df, n=5):
    """
    量简单移动平均	VMA(5)	VMA=MA(volume,N)
    VOLUME表示成交量；N表示天数
    """
    _vma = pd.DataFrame()
    _vma['date'] = df.date
    _vma['vma'] = df.volume.rolling(n).mean()
    return _vma


def _ema_n(arr, n):
    return arr.ewm(ignore_na=False, span=n, min_periods=0, adjust=False).mean()


def vmacd(df, qn=12, sn=26, m=9):
    """
    量指数平滑异同平均	vmacd(12,26,9)
    今日EMA（N）=2/（N+1）×今日成交量+(N-1)/（N+1）×昨日EMA（N）
    DIFF= EMA（N1）- EMA（N2）
    DEA(DIF,M)= 2/(M+1)×DIF +[1-2/(M+1)]×DEA(REF(DIF,1),M)
    MACD（BAR）=2×（DIF-DEA）
    """
    _vmacd = pd.DataFrame()
    _vmacd['date'] = df.date
    _vmacd['diff'] = _ema_n(df.volume, qn) - _ema_n(df.volume, sn)
    _vmacd['dea'] = _ema_n(_vmacd['diff'], m)  # TODO: 不能用_vmacd.diff, 不知道为什么
    _vmacd['macd'] = (_vmacd['diff'] - _vmacd['dea'])
    return _vmacd


def vosc(df, n=12, m=26):
    """
    成交量震荡	vosc(12,26)
    VOSC=（MA（VOLUME,SHORT）- MA（VOLUME,LONG））/MA（VOLUME,SHORT）×100
    """
    _c = pd.DataFrame()
    _c['date'] = df['date']
    _c['osc'] = (df.volume.rolling(n).mean() - df.volume.rolling(m).mean()) / df.volume.rolling(n).mean() * 100
    return _c


def tapi(df, n=6):
    """
    加权指数成交值	tapi(6)
    TAPI=每日成交总值/当日加权指数=a/PI；A表示每日的成交金额，PI表示当天的股价指数即指收盘价
    """


def vstd(df, n=10):
    """
    成交量标准差	vstd(10)
    VSTD=STD（Volume,N）=[∑（Volume-MA(Volume，N)）^2/N]^0.5
    """
    _vstd = pd.DataFrame()
    _vstd['date'] = df.date
    _vstd['vstd'] = df.volume.rolling(n).std(ddof=1)
    return _vstd


def adtm(df, n=23, m=8):
    """
    动态买卖气指标	adtm(23,8)
    如果开盘价≤昨日开盘价，DTM=0
    如果开盘价＞昨日开盘价，DTM=(最高价-开盘价)和(开盘价-昨日开盘价)的较大值
    如果开盘价≥昨日开盘价，DBM=0
    如果开盘价＜昨日开盘价，DBM=(开盘价-最低价)
    STM=DTM在N日内的和
    SBM=DBM在N日内的和
    如果STM > SBM,ADTM=(STM-SBM)/STM
    如果STM < SBM , ADTM = (STM-SBM)/SBM
    如果STM = SBM,ADTM=0
    ADTMMA=MA(ADTM,M)
    """
    _adtm = pd.DataFrame()
    _adtm['date'] = df.date
    _m = pd.DataFrame()
    _m['cc'] = df.open - df.open.shift(1)
    _m['ho'] = df.high - df.open
    _m['ol'] = df.open - df.low
    _m['dtm'] = _m.apply(lambda x: max(x.ho, x.cc) if x.cc > 0 else 0, axis=1)
    _m['dbm'] = _m.apply(lambda x: x.ol if x.cc < 0 else 0, axis=1)
    _m['stm'] = _m.dtm.rolling(n).sum()
    _m['sbm'] = _m.dbm.rolling(n).sum()
    _m['ss'] = _m.stm - _m.sbm
    _adtm['adtm'] = _m.apply(lambda x: x.ss / x.stm if x.ss > 0 else (x.ss / x.sbm if x.ss < 0 else 0), axis=1)
    _adtm['adtmma'] = _adtm.adtm.rolling(m).mean()
    return _adtm


def mi(df, n=12):
    """
    动量指标	mi(12)
    A=CLOSE-REF(CLOSE,N)
    MI=SMA(A,N,1)
    """
    _mi = pd.DataFrame()
    _mi['date'] = df.date
    a = df.close - df.close.shift(n)
    _mi['mi'] = sma(a, n, 1)
    return _mi


def micd(df, n=3, m=10, k=20):
    """
    异同离差动力指数	micd(3,10,20)
    MI=CLOSE-ref(CLOSE,1)AMI=SMA(MI,N1,1)
    DIF=MA(ref(AMI,1),N2)-MA(ref(AMI,1),N3)
    MICD=SMA(DIF,10,1)
    """
    _micd = pd.DataFrame()
    _micd['date'] = df.date
    mi = df.close - df.close.shift(1)
    ami = pd.Series(sma(mi, n, 1))
    dif = ami.shift(1).rolling(m).mean() - ami.shift(1).rolling(k).mean()
    _micd['micd'] = sma(dif, 10, 1)
    return _micd


def rc(df, n=50):
    """
    变化率指数	rc(50)
    RC=收盘价/REF（收盘价，N）×100
    ARC=EMA（REF（RC，1），N，1）
    """
    _rc = pd.DataFrame()
    _rc['date'] = df.date
    _rc['rc'] = df.close / df.close.shift(n) * 100
    _rc['arc'] = sma(_rc.rc.shift(1), n, 1)
    return _rc


def rccd(df, n=59, m=21, k=28):
    """  # TODO: 计算结果错误，稍后检查
    异同离差变化率指数	rccd(59,21,28)
    RC=收盘价/REF（收盘价，N）×100%
    ARC=EMA(REF(RC,1),N,1)
    DIF=MA(ref(ARC,1),N1)-MA MA(ref(ARC,1),N2)
    RCCD=SMA(DIF,N,1)
    """
    _rccd = pd.DataFrame()
    _rccd['date'] = df.date
    rc = df.close / df.close.shift(n) * 100
    arc = pd.Series(sma(rc.shift(1), n, 1))
    dif = arc.shift(1).rolling(m).mean() - arc.shift(1).rolling(k).mean()
    _rccd['rc'] = rc
    _rccd['arc'] = arc
    _rccd['dif'] = dif
    _rccd['rccd'] = sma(dif, n, 1)
    return _rccd


def srmi(df, n=9):
    """
    SRMIMI修正指标	srmi(9)
    如果收盘价>N日前的收盘价，SRMI就等于（收盘价-N日前的收盘价）/收盘价
    如果收盘价<N日前的收盘价，SRMI就等于（收盘价-N日签的收盘价）/N日前的收盘价
    如果收盘价=N日前的收盘价，SRMI就等于0
    """
    _srmi = pd.DataFrame()
    _srmi['date'] = df.date
    _m = pd.DataFrame()
    _m['close'] = df.close
    _m['cp'] = df.close.shift(n)
    _m['cs'] = df.close - df.close.shift(n)
    _srmi['srmi'] = _m.apply(lambda x: x.cs/x.close if x.cs > 0 else (x.cs/x.cp if x.cs < 0 else 0), axis=1)
    return _srmi


def dptb(df, n=7):
    """
    大盘同步指标	dptb(7)
    DPTB=（统计N天中个股收盘价>开盘价，且指数收盘价>开盘价的天数或者个股收盘价<开盘价，且指数收盘价<开盘价）/N
    """
    ind = ts.get_k_data("sh000001", start=df.date.iloc[0], end=df.date.iloc[-1])
    sd = df.copy()
    sd.set_index('date', inplace=True)  # 可能出现停盘等情况，所以将date设为index
    ind.set_index('date', inplace=True)
    _dptb = pd.DataFrame(index=data.date)
    q = ind.close - ind.open
    _dptb['p'] = sd.close - sd.open
    _dptb['q'] = q
    _dptb['m'] = _dptb.apply(lambda x: 1 if (x.p > 0 and x.q > 0) or (x.p < 0 and x.q < 0) else np.nan, axis=1)
    _dptb['jdrs'] = _dptb.m.rolling(n).count() / n
    _dptb.drop(columns=['p', 'q', 'm'], inplace=True)
    _dptb.reset_index(inplace=True)
    return _dptb


def jdqs(df, n=20):
    """
    阶段强势指标	jdqs(20)
    JDQS=（统计N天中个股收盘价>开盘价，且指数收盘价<开盘价的天数）/（统计N天中指数收盘价<开盘价的天数）
    """
    ind = ts.get_k_data("sh000001", start=df.date.iloc[0], end=df.date.iloc[-1])
    sd = df.copy()
    sd.set_index('date', inplace=True)   # 可能出现停盘等情况，所以将date设为index
    ind.set_index('date', inplace=True)
    _jdrs = pd.DataFrame(index=data.date)
    q = ind.close - ind.open
    _jdrs['p'] = sd.close - sd.open
    _jdrs['q'] = q
    _jdrs['m'] = _jdrs.apply(lambda x: 1 if (x.p > 0 and x.q < 0) else np.nan, axis=1)
    q[q > 0] = np.nan
    _jdrs['t'] = q
    _jdrs['jdrs'] = _jdrs.m.rolling(n).count() / _jdrs.t.rolling(n).count()
    _jdrs.drop(columns=['p', 'q', 'm', 't'], inplace=True)
    _jdrs.reset_index(inplace=True)
    return _jdrs


def jdrs(df, n=20):
    """
    阶段弱势指标	jdrs(20)
    JDRS=（统计N天中个股收盘价<开盘价，且指数收盘价>开盘价的天数）/（统计N天中指数收盘价>开盘价的天数）
    """
    ind = ts.get_k_data("sh000001", start=df.date.iloc[0], end=df.date.iloc[-1])
    sd = df.copy()
    sd.set_index('date', inplace=True)
    ind.set_index('date', inplace=True)
    _jdrs = pd.DataFrame(index=data.date)
    q = ind.close - ind.open
    _jdrs['p'] = sd.close - sd.open
    _jdrs['q'] = q
    _jdrs['m'] = _jdrs.apply(lambda x: 1 if (x.p < 0 and x.q > 0) else np.nan, axis=1)
    q[q < 0] = np.nan
    _jdrs['t'] = q
    _jdrs['jdrs'] = _jdrs.m.rolling(n).count() / _jdrs.t.rolling(n).count()
    _jdrs.drop(columns=['p', 'q', 'm', 't'], inplace=True)
    _jdrs.reset_index(inplace=True)
    return _jdrs


def zdzb(df, n=125, m=5, k=20):
    """
    筑底指标	zdzb(125,5,20)
    A=（统计N1日内收盘价>=前收盘价的天数）/（统计N1日内收盘价<前收盘价的天数）
    B=MA（A,N2）
    D=MA（A，N3）
    """
    _zdzb = pd.DataFrame()
    _zdzb['date'] = df.date
    p = df.close - df.close.shift(1)
    q = p.copy()
    p[p < 0] = np.nan
    q[q >= 0] = np.nan
    _zdzb['a'] = p.rolling(n).count() / q.rolling(n).count()
    _zdzb['b'] = _zdzb.a.rolling(m).mean()
    _zdzb['d'] = _zdzb.a.rolling(k).mean()
    return _zdzb


def atr(df, n=14):
    """
    真实波幅	atr(14)
    TR:MAX(MAX((HIGH-LOW),ABS(REF(CLOSE,1)-HIGH)),ABS(REF(CLOSE,1)-LOW))
    ATR:MA(TR,N)
    """
    _atr = pd.DataFrame()
    _atr['date'] = df.date
    # _atr['tr'] = np.maximum(df.high - df.low, (df.close.shift(1) - df.low).abs())
    # _atr['tr'] = np.maximum.reduce([df.high - df.low, (df.close.shift(1) - df.high).abs(), (df.close.shift(1) - df.low).abs()])
    _atr['tr'] = np.vstack([df.high - df.low, (df.close.shift(1) - df.high).abs(), (df.close.shift(1) - df.low).abs()]).max(axis=0)
    _atr['atr'] = _atr.tr.rolling(n).mean()
    return _atr


def mass(df, n=9, m=25):
    """
    梅丝线	mass(9,25)
    AHL=MA(（H-L）,N1)
    BHL= MA（AHL，N1）
    MASS=SUM（AHL/BHL，N2）
    H：表示最高价；L：表示最低价
    """
    _mass = pd.DataFrame()
    _mass['date'] = df.date
    ahl = (df.high - df.low).rolling(n).mean()
    bhl = ahl.rolling(n).mean()
    _mass['mass'] = (ahl / bhl).rolling(m).sum()
    return _mass


def vhf(df, n=28):
    """
    纵横指标	vhf(28)
    VHF=（N日内最大收盘价与N日内最小收盘价之前的差）/（N日收盘价与前收盘价差的绝对值之和）
    """
    _vhf = pd.DataFrame()
    _vhf['date'] = df.date
    _vhf['vhf'] = (df.close.rolling(n).max() - df.close.rolling(n).min()) / (df.close - df.close.shift(1)).abs().rolling(n).sum()
    return _vhf


def cvlt(df, n=10):
    """
    佳庆离散指标	cvlt(10)
    cvlt=（最高价与最低价的差的指数移动平均-前N日的最高价与最低价的差的指数移动平均）/前N日的最高价与最低价的差的指数移动平均
    """
    _cvlt = pd.DataFrame()
    _cvlt['date'] = df.date
    p = _ema_n(df.high.shift(n) - df.low.shift(n), n)
    _cvlt['cvlt'] = (_ema_n(df.high - df.low, n) - p) / p * 100
    return _cvlt


def up_n(df):
    """
    连涨天数	up_n	连续上涨天数，当天收盘价大于开盘价即为上涨一天 # 同花顺实际结果用收盘价-前一天收盘价
    """
    _up = pd.DataFrame()
    _up['date'] = df.date
    p = df.close - df.close.shift()
    p[p > 0] = 1
    p[p < 0] = 0
    m = []
    for k, g in itertools.groupby(p):
        t = 0
        for i in g:
            if k == 0:
                m.append(0)
            else:
                t += 1
                m.append(t)
    # _up['p'] = p
    _up['up'] = m
    return _up


def down_n(df):
    """
    连跌天数	down_n	连续下跌天数，当天收盘价小于开盘价即为下跌一天
    """
    _down = pd.DataFrame()
    _down['date'] = df.date
    p = df.close - df.close.shift()
    p[p > 0] = 0
    p[p < 0] = 1
    m = []
    for k, g in itertools.groupby(p):
        t = 0
        for i in g:
            if k == 0:
                m.append(0)
            else:
                t += 1
                m.append(t)
    _down['down'] = m
    return _down


def join_frame(d1, d2, column='date'):
    # 将两个DataFrame 按照datetime合并
    return d1.join(d2.set_index(column), on=column)


if __name__ == "__main__":
    import tushare as ts
    data = ts.get_k_data("000063", start="2018-03-01")
    # data = ts.get_k_data("601138", start="2017-05-01")

    # maf = ma(data, n=[5, 10, 20])
    # 将均线合并到data中
    # print(join_frame(data, maf))

    # data = pd.DataFrame({"close": [1,2,3,4,5,6,7,8,9,0]})
    # print(ma(data))
    # mdf = md(data)
    # print(md(data, n=26))
    # print(join_frame(data, mdf))
    # emaf = ema(data)
    # print(ema(data, 5))
    # print(join_frame(data, emaf))
    # print(macd(data))
    # print(kdj(data))
    # print(vrsi(data, 6))
    # print(boll(data))
    # print(bbiboll(data))
    # print(join_frame(data, wr(data, n=14)))
    # print(join_frame(data, bias(data, n=[6, 12, 24])))
    # print(join_frame(data, asi(data, n=[6, 12, 24])))
    # print(join_frame(data, vr_rate(data)))
    # print(join_frame(data, arbr(data)))
    # print(join_frame(data, dpo(data)))
    # print(join_frame(data, trix(data)))
    # print(join_frame(data, bbi(data)))
    # print(ts.top_list(date="2018-12-20"))
    # print(join_frame(data, mtm(data)))
    # print(join_frame(data, obv(data)))
    # print(join_frame(data, cci(data)))
    # print(join_frame(data, priceosc(data)))
    # print(dbcd(data))
    # print(roc(data))
    # print(vroc(data))
    # print(cr(data))
    # print(psy(data))
    # print(wad(data))
    # print(mfi(data))
    # print(pvt(data))
    # print(wvad(data))
    # print(cdp(data))
    # print(env(data))
    # print(mike(data))
    # print(vr(data))
    # print(vma(data))
    # print(vmacd(data))
    # print(vosc(data))
    # print(vstd(data))
    # print(adtm(data))
    # print(mi(data))
    # print(micd(data))
    # print(rc(data))
    # print(rccd(data))
    # print(srmi(data))
    # print(dptb(data))
    # print(jdqs(data))
    # pd.set_option('display.max_rows', 1000)
    # print(jdrs(data))
    print(join_frame(data, jdrs(data)))
    # print(data)
    # print(zdzb(data))
    # print(atr(data))
    # print(mass(data))
    # print(vhf(data))
    # print(cvlt(data))
    # print(up_n(data))
    # print(down_n(data))
