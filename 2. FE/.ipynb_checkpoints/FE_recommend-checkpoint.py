# -*- coding: utf-8 -*-

import os
import sys
import gc
import math
import joblib
sys.path.append(os.path.join('..', '2. FE'))
import FE_innData as FEin
import FE_extData as FEex

import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from datetime import datetime, time

sale = pd.read_excel(os.path.join('..', '..', '0.Data', '01_제공데이터', '2020 빅콘테스트 데이터분석분야-챔피언리그_2019년 실적데이터_v1_200818.xlsx'), skiprows = 1)

sale = sale.loc[(sale['상품군'] != '무형') & (sale['취급액'].notnull())]

sale['방송날'] = sale['방송일시'].dt.date
sale['방송월'] = sale['방송일시'].dt.month
sale['방송일'] = sale['방송일시'].dt.day
sale['방송시간(시간)'] = sale['방송일시'].dt.hour
sale['방송일시'] = sale.apply(lambda x : datetime.combine(x['방송날'], time(x['방송시간(시간)'])), axis = 1)

sale['판매량'] = sale['취급액'] / sale['판매단가']
sale['판매량'] = sale['판매량'].fillna(0).apply(lambda x : math.ceil(x))

## ======== 각 상품군별 시계열 Feature 생성 ========
print('========== Start TimeSeries FE preprocessing for Recommend==========')
def makeTimeFE(piv):
    pivT = piv.T
    ema_s = pivT.ewm(span=4).mean()
    ema_m = pivT.ewm(span=12).mean()
    ema_l = pivT.ewm(span=26).mean()
    macd = ema_s - ema_l
    sig = macd.ewm(span=9).mean()
    rol14 = pivT.fillna(0).rolling(14).mean()
    rol30 = pivT.fillna(0).rolling(30).mean()
    
#     for tb, column in zip([ema_s, ema_m, ema_l, macd, sig, rol14, rol30], ['ema_s', 'ema_m', 'ema_l', 'macd', 'sig', 'rol14', 'rol30']):
#     new_columns = list(map(lambda x : '_'.join((column, x, types)), tb.columns))
#     tb.columns = new_columns
    
    timeFE = pd.concat([ema_s, ema_m, ema_l, macd, sig, rol14, rol30], axis = 1).reset_index()
    
    return timeFE

timeS = joblib.load(os.path.join('..', '..', '0.Data', '04_임시데이터', 'data4time.pkl'))

piv = pd.pivot_table(sale, index = '상품군', columns = '방송일시', values = '판매량', aggfunc='sum')
timeA = makeTimeFE(piv)

piv = pd.pivot_table(sale, index = '상품군', columns = '방송시간(시간)', values = '판매량', aggfunc=np.mean)
timeB = makeTimeFE(piv)

piv = pd.pivot_table(sale, index = '상품군', columns = '방송일시', values = '취급액', aggfunc='sum')
timeC = makeTimeFE(piv)

piv = pd.pivot_table(sale, index = '상품군', columns = '방송시간(시간)', values = '취급액', aggfunc='sum')
timeD = makeTimeFE(piv)

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def getPCA(date, df, types):
    pca = PCA(n_components = 0.95) 
    scaler = MinMaxScaler()
    raw = pd.DataFrame(scaler.fit_transform(df)).fillna(0)
    
    df = pd.DataFrame(pca.fit_transform(raw))
    df = pd.concat([date, df], axis = 1)
    df.columns = [df.columns[0]] + list(map(lambda x : types + str(x), df.columns[1:]))
    
    return df

scale_timeS = getPCA(timeS[['방송날']], timeS.iloc[:, 1:], 'type1_')

time1 = timeA.merge(timeC, on = '방송일시', how = 'left')
scale_time1 = getPCA(time1[['방송일시']], time1.iloc[:, 1:], 'type2_')

time2 = timeB.merge(timeD, on = '방송시간(시간)', how = 'left')
scale_time2 = getPCA(time2[['방송시간(시간)']], time2.iloc[:, 1:], 'type3_')

scale_timeS['방송월'] = scale_timeS['방송날'].dt.month
scale_timeS['방송일'] = scale_timeS['방송날'].dt.day
scale_time1['방송월'] = scale_time1['방송일시'].dt.month
scale_time1['방송일'] = scale_time1['방송일시'].dt.day
scale_time1['방송시간(시간)'] = scale_time1['방송일시'].dt.hour

joblib.dump({
    'scale_timeS' : scale_timeS,
    'scale_timeY' : scale_time1,
    'scale_timeR' : scale_time2
}, os.path.join('..', '..', '0.Data', '04_임시데이터', 'time4wnd.pkl'))
print('>>>>>>>>>> Saved TimeSeries FE preprocessing for Recommend!! <<<<<<<<<<')

## ======== 각 상품군별 판매량 Feature 생성 ========
print('========== Start Volume FE preprocessing for Recommend==========')
volume_v1 = sale.groupby(['방송월', '방송시간(시간)']).agg({
    '판매량' : [('월별시간별평균판매량', np.mean),
            ('월별시간별중간판매량', np.median)],
    '판매단가' : [('월별시간별평균판매단가', np.mean),
             ('월별시간별중간판매단가', np.median)]
}).reset_index()

volume_v2 = sale.groupby(['방송시간(시간)']).agg({
    '판매량' : [('시간별평균판매량', np.mean),
            ('시간별중간판매량', np.median)],
    '판매단가' : [('시간별평균판매단가', np.mean),
             ('시간별중간판매단가', np.median)]
}).reset_index()

def prepColumn(df):
    columns = []
    for i, c in enumerate(df.columns):
        if c[1] == '':
            columns.append(c[0])
        else:
            columns.append(c[1])
    df.columns = columns
    return df

volume_v1 = prepColumn(volume_v1)
volume_v2 = prepColumn(volume_v2)

joblib.dump({
    'volume_v1' : scale_timeS,
    'volume_v2' : scale_time1,
}, os.path.join('..', '..', '0.Data', '04_임시데이터', 'volume4wnd.pkl'))
print('>>>>>>>>>> Saved Volume FE preprocessing for Recommend!! <<<<<<<<<<')


## ======== 시청률 Feature 생성 ========
print('========== Start Rating FE preprocessing for Recommend==========')
rate = pd.read_excel(os.path.join('..', '..', '0.Data', '01_제공데이터', '2020 빅콘테스트 데이터분석분야-챔피언리그_시청률 데이터.xlsx'), skiprows = 1)

rate_v1 = rate.loc[[1440]].T.reset_index().loc[1:365].rename(columns = {'index' : '방송일시', 1440 : '일별평균시청률'})

rate_v2 = rate.iloc[:, -1].reset_index().rename(columns=  {'index' : '시간대', '2019-01-01 to 2019-12-31'  : '분당평균시청률'})
rate_v2['시간대'] = rate['시간대']
rate_v2 = rate_v2.drop(1440)

rate['방송시간(시간)'] = rate['시간대'].apply(lambda x: x[:2])

rate_v3 = rate.groupby('방송시간(시간)').max()
rate_v4 = rate.groupby('방송시간(시간)').mean()
rate_v5 = rate.groupby('방송시간(시간)').median()

rateT = rate.T.reset_index().iloc[:,:-1].loc[:365]
rateT.columns = rateT.loc[0]
rateT = rateT.drop(0)
rateT['방송월'] = pd.to_datetime(rateT['시간대']).dt.month
rateT = rateT.drop('시간대', axis = 1).fillna(0).groupby('방송월').mean().T.reset_index()
rateT['방송시간(시간)'] = rateT[0].apply(lambda x : x[:2])

rate_v3 = rate.groupby('방송시간(시간)').max()
rate_v4 = rate.groupby('방송시간(시간)').mean()
rate_v5 = rate.groupby('방송시간(시간)').median()

rateT = rate.T.reset_index().iloc[:,:-1].loc[:365]
rateT.columns = rateT.loc[0]
rateT = rateT.drop(0)
rateT['방송월'] = pd.to_datetime(rateT['시간대']).dt.month
rateT = rateT.drop('시간대', axis = 1).fillna(0).groupby('방송월').mean().T.reset_index()
rateT['방송시간(시간)'] = rateT[0].apply(lambda x : x[:2])
rate_v6 = rateT.drop(0, axis = 1).groupby('방송시간(시간)').max()
rate_v7 = rateT.drop(0, axis = 1).groupby('방송시간(시간)').mean()
rate_v8 = rateT.drop(0, axis = 1).groupby('방송시간(시간)').median()

rate_v1['방송월'] = pd.to_datetime(rate_v1['방송일시']).dt.month
rate_v1['방송일'] = pd.to_datetime(rate_v1['방송일시']).dt.day
rate_v3 = rate_v3.iloc[:, 1:-1].drop('월화').T # 일별시간별 최대시청률
rate_v4 = rate_v4.iloc[:, :-1].drop('월화').T # 일별시간별 평균시청률
rate_v5 = rate_v5.iloc[:, :-1].drop('월화').T # 일별시간별 중간시청률

rate_v3.columns = np.arange(0, 24)
rate_v4.columns = np.arange(0, 24)
rate_v5.columns = np.arange(0, 24)
rate_v3 = rate_v3.reset_index().rename(columns = {'index' : '방송일시'})
rate_v4 = rate_v4.reset_index().rename(columns = {'index' : '방송일시'})
rate_v5 = rate_v5.reset_index().rename(columns = {'index' : '방송일시'})
rate_v3['방송월'] = pd.to_datetime(rate_v3['방송일시']).dt.month
rate_v3['방송일'] = pd.to_datetime(rate_v3['방송일시']).dt.day
rate_v4['방송월'] = pd.to_datetime(rate_v4['방송일시']).dt.month
rate_v4['방송일'] = pd.to_datetime(rate_v5['방송일시']).dt.day
rate_v5['방송월'] = pd.to_datetime(rate_v5['방송일시']).dt.month
rate_v5['방송일'] = pd.to_datetime(rate_v5['방송일시']).dt.day
rate_v6 = rate_v6.reset_index(drop = True).reset_index().rename(columns = {'index' : '방송시간(시간)'})
rate_v7 = rate_v7.reset_index(drop = True).reset_index().rename(columns = {'index' : '방송시간(시간)'})
rate_v8 = rate_v8.reset_index(drop = True).reset_index().rename(columns = {'index' : '방송시간(시간)'})

joblib.dump({
    'rate_v1' : rate_v1,
    'rate_v2' : rate_v2,
    'rate_v3' : rate_v3,
    'rate_v4' : rate_v4,
    'rate_v5' : rate_v5,
    'rate_v6' : rate_v6,
    'rate_v7' : rate_v7,
    'rate_v8' : rate_v8,
}, os.path.join('..', '..', '0.Data', '04_임시데이터', 'rate4wnd.pkl'))

print('>>>>>>>>>> Saved Rating FE preprocessing for Recommend!! <<<<<<<<<<')