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
import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder, MinMaxScaler

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required = True)
args = parser.parse_args()

sale = pd.read_excel(os.path.join('..', '..', '0.Data', '01_제공데이터', '2020 빅콘테스트 데이터분석분야-챔피언리그_2019년 실적데이터_v1_200818.xlsx'), skiprows = 1)
sale = sale.loc[(sale['상품군'] != '무형') & (sale['취급액'].notnull())]

meta_train = pd.read_excel(os.path.join('..', '..', '0.Data', '01_제공데이터', 'train수작업_meta.xlsx'))
meta_test = pd.read_excel(os.path.join('..', '..', '0.Data', '02_평가데이터', 'test수작업_meta.xlsx'))
meta_train = meta_train.loc[meta_train['상품군'] != '무형']
meta_test = meta_test.loc[meta_test['상품군'] != '무형']
meta = pd.concat([meta_train, meta_test], axis = 0)
item = meta.loc[meta['NEW상품명'].notnull(), ['NEW상품명', '상품군', '마더코드']].drop_duplicates('NEW상품명').reset_index(drop = True).reset_index().rename(columns = {'index' : 'NEW상품코드'})
item2 = item[['NEW상품코드', 'NEW상품명', '상품군', '마더코드']]

sale['방송날'] = sale['방송일시'].dt.date
sale['방송월'] = sale['방송일시'].dt.month
sale['방송일'] = sale['방송일시'].dt.day
sale['방송시간(시간)'] = sale['방송일시'].dt.hour
sale['방송일시'] = sale.apply(lambda x : datetime.datetime.combine(x['방송날'], datetime.time(x['방송시간(시간)'])), axis = 1)

sale['판매량'] = sale['취급액'] / sale['판매단가']
sale['판매량'] = sale['판매량'].fillna(0).apply(lambda x : math.ceil(x))

data = joblib.load(os.path.join('..', '..', '0.Data', '04_임시데이터', 'recommend_candidate.pkl'))
locals().update(data)


user_item_based = {}
for u in user_based.keys():
    user_item_based[u] = []
    for n in user_based[u]:
        v = raw.loc[n].sort_values(ascending = False).index[0]
        res = item_based[v]
        user_item_based[u].extend(res)
        
        
us = users[['user', 'segment']]
candidate = {}
for i, s in tqdm(us.values):
    try:
        c1 = user_item_based[i]
        c2 = user_content[i]
        c3 = rec1[s]
        c4 = rec2[s]
    except:
        continue
    candidate[i]= list(set(c1 + c2 + c3 + c4))

final = {}
for k in candidate.keys():
    v = list(item2.loc[item2['마더코드'].isin(candidate[k]), 'NEW상품코드'].values)
    final[k] = v

total = pd.DataFrame()
for k in tqdm(final.keys()):
    mon = int(k.split('-')[0])
    hour = int(k.split('-')[1])
    
    d = datetime.date(2020, mon, 1)
    t = datetime.time(hour)
    
    dt = datetime.datetime.combine(d, t)
    
    temp = pd.DataFrame(final[k])
    temp['방송일시'] = dt
    total = pd.concat([total, temp])
total = total.rename(columns = {0 : 'NEW상품코드'})

meta_train = pd.read_excel(os.path.join('..', '..', '0.Data', '01_제공데이터', 'train수작업_meta.xlsx'))
meta_test = pd.read_excel(os.path.join('..', '..', '0.Data', '02_평가데이터', 'test수작업_meta.xlsx'))
meta_train = meta_train.loc[meta_train['상품군'] != '무형']
meta_test = meta_test.loc[meta_test['상품군'] != '무형']
meta = pd.concat([meta_train, meta_test], axis = 0)

item = meta.loc[meta['NEW상품명'].notnull(), ['NEW상품명', '상품군', '마더코드']].drop_duplicates('NEW상품명').reset_index(drop = True).reset_index().rename(columns = {'index' : 'NEW상품코드'})

meta = meta.merge(item[['NEW상품명', 'NEW상품코드']], on = 'NEW상품명', how = 'left').drop_duplicates('NEW상품명')

res = total.merge(meta, on = 'NEW상품코드', how = 'left')[['방송일시', '마더코드', '상품코드', '상품명', '상품군', '모델명', '상품명다시', '판매단가', '결제방법', '브랜드', 'NEW상품명', '성별','단위', '옵션','NS카테고리']]
res = res.sort_values(['방송일시'])
res["노출(분)"] = 20
res = res[['방송일시', '노출(분)', '마더코드', '상품코드', '상품명', '상품군', '판매단가', '결제방법', '브랜드', '모델명', '상품명다시', 'NEW상품명', '성별','단위', '옵션','NS카테고리']]
res['취급액'] = None

res = res.reset_index(drop = True)
res.to_excel(os.path.join('..', '..', '0.Data', '01_제공데이터', 'data4recommend.xlsx'), index = False)
    
    
if args.dataset == 'train':
    ## ======== 각 상품군별 시계열 Feature 생성 ========
    print('========== Start TimeSeries FE preprocessing for Recommend==========')
    def makeTimeFE(piv, types):
        if types == 'day':
            pivT = piv.T
            ema_s = pivT.ewm(span=4).mean()
            ema_m = pivT.ewm(span=12).mean()
            ema_l = pivT.ewm(span=26).mean()
            macd = ema_s - ema_l
            sig = macd.ewm(span=9).mean()
            rol14 = pivT.fillna(0).rolling(14).mean()
            rol30 = pivT.fillna(0).rolling(30).mean()
        elif types == 'hour':
            pivT = piv.T
            ema_s = pivT.ewm(span=1).mean()
            ema_m = pivT.ewm(span=3).mean()
            ema_l = pivT.ewm(span=6).mean()
            macd = ema_s - ema_l
            sig = macd.ewm(span=9).mean()
            rol14 = pivT.fillna(0).rolling(1).mean()
            rol30 = pivT.fillna(0).rolling(3).mean()

        timeFE = pd.concat([ema_s, ema_m, ema_l, macd, sig, rol14, rol30], axis = 1).reset_index()

        return timeFE

    timeS = joblib.load(os.path.join('..', '..', '0.Data', '04_임시데이터', 'data4time.pkl'))

    piv = pd.pivot_table(sale, index = '상품군', columns = '방송일시', values = '판매량', aggfunc='sum')
    timeA = makeTimeFE(piv, 'day')

    piv = pd.pivot_table(sale, index = '상품군', columns = '방송시간(시간)', values = '판매량', aggfunc=np.mean)
    timeB = makeTimeFE(piv, 'hour')

    piv = pd.pivot_table(sale, index = '상품군', columns = '방송일시', values = '취급액', aggfunc='sum')
    timeC = makeTimeFE(piv, 'day')

    piv = pd.pivot_table(sale, index = '상품군', columns = '방송시간(시간)', values = '취급액', aggfunc='sum')
    timeD = makeTimeFE(piv, 'hour')

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

    scale_timeS = scale_timeS.drop(365)
    
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

    count = pd.pivot_table(sale, index = '방송시간(시간)', columns = '상품군', values = '판매량', aggfunc = 'count').reset_index()
    count.columns = ['방송시간(시간)'] + list(map(lambda x : 'count_' + x, count.columns[1:]))

    hour = pd.pivot_table(sale, index = '방송일시', columns = '상품군', values = '판매량', aggfunc = 'count').reset_index()
    hour.columns = ['방송일시'] + list(map(lambda x : 'hour_' + x, hour.columns[1:]))
    hour['방송일'] = hour['방송일시'].dt.day
    hour['방송월'] = hour['방송일시'].dt.month
    hour['방송시간(시간)'] = hour['방송일시'].dt.hour



    joblib.dump({
        'volume_v1' : volume_v1,
        'volume_v2' : volume_v2,
        'volume_v3' : count,
        'volume_v4' : hour
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

elif args.dataset == 'test':
    pass

def prep4WnD(data, label = None):
    data[['방송월', '방송시간(시간)', '방송시간(분)']] = data[['방송월', '방송시간(시간)', '방송시간(분)']].astype(int)

    data =  data.merge(scale_timeS,  on = ['방송월', '방송일'], how = 'left').fillna(0)

    data =  data.merge(scale_timeY,  on = ['방송월', '방송일', '방송시간(시간)'], how = 'left').fillna(0)

    data =  data.merge(scale_timeR,  on = ['방송시간(시간)'], how = 'left').fillna(0)

    data = data.merge(volume_v3, on = '방송시간(시간)', how = 'left')

    data = data.merge(volume_v4, on = ['방송월', '방송일', '방송시간(시간)'], how = 'left')

    data = data.fillna(0)

    data = data.merge(rate_v1[['방송월', '방송일', '일별평균시청률']], on = ['방송월', '방송일'], how = 'left')
    data['일별시간별최대시청률'] = None
    data['일별시간별평균시청률'] = None
    data['일별시간별중간시청률'] = None

    for m, d, h in tqdm(data[['방송월', '방송일', '방송시간(시간)']].drop_duplicates().values):
        max_r = rate_v3.loc[(rate_v3['방송월'] == m) & (rate_v3['방송일'] == d), h].values[0]
        min_r = rate_v4.loc[(rate_v4['방송월'] == m) & (rate_v4['방송일'] == d), h].values[0]
        med_r = rate_v5.loc[(rate_v5['방송월'] == m) & (rate_v5['방송일'] == d), h].values[0]

        data.loc[(data['방송월'] == m) & (data['방송일'] == d) & (data['방송시간(시간)'] == h), ['일별시간별최대시청률', '일별시간별평균시청률', '일별시간별중간시청률']] = [max_r, min_r, med_r]

    data['시간별월별최대시청률'] = None
    data['시간별월별평균시청률'] = None
    data['시간별월별중간시청률'] = None

    for m,h in tqdm(data[['방송월', '방송시간(시간)']].drop_duplicates().values):
        max_r = rate_v6.loc[(rate_v6['방송시간(시간)'] == h), m].values[0]
        min_r = rate_v7.loc[(rate_v7['방송시간(시간)'] == h), m].values[0]
        med_r = rate_v8.loc[(rate_v8['방송시간(시간)'] == h), m].values[0]

        data.loc[(data['방송월'] == m) & (data['방송시간(시간)'] == h), ['시간별월별최대시청률', '시간별월별평균시청률', '시간별월별중간시청률']] = [max_r, min_r, med_r]

    data = data.merge(volume_v1, on = ['방송월', '방송시간(시간)'], how = 'left')

    data = data.merge(volume_v2, on = ['방송시간(시간)'], how = 'left')

    X = data[COLUMNS]
    for c in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c])

    if args.dataset == 'train':
        
        label = pd.get_dummies(label).values
        
        from sklearn.model_selection import train_test_split
        
        x_train, x_valid, y_train, y_valid = train_test_split(X, label, test_size = 0.2, random_state = 42)

        x_train_category = np.array(x_train[CATEGORICAL_COLUMNS])
        x_valid_category = np.array(x_valid[CATEGORICAL_COLUMNS])
        x_train_continue = np.array(x_train[CONTINUOUS_COLUMNS], dtype = 'float64')
        x_valid_continue = np.array(x_valid[CONTINUOUS_COLUMNS], dtype = 'float64')

        scaler = MinMaxScaler()
        x_train_continue = scaler.fit_transform(x_train_continue)
        x_valid_continue = scaler.transform(x_valid_continue)

        poly = PolynomialFeatures(degree=2, interaction_only=True)
        x_train_category_poly = poly.fit_transform(x_train_category)
        x_valid_category_poly = poly.transform(x_valid_category)
        
        joblib.dump(scaler, os.path.join('..', '..', '0.Data', '04_임시데이터', 'scaler4rec.pkl'))
        
        data4train = (x_train_continue, x_train_category, x_train_category_poly, y_train)
        data4valid = (x_valid_continue, x_valid_category, x_valid_category_poly, y_valid)
        return X, data4train, data4valid
    
    elif args.dataset == 'test':
        
        X_category = np.array(X[CATEGORICAL_COLUMNS])
        X_continue = np.array(X[CONTINUOUS_COLUMNS], dtype = 'float64')
        
        scaler = joblib.load(os.path.join('..', '..', '0.Data', '04_임시데이터', 'scaler4rec.pkl'))
        
        X_continue = scaler.fit_transform(X_continue)
        
        poly = PolynomialFeatures(degree=2, interaction_only=True)
        X_category_poly = poly.fit_transform(X_category)
        
        data4test = (X_continue, X_category, X_category_poly)
        
        return X, data4test
    
print('========== Start Feature Engineering ==========')
# 본 데이터 불러오기
print(f'{args.dataset} Data Load.....')

def preprocessing(df):
    df['방송날'] = df['방송일시'].dt.date    
    df['방송년도'] = df['방송일시'].dt.year
    df['방송월'] = df['방송일시'].dt.month
    df['방송일'] = df['방송일시'].dt.day
    df['방송시간(시간)'] = df['방송일시'].dt.hour
    df['방송시간(분)'] = df['방송일시'].dt.minute
    
    # 날씨 데이터 FE  
    w_19 = pd.read_csv(os.path.join('..', '..', '0.Data', '03_외부데이터', '2019_weather.csv'), encoding = 'cp949', dtype='unicode')
    w_20 = pd.read_csv(os.path.join('..', '..', '0.Data', '03_외부데이터', '2020_weather.csv'), encoding = 'cp949', dtype='unicode')
    df_wth = pd.concat([w_19, w_20], axis = 0)
    df_wth = FEex.preprocessing_weather(df_wth)

    # 미세먼지 데이터
    dust_2019 = pd.read_csv(os.path.join('..', '..', '0.Data', '03_외부데이터', '2019_dust.csv'), encoding = 'cp949')
    dust_2020 = pd.read_csv(os.path.join('..', '..', '0.Data', '03_외부데이터', '2020_dust.csv'), encoding = 'cp949')
    df_dust = pd.concat([dust_2019, dust_2020], axis = 0)
    df_dust = FEex.preprocessing_dust(df_dust, 'train')

    df = df.merge(df_wth, left_on = ['방송년도', '방송월', '방송일', '방송시간(시간)'], right_on = ['연도', '월', '일' ,'시간'], how = 'left').drop(['연도', '월', '일', '시간'], axis = 1)
    df = df.merge(df_dust, left_on = ['방송년도', '방송월', '방송일', '방송시간(시간)'], right_on = ['연도', '월', '일', '시간'], how ='left').drop(['연도', '월', '일', '시간'], axis = 1)
    
    # 경제 데이터
    df_eco = FEex.preprocessing_economy()
    df = df.merge(df_eco, left_on = ['방송년도', '방송월'], right_on = ['연도', '월'], how = 'left').drop(['연도', '월'], axis = 1)
    df = FEin.engineering_DatePrice(df, 'recommend')
    
    df = df[list(df.columns[:1]) + list(df.columns[8:14]) + list(df.columns[1:8]) + list(df.columns[14:])]
    
    return df

if args.dataset == 'train':
    train = joblib.load(os.path.join('..', '..', '0.Data', '04_임시데이터', 'train_data4WnD.pkl'))
    test = joblib.load(os.path.join('..', '..', '0.Data', '04_임시데이터', 'test_data4WnD.pkl'))
    
elif args.dataset == 'test':
    data = pd.read_excel(os.path.join('..', '..', '0.Data', '01_제공데이터', '2020 빅콘테스트 데이터분석분야-챔피언리그_방송편성표추천데이터.xlsx'))
    data = preprocessing(data)
else:
    print('dataset error.....')


# 내부데이터
volume4wnd = joblib.load(os.path.join('..', '..', '0.Data', '04_임시데이터', 'volume4wnd.pkl'))
locals().update(volume4wnd)
time4wnd = joblib.load(os.path.join('..', '..', '0.Data', '04_임시데이터', 'time4wnd.pkl'))
locals().update(time4wnd)
rate4wnd = joblib.load(os.path.join('..', '..', '0.Data', '04_임시데이터', 'rate4wnd.pkl'))
locals().update(rate4wnd)
    

COLUMNS = ['방송월',
                 '방송일',
                 '방송시간(시간)',
                 '경상지수',
                 '불변지수',
                 'pca_1',
                 'pca_2',
                 'pca_3',
                 'pca_4',
                 'pca_5',
                 '강수량(mm)_경기',
                 '강수량(mm)_광주',
                 '강수량(mm)_대구',
                 '강수량(mm)_대전',
                 '강수량(mm)_부산',
                 '강수량(mm)_서울',
                 '강수량(mm)_울산',
                 '강수량(mm)_인천',
                 '기온(°C)_경기',
                 '기온(°C)_광주',
                 '기온(°C)_대구',
                 '기온(°C)_대전',
                 '기온(°C)_부산',
                 '기온(°C)_서울',
                 '기온(°C)_울산',
                 '기온(°C)_인천',
                 '습도(%)_경기',
                 '습도(%)_광주',
                 '습도(%)_대구',
                 '습도(%)_대전',
                 '습도(%)_부산',
                 '습도(%)_서울',
                 '습도(%)_울산',
                 '습도(%)_인천',
                 '시정(10m)_경기',
                 '시정(10m)_광주',
                 '시정(10m)_대구',
                 '시정(10m)_대전',
                 '시정(10m)_부산',
                 '시정(10m)_서울',
                 '시정(10m)_울산',
                 '시정(10m)_인천',
                 '지면온도(°C)_경기',
                 '지면온도(°C)_광주',
                 '지면온도(°C)_대구',
                 '지면온도(°C)_대전',
                 '지면온도(°C)_부산',
                 '지면온도(°C)_서울',
                 '지면온도(°C)_울산',
                 '지면온도(°C)_인천',
                 '체감온도_경기',
                 '체감온도_광주',
                 '체감온도_대구',
                 '체감온도_대전',
                 '체감온도_부산',
                 '체감온도_서울',
                 '체감온도_울산',
                 '체감온도_인천',
                 '풍속(m/s)_경기',
                 '풍속(m/s)_광주',
                 '풍속(m/s)_대구',
                 '풍속(m/s)_대전',
                 '풍속(m/s)_부산',
                 '풍속(m/s)_서울',
                 '풍속(m/s)_울산',
                 '풍속(m/s)_인천',
                 '최고PM10_경기',
                 '최고PM10_광주',
                 '최고PM10_부산',
                 '최고PM10_서울',
                 '최고PM10_울산',
                 '최고PM10_인천',
                 '최고PM25_경기',
                 '최고PM25_광주',
                 '최고PM25_대구',
                 '최고PM25_대전',
                 '최고PM25_부산',
                 '최고PM25_서울',
                 '최고PM25_울산',
                 '최고PM25_인천',
                 '최저PM10_경기',
                 '최저PM10_광주',
                 '최저PM10_대구',
                 '최저PM10_대전',
                 '최저PM10_부산',
                 '최저PM10_서울',
                 '최저PM10_울산',
                 '최저PM10_인천',
                 '최저PM25_경기',
                 '최저PM25_광주',
                 '최저PM25_대구',
                 '최저PM25_대전',
                 '최저PM25_부산',
                 '최저PM25_서울',
                 '최저PM25_울산',
                 '최저PM25_인천',
                 '평균PM10_경기',
                 '평균PM10_광주',
                 '평균PM10_대구',
                 '평균PM10_대전',
                 '최고PM10_대구',
                 '최고PM10_대전',
                 '평균PM10_부산',
                 '평균PM10_서울',
                 '평균PM10_울산',
                 '평균PM10_인천',
                 '평균PM25_경기',
                 '평균PM25_광주',
                 '평균PM25_대구',
                 '평균PM25_대전',
                 '평균PM25_부산',
                 '평균PM25_서울',
                 '평균PM25_울산',
                 '평균PM25_인천',
                 'isHoliday',
                 '평일여부',
                 '방송시간대',
                 '계절',
                 '분기',
                 '일별평균시청률',
                 '일별시간별최대시청률',
                 '일별시간별평균시청률',
                 '일별시간별중간시청률',
                 '시간별월별최대시청률',
                 '시간별월별평균시청률',
                 '시간별월별중간시청률',
                 '월별시간별평균판매량',
                 '월별시간별중간판매량',
                 '월별시간별평균판매단가',
                 '월별시간별중간판매단가',
                 '시간별평균판매량',
                 '시간별중간판매량',
                 '시간별평균판매단가',
                 '시간별중간판매단가',
                 'count_가구',
                 'count_가전',
                 'count_건강기능',
                 'count_농수축',
                 'count_생활용품',
                 'count_속옷',
                 'count_의류',
                 'count_이미용',
                 'count_잡화',
                 'count_주방',
                 'count_침구',
                 'hour_가구',
                 'hour_가전',
                 'hour_건강기능',
                 'hour_농수축',
                 'hour_생활용품',
                 'hour_속옷',
                 'hour_의류',
                 'hour_이미용',
                 'hour_잡화',
                 'hour_주방',
                 'hour_침구',
                 'type1_0',
                 'type1_1',
                 'type1_2',
                 'type1_3',
                 'type1_4',
                 'type1_5',
                 'type1_6',
                 'type1_7',
                 'type1_8',
                 'type1_9',
                 'type1_10',
                 'type1_11',
                 'type1_12',
                 'type1_13',
                 'type1_14',
                 'type1_15',
                 'type1_16',
                 'type1_17',
                 'type1_18',
                 'type1_19',
                 'type1_20',
                 'type2_0',
         'type2_1',
         'type2_2',
         'type2_3',
         'type2_4',
         'type2_5',
         'type2_6',
         'type2_7',
         'type2_8',
         'type2_9',
         'type2_10',
         'type2_11',
         'type2_12',
         'type2_13',
         'type2_14',
         'type2_15',
         'type2_16',
         'type2_17',
         'type2_18',
         'type2_19',
         'type2_20',
         'type2_21',
         'type2_22',
         'type2_23',
         'type2_24',
         'type2_25',
         'type2_26',
         'type2_27',
         'type2_28',
         'type2_29',
         'type2_30',
         'type2_31',
         'type2_32',
         'type2_33',
         'type2_34',
         'type2_35',
         'type2_36',
         'type2_37',
         'type2_38',
         'type2_39',
         'type2_40',
         'type2_41',
         'type2_42',
         'type2_43',
                 'type3_0',
                 'type3_1',
                 'type3_2',
                 'type3_3',
                 'type3_4',
                 'type3_5',
                 'type3_6',
                 'type3_7',
                 'type3_8',
                 'type3_9',
                 'type3_10'
              ]
    
CATEGORICAL_COLUMNS = ['isHoliday', '평일여부', '방송시간대', '계절', '분기']
CONTINUOUS_COLUMNS = list(set(COLUMNS) - set(CATEGORICAL_COLUMNS))

if args.dataset == 'train':
    data = pd.concat([train['X'], test['X']], axis = 0).reset_index(drop = True)
    label = pd.concat([train['label'], test['label']], axis = 0)
    
    X, data4train, data4valid = prep4WnD(data, label)
    
    joblib.dump({
        'X' : X,
        'data4train' : data4train,
        'data4valid' : data4valid
    }, os.path.join('..', '..', '0.Data', '05_분석데이터', 'train_Rec.pkl'))
    
    print('Train Data saved!')

elif args.dataset == 'test':
    
    X, data4test = prep4WnD(data)
    
    joblib.dump({
        'X' : X,
        'data4test' : data4test,
    }, os.path.join('..', '..', '0.Data', '05_분석데이터', 'test_Rec.pkl'))

    print('Test Data saved!')