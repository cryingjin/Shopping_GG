# -*- coding: utf-8 -*-

import os
import gc
import random
import datetime
import numpy as np
import pandas as pd

def prepColumns(df):
    df.columns = list(map(lambda x : '_'.join(x), df.columns))
    return df

############## 날짜 데이터(출처 : 기상자료개방포털) ##############
def preprocessing_weather(df):
    df = df.loc[df['지점명'].str.contains('인천|울산|대구|대전|수원|부산|광주|서울')]
    df.loc[df['지점명'] == '수원', '지점명'] = '경기'
    df = df.reset_index(drop = True)
    
    # 변수 정리
    df = df[list(df.columns[(df.isnull().sum() / df.shape[0]) < 0.5]) + ['강수량(mm)']]
    df = df.drop(['풍향(16방위)', '증기압(hPa)', '이슬점온도(°C)', '현지기압(hPa)', '해면기압(hPa)',
                  '일조(hr)', '전운량(10분위)', '중하층운량(10분위)', '최저운고(100m )', '운형(운형약어)'], axis = 1)
    
    
    # 체감온도 : 13.12 + 0.6215T - 11.37V^0.16 + 0.3965V^0.16T (T : 기온(°C), V : 풍속(km/h))
    df.iloc[:, 3:] = df.iloc[:, 3:].fillna(0).astype(float)
    df['체감온도'] = df.apply(lambda x: 13.12 + 0.6215 * x['기온(°C)'] - 11.37 * (x['풍속(m/s)'] * 3.6)**0.16 + 0.3965 * (x['풍속(m/s)'] * 3.6)**0.16 * x['기온(°C)'], axis = 1)
    df.iloc[:, 3:] = df.iloc[:, 3:].fillna(0).astype(float)

    df = pd.pivot_table(df, index = '일시', columns = '지점명')
    df = prepColumns(df).reset_index()
    # merge를 위한 날짜 생성 
    df['일시'] = pd.to_datetime(df['일시'])
    df['연도'] = df['일시'].dt.year
    df['월'] = df['일시'].dt.month
    df['일'] = df['일시'].dt.day
    df['시간'] = df['일시'].dt.hour
    df = df.sort_values('일시')
    df = df.drop('일시', axis = 1)
    return df

############## 미세먼지(PM10, PM25) 데이터(출처 : 에어코리아) ##############
def preprocessing_dust(df):
    temp = df.loc[df['지역'].str.contains('서울|경기|인천|부산|울산|대구|대전|광주')]
    del df
    gc.collect()
    value = temp['지역'].apply(lambda x : x[:2]).values
    df = temp.copy()
    df['지역'] = value
    del temp
    gc.collect()

    prep_df = df.groupby(['지역', '측정일시']).agg({
        'PM10' : [('최고PM10', np.max), ('최저PM10', np.min), ('평균PM10', np.mean)],
        'PM25' : [('최고PM25', np.max), ('최저PM25', np.min), ('평균PM25', np.mean)]
    }).reset_index()
    
    prep_df.columns = prep_df.columns = ['지역', '측정일시', '최고PM10', '최저PM10', '평균PM10', '최고PM25', '최저PM25', '평균PM25']
    
    prep_df['측정일시'] = prep_df['측정일시'].astype(str).apply(lambda x : x[:8] + '00' if x[8:] == '24' else x)
    prep_df['측정일시'] = pd.to_datetime(prep_df['측정일시'].astype(str).apply(lambda x : '-'.join((x[:4], x[4:6], x[6:8])) + ' '+ x[8:] + ':00:00'))
    prep_df.loc[prep_df['측정일시'].dt.hour == 0, '측정일시'] = prep_df.loc[prep_df['측정일시'].dt.hour == 0, '측정일시'] + datetime.timedelta(days = 1)
    
    del df
    gc.collect()
    
    prep_df = pd.pivot_table(prep_df, index = '측정일시', columns = '지역')
    prep_df = prepColumns(prep_df).reset_index()
    
    # merge를 위한 날짜 생성
    prep_df['측정일시'] = pd.to_datetime(prep_df['측정일시'])
    prep_df['연도'] = prep_df['측정일시'].dt.year
    prep_df['월'] = prep_df['측정일시'].dt.month
    prep_df['일'] = prep_df['측정일시'].dt.day
    prep_df['시간'] = prep_df['측정일시'].dt.hour
    prep_df = prep_df.drop('측정일시', axis = 1)
    return prep_df

############## 경제 데이터 ##############
def preprocessing_economy():
    df1 = pd.read_excel(os.path.join('..', '..', '0.Data', '03_외부데이터', '소매업태별 판매액지수.xlsx')) # 출처 : KOSIS
    df2 = pd.read_csv(os.path.join('..', '..', '0.Data', '03_외부데이터', '소비자동향조사 전국.csv'), encoding = 'cp949') # 출처 : KOSIS
    df3 = pd.read_excel(os.path.join('..', '..', '0.Data', '03_외부데이터', '온라인쇼핑몰 판매매체별 상품군별거래액.xlsx')) # 출처 : KOSIS
    df4 = pd.read_excel(os.path.join('..', '..', '0.Data', '03_외부데이터', '지역별 소비유형별 개인 신용카드.xlsx')) # 출처 : 한국은행
    
    # df1 정제
    df1['업태별'] = df1['업태별'].apply(lambda x : x.strip())
    df1.columns = ['업태별'] + list(map(lambda x : x.replace(' ','')[:7], df1.columns[1:]))
    
    t = df1.loc[[0,19]].T.drop('업태별').reset_index()
    t = t.loc[t[19] != '-']
    t[19] = t[19].astype(float)
    df1 = pd.pivot_table(t, index = 'index', columns = 0, values = 19)
    df1.columns = ['경상지수', '불변지수']
    df1.index.name = None
    
    # df2 정제
    selected = ['현재생활형편CSI',
                '현재경기판단CSI',
                '생활형편전망CSI',
                '소비지출전망CSI',
                '주택가격전망CSI',
                '임금수준전망CSI',
                '소비자심리지수']
    
    df2 = df2.loc[df2['분류코드별'] == '전체'].reset_index(drop = True)
    df2.index = df2['지수코드별']
    df2 = df2.T.drop(['지수코드별', '분류코드별', '항목', '단위'])
    df2.columns = list(map(lambda x : x.strip(), df2.columns))
    df2 = df2[selected]
    
    # df3 정제
    df3 = df3.loc[df3['판매매체별'] == '계'].T
    df3.columns = df3.loc['상품군별']
    df3 = df3.drop(['상품군별', '판매매체별'])
    df3.index= list(map(lambda x : x.replace(' ', '')[:7], df3.index))
    
    # df4 정제
    regional_consumed = pd.pivot_table(df4, index = 'TIME', columns = 'ITEM_NAME1', values = 'DATA_VALUE')
    categorical_consumed = pd.pivot_table(df4, index = 'TIME', columns = 'ITEM_NAME2', values = 'DATA_VALUE')
    categorical_consumed.index.name = None; regional_consumed.index.name = None
    
    
    def makeDate(df):
        df = df.reset_index().rename(columns = {'index' : '날짜'})
        if type(df['날짜'][0]) == str:
            try:
                df['날짜'] = pd.to_datetime(df['날짜'])
            except:
                df['날짜'] = df['날짜'].apply(lambda x : x.replace('월', ''))
                df['날짜'] = pd.to_datetime(df['날짜'].apply(lambda x : x.replace('월', '')))
        else:
            df['날짜'] = df['날짜'].astype(str).apply(lambda x : '-'.join((x[:4],x[4:])))
            df['날짜'] = pd.to_datetime(df['날짜'])
            
        return df
    df1 = makeDate(df1)
    df2 = makeDate(df2)
    df3 = makeDate(df3)
    df4_a = makeDate(regional_consumed)
    df4_b = makeDate(categorical_consumed)
    
    df = df4_a.merge(df4_b, on = '날짜', how = 'left').merge(df3, on = '날짜', how = 'left').merge(df2, on = '날짜', how = 'left').merge(df1, on = '날짜', how = 'left')
    
    # 경제지수는 한달 전 수치를 사용하기 위해서
    from dateutil.relativedelta import relativedelta
    df['날짜'] = df['날짜'].apply(lambda x: x + relativedelta(months = 1))
    df['연도'] = df['날짜'].dt.year
    df['월'] = df['날짜'].dt.month
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 5)
    X = pca.fit_transform(df.iloc[:,1:60])
    df = pd.concat([df.iloc[:, -2:], df.iloc[:, 60:-2], pd.DataFrame(X, columns = ['pca_1',  'pca_2', 'pca_3', 'pca_4', 'pca_5'])], axis = 1)
    df = df.astype(float)
    return df





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    