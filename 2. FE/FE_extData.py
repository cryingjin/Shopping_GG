import os
import random
import datetime
import numpy as np
import pandas as pd


def preprocessing_ext_weather(df):
    df = df.loc[df['지점명'].str.contains('인천|울산|대구|대전|수원|부산|광주|서울')]
    df = df.rename(columns = {'수원' : '경기'})
    
    # 변수 정리
    df = df[list(df.columns[(df.isnull().sum() / df.shape[0]) < 0.5]) + ['강수량(mm)']]
    df = df.drop(['지점', '풍향(16방위)', '증기압(hPa)', '이슬점온도(°C)', '현지기압(hPa)', '해면기압(hPa)',
                  '일조(hr)', '일조 QC플래그', '전운량(10분위)', '중하층운량(10분위)', '최저운고(100m )', '운형(운형약어)'], axis = 1)
    
    
    # 체감온도 : 13.12 + 0.6215T - 11.37V^0.16 + 0.3965V^0.16T (T : 기온(°C), V : 풍속(km/h))
    df['체감온도'] = df.apply(lambda x: 13.12 + 0.6215 * x['기온(°C)'] - 11.37 * (x['풍속(m/s)'] * 3.6)**0.16 + 0.3965 * (x['풍속(m/s)'] * 3.6)**0.16 * x['기온(°C)'], axis = 1)
    
    # merge를 위한 날짜 생성 
    df['일시'] = pd.to_datetime(df['일시'])
    df['연도'] = df['일시'].dt.year
    df['월'] = df['일시'].dt.month
    df['일'] = df['일시'].dt.day
    df['시간'] = df['일시'].dt.hour
    
    return df


def preprocessing_ext_dust(df):
    df = df.loc[df['지역'].str.contains('서울|경기|인천|부산|울산|대구|대전|광주')]
    df['지역'] = df['지역'].apply(lambda x : x[:2])
    
    prep_df = df.groupby(['지역', '측정일시']).agg({
        'PM10' : [('최고PM10', np.max), ('최저PM10', np.min), ('평균PM10', np.mean)],
        'PM25' : [('최고PM25', np.max), ('최저PM25', np.min), ('평균PM25', np.mean)]
    }).reset_index()
    
    prep_df.columns = prep_df.columns = ['지역', '측정일시', '최고PM10', '최저PM10', '평균PM10', '최고PM25', '최저PM25', '평균PM25']
    
    prep_df['측정일시'] = prep_df['측정일시'].astype(str).apply(lambda x : x[:8] + '00' if x[8:] == '24' else x)
    prep_df['측정일시'] = pd.to_datetime(prep_df['측정일시'].astype(str).apply(lambda x : '-'.join((x[:4], x[4:6], x[6:8])) + ' '+ x[8:] + ':00:00'))
    prep_df.loc[prep_df['측정일시'].dt.hour == 0, '측정일시'] = prep_df.loc[prep_df['측정일시'].dt.hour == 0, '측정일시'] + datetime.timedelta(days = 1)
    
    # merge를 위한 날짜 생성 
    df['일시'] = pd.to_datetime(df['일시'])
    df['연도'] = df['일시'].dt.year
    df['월'] = df['일시'].dt.month
    df['일'] = df['일시'].dt.day
    df['시간'] = df['일시'].dt.hour
    
    return df






    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    