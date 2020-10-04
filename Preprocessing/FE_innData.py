# -*- coding: utf-8 -*-

import os
import sys
import math
import joblib
import numpy as np
import pandas as pd
import json
import requests
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


############## 제공데이터 전처리 ##############
def engineering_data(df, dataset):
    
    if dataset == 'train':
        # 수작업 데이터 불러오기 (2019 제공데이터를 가지고 수작업 한 파일)
        meta = pd.read_excel(os.path.join('..','data', '01_제공데이터', 'train수작업_meta.xlsx'))
        df = df.merge(meta[['상품코드', 'NEW상품명', '브랜드', '결제방법', '상품명다시', '단위', '모델명', '성별', 'NS카테고리', '옵션']], on = '상품코드', how = 'left')
        item = meta[['NEW상품명', '상품군']].drop_duplicates().reset_index(drop = True).reset_index().rename(columns = {'index' : 'NEW상품코드'})
    elif dataset == 'test' or dataset == 'recommend':
        # 수작업 데이터 불러오기 (2019 제공데이터를 가지고 수작업 한 파일 + 2020 평가데이터를 가지고 수작업 한 파일)
        meta_train = pd.read_excel(os.path.join('..','data', '01_제공데이터', 'train수작업_meta.xlsx'))
        meta_test = pd.read_excel(os.path.join('..','data', '02_평가데이터', 'test수작업_meta.xlsx'))
        meta = pd.concat([meta_train, meta_test], axis = 0)
        meta = meta.loc[meta['상품군'] != '무형'].drop_duplicates('상품코드')
        
        if dataset == 'test':
            df = df.merge(meta[['상품코드', 'NEW상품명', '브랜드', '결제방법', '상품명다시', '단위', '모델명', '성별', 'NS카테고리', '옵션']], on = '상품코드', how = 'left')
        item = meta[['NEW상품명', '상품군']].drop_duplicates().reset_index(drop = True).reset_index().rename(columns = {'index' : 'NEW상품코드'})
    else:
        print('dataset error.....')
    
    # 집계 데이터프레임 칼럼 정리
    def prepColumn(df):
        columns = []
        for i, c in enumerate(df.columns):
            if c[1] == '':
                columns.append(c[0])
            else:
                columns.append(c[1])
        df.columns = columns
        return df
    
    # NEW상품명 기준 집계 데이터프레임 생성
    temp = meta.groupby('NEW상품명').agg({
        '판매단가' : [('NEW_최고판매단가', np.max),
                 ('NEW_최저판매단가', np.min),
                  ('NEW_평균판매단가', np.mean),
                  ('NEW_중간판매단가', np.median),
                 ('NEW_최고-최저', lambda x : np.max(x) - np.min(x)),
                  ('NEW_분산', np.var),
                  ('NEW_표준편차', np.std)
                 ]
    }).reset_index()
    temp = prepColumn(temp).fillna(0)
    
    item = item.merge(temp, on = 'NEW상품명', how = 'left')
    
    # 가격대 [전체_가격대]
    item['전체_가격대'] = item['NEW_최고판매단가'].apply(lambda x : '저가' if x <= 59000 else ('중저가' if 59000 < x <= 109900 else ('고가' if 109900 < x < 509000 else '초고가')))
    
    # 상품군 내 가격대 [상품군_가격대]
#     item.to_excel('./item.xlsx', index = False)
    for c in item['상품군'].unique():
        if c == '무형':
            continue
        item.loc[item['상품군'] == c, '상품군_가격대'] = pd.qcut(item.loc[item['상품군'] == c, 'NEW_최고판매단가'], q = 3, labels = False)
    
    # 마더코드 기준 집계 데이터프레임 생성
    mothercode = meta.groupby('마더코드').agg({
        '판매단가' : [('마더코드_최고판매단가', np.max),
                 ('마더코드_최저판매단가', np.min),
                  ('마더코드_평균판매단가', np.mean),
                  ('마더코드_최고-최저', lambda x : np.max(x) - np.min(x)),
                  ('마더코드_분산', np.var),
                  ('마더코드_표준편차', np.std)
                 ]
    }).reset_index()
    mothercode = prepColumn(mothercode).fillna(0)

    # 상품군 기준 집계 데이터프레임 생성
    itemcategory = df.groupby('상품군').agg({
        '판매단가' : [('상품군_최고판매단가', np.max),
                 ('상품군_최저판매단가', np.min),
                  ('상품군_평균판매단가', np.mean),
                  ('상품군_중간판매단가', np.median),
                 ('상품군_최고-최저', lambda x : np.max(x) - np.min(x)),
                 ('상품군_표준편차', np.std),
                  ('상품군_분산', np.var)
                 ]
    }).reset_index()
    itemcategory = prepColumn(itemcategory).fillna(0)
    
    # NEW아이템 가격 집계 merge
    df = df.merge(item.drop('상품군', axis = 1), on = 'NEW상품명', how = 'left')
    
    
    # 상푼군 가격 집계 merge
    df = df.merge(itemcategory, on = '상품군', how = 'left')
    
    # 마더코드 가격 집계 merge
    df = df.merge(mothercode, on = '마더코드', how = 'left')
    
    if dataset == 'train':
        item.to_csv(os.path.join('..','data', '01_제공데이터', 'item_meta(train).csv'), index = False, encoding = 'cp949')
    elif dataset == 'test':
        item.to_csv(os.path.join('..','data', '02_평가데이터', 'item_meta(test).csv'), index = False, encoding = 'cp949')
    
    return df

############## 시간차 관련 FE ##############
def engineering_TimeDiff(df) :

    # [동일상품 방송 시간차] 동일 상품 별 시간 간격
    df['방송날'] = df['방송일시'].dt.date
    t = df.groupby(['NEW상품명','방송날'])['취급액'].sum().reset_index()[['NEW상품명','방송날']]
    total = []
    for item in t['NEW상품명'].unique():
        timediff = t.loc[t['NEW상품명'] == item, '방송날']
        a = pd.DataFrame(list(zip(timediff, timediff.diff())))
        a['NEW상품명'] = item
        total.extend(a.values)
    timediff = pd.DataFrame(total, columns = ['방송날', '방송시간차', 'NEW상품명'])
    timediff.loc[timediff['방송시간차'].isnull(), '방송시간차'] = 0
    timediff['방송시간차'] = timediff['방송시간차'].apply(lambda x : x.days if x != 0 else x)
    df = df.merge(timediff, on = ['방송날', 'NEW상품명'], how = 'left')
    
    # [상품 및 브랜드 총 판매 횟수 ] 동일 상품 / 브랜드 총 방송횟수
#     df = df.merge(df.groupby('NEW상품명')['방송날'].nunique().reset_index().rename(columns = {'방송날' : '상품방송횟수'}), on = 'NEW상품명', how = 'left')
#     df = df.merge(df.groupby('브랜드')['방송날'].nunique().reset_index().rename(columns = {'방송날' : '브랜드방송횟수'}), on = '브랜드', how = 'left')
    df = df.drop('방송날', axis = 1)
    
    # [조기매진] (20분 이하 혹은 20분과 30분 사이에 조기 종료된 프로그램 선별)
    df['조기매진'] = df['노출(분)'].map(lambda x: 1 if ((x < 20) | (x > 20) & (x < 30)) else 0)     # 20분 이하, 20-30 분 사이
    
    # 방송일시의 nan 값 채워주는 부분
    df["노출(분)"] = df["노출(분)"].fillna(method='ffill')
    
    # [옵션여부]
    df['옵션'] = df['옵션'].fillna(0)
    df['옵션여부'] = df['옵션'].apply(lambda x : 1 if x != 0 else 0)
    
    # 무형 상품군 NEW상품명 채워주기(무형 상품군은 어짜피 나중에 버릴것)
    df.loc[df['NEW상품명'].isnull(), 'NEW상품명'] = df.loc[df['NEW상품명'].isnull(), '상품명']
    
    return df


############## 날짜&가격 관련 FE ##############
def engineering_DatePrice(df, dataset):
    # 공휴일여부
    key = '8wBiTSHPiK2z%2By8ETu%2FpYv%2FMAAdZoR8rZg3PIvSNCcD%2F26BiBPaosFs2dzrVJ%2BHUeaQGWb9c3T4vvNgMpI7fdw%3D%3D'
    def getHoliday(year):
        url = f'http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo?solYear={year}&ServiceKey={key}&_type=json&numOfRows=20'
        response = requests.get(url)
        holidays = response.json()['response']['body']['items']['item']
        holidays = pd.DataFrame(holidays)
        holidays['locdate'] = holidays['locdate'].astype(str).apply(lambda x : '-'.join([x[:4], x[4:6], x[6:]]))
        return holidays
    
    ## [공휴일여부]
    try:
        if dataset == 'train':
            year = 2019
        elif dataset == 'test':
            year = 2020

        holidays = getHoliday(year)
    except:
        holidays = pd.read_excel(os.path.join('..','data', '03_외부데이터', '특일정보.xlsx'))
    df = df.merge(holidays[['locdate', 'isHoliday']], left_on = df['방송일시'].dt.date.astype(str), right_on = 'locdate', how = 'left').drop('locdate', axis = 1)
    df['isHoliday'] = df['isHoliday'].apply(lambda x : 1 if x == 'Y' else 0)
    
    ## [날짜 및 시간대]
    df['방송년도'] = df['방송일시'].dt.year
    df['방송월'] = df['방송일시'].dt.month
    df['방송일'] = df['방송일시'].dt.day
    df['방송시간(시간)'] = df['방송일시'].dt.hour
    df['방송시간(분)'] = df['방송일시'].dt.minute
    
    if dataset == 'train':
        df.loc[df['방송년도'] == 2020, '방송월'] = 12 # 2019년 12월31일의 연장방송이므로 
        df.loc[df['방송년도'] == 2020, '방송일'] = 31
    
    ## [평일여부] (평일 : 0, 주말 : 1)
    df['평일여부'] = df['방송일시'].dt.weekday.apply(lambda x : 0 if x < 5 else 1)
    
    ## [방송시간대] (아침/오전/오후/저녁/밤 (아침(6:00~9:00)/오전(9:00~12:00)/오후(12:00~18:00)/저녁(18:00~22:00)/밤(22:00~2:00))
    df['방송시간대'] = df['방송일시'].dt.hour.apply(lambda x : '아침' if 5 < x <= 9 else 
                                          ('오전' if 10 <= x <= 12 else
                                           ( '오후' if 13 <= x <= 18 else
                                            ('저녁' if 19 <= x <= 22 else
                                             ('밤' if 23 <= x or x < 3 else x)
                                            ))))
    
    ## [계절] (봄(3~5), 여름6~8), 가을(9~11), 겨울(12~2))
    df['계절'] = df['방송월'].apply(lambda x : '봄' if 3 <= x <= 5 else
                            ('여름' if 6 <= x <= 8 else
                             ('가을' if 9 <= x <= 11 else 
                              ('겨울' if x < 3 or x > 11 else x))))
    
    ## [분기] 
    df['분기'] = df['방송월'].apply(lambda x : '1분기' if 1 <= x <= 3 else
                            ('2분기' if 4 <= x <= 6 else
                             ('3분기' if 7 <= x <= 9 else 
                              ('4분기' if 10 <= x <= 12 else x))))
    
    if dataset == 'recommend':
        return df
    ## [상품군] 평균 판매단가 - 해당 상품 판매단가
    df['상품군평균판매단가차이'] = df['상품군_평균판매단가'] - df['판매단가']
    
    # [결합상품여부]
    df['결합상품'] = df['NEW상품명'].apply(lambda x : 1 if '+' in x else 0)
    
    # [zscore]
    def zscore(price, mean, std):
        if std == 0:
            return 0
        else:
            return (price - mean) / std
    
    df["상품군_zscore"] = df.apply(lambda x: zscore(x["판매단가"], x["상품군_평균판매단가"], x["상품군_표준편차"]), axis=1)
    df["마더코드_zscore"] = df.apply(lambda x: zscore(x["판매단가"], x["마더코드_평균판매단가"], x["마더코드_표준편차"]), axis=1)
    df["NEW_zscore"] = df.apply(lambda x: zscore(x["판매단가"], x["NEW_평균판매단가"], x["NEW_표준편차"]), axis=1)

    # z-score 만들어 줬으니 평균, 분산, 표준편차 제거!
    stat_mean = df.columns[df.columns.str.contains("_평균판매단가")]
    stat_var = df.columns[df.columns.str.contains("분산")]
    stat_std = df.columns[df.columns.str.contains("표준편차")]

    stats = list(stat_mean) + list(stat_var) + list(stat_std)
    df = df.drop(stats, axis = 1)
    
    
    return df

############## 판매량 관련 FE ##############
def engineering_order(df, dataset):
    # 방송날짜별 상품군별 취급액 계산
    df['방송날'] = df['방송일시'].dt.date    
    df['방송년도'] = df['방송일시'].dt.year
#     df['방송월'] = df['방송일시'].dt.month
    df['방송시간(시간)'] = df['방송일시'].dt.hour
    df['방송시간(분)'] = df['방송일시'].dt.minute
    if dataset == 'train':
        df['판매량'] = df['취급액'] / df['판매단가']
        df['판매량'] = df['판매량'].fillna(0).apply(lambda x : math.ceil(x))

        # [월별 상품군 판매량]
        temp_month = pd.pivot_table(df, index = '상품군', columns = '방송월', values = '판매량', aggfunc = np.mean).T.reset_index()
        temp_month.columns = [temp_month.columns[0]] + list(map(lambda x :  x, temp_month.columns[1:]))
        
        # [시간대별(시각) 상품군 판매량]
        temp_hour = pd.pivot_table(df, index = '상품군', columns = '방송시간(시간)', values = '판매량', aggfunc = np.mean).T.reset_index()
        temp_hour.columns = [temp_hour.columns[0]] + list(map(lambda x : x, temp_hour.columns[1:]))

        # [시간별(분) 상품군 판매량]
        temp_minute = pd.pivot_table(df, index = '상품군', columns = '방송시간(분)', values = '판매량', aggfunc = np.mean).T.reset_index()
        temp_minute.columns = [temp_minute.columns[0]] + list(map(lambda x : x, temp_minute.columns[1:]))
        
        # test 데이터 적용을 위한 저장
        joblib.dump({
            'volume4month' : temp_month,
            'volume4hour' : temp_hour,
            'volume4minute' : temp_minute
        },
            os.path.join('..','data', '04_임시데이터', 'data4volume.pkl'))
        
    elif dataset == 'test' or dataset == 'recommend':
        volume = joblib.load(os.path.join('..','data', '04_임시데이터', 'data4volume.pkl'))
        temp_month = volume['volume4month']
        temp_hour = volume['volume4hour']
        temp_minute = volume['volume4minute']
    else:
        print('dataset error.....')
    
    
    # [월별 상품군 판매량]
    df['상품군별월별평균판매량'] = None
    for i in range(1, 13):
        for cate in df['상품군'].unique():
            df.loc[(df['방송월'] == i) & (df['상품군'] == cate), '상품군별월별평균판매량'] = temp_month[cate].loc[temp_month['방송월'] == i].values[0]

    # [시간대별(시각) 상품군 판매량]  
    df['상품군별시간대별평균판매량'] = None
    for i in range(24):
        for cate in df['상품군'].unique():
            try:
                df.loc[(df['방송시간(시간)'] == i) & (df['상품군'] == cate), '상품군별시간대별평균판매량'] = temp_hour[cate].loc[temp_hour['방송시간(시간)'] == i].values[0]
            except:
                continue
                
    # [시간별(분) 상품군 판매량]                
    df['상품군별시간분별평균판매량'] = None
    for i in df['방송시간(분)'].unique():
        for cate in df['상품군'].unique():
            try:
                df.loc[(df['방송시간(분)'] == i) & (df['상품군'] == cate), '상품군별시간분별평균판매량'] = temp_minute[cate].loc[temp_minute['방송시간(분)'] == i].values[0]
            except:
                 continue
        
    df[['상품군별월별평균판매량', '상품군별시간대별평균판매량', '상품군별시간분별평균판매량']] = df[['상품군별월별평균판매량', '상품군별시간대별평균판매량', '상품군별시간분별평균판매량']].astype(float)
    
    # [할인율]
#     rt = pd.read_csv(os.path.join('..','data', '01_제공데이터', 'prep_discountRt.csv'), encoding = 'cp949')
#     df['할인율'] = rt.values
    
    return df

############## 시계열 관련 FE ##############
def engineering_timeSeries(df, dataset):
    
    df['방송날'] = df['방송일시'].dt.date
    df['방송날'] = pd.to_datetime(df['방송날'])
    
    if dataset == 'train':
        piv = pd.pivot_table(df, index = '상품군', columns = '방송날', values = '판매량', aggfunc=np.mean)
        pivT = piv.T

        ema_s = pivT.ewm(span=4).mean()
        ema_m = pivT.ewm(span=12).mean()
        ema_l = pivT.ewm(span=26).mean()
        macd = ema_s - ema_l
        sig = macd.ewm(span=9).mean()

        rol14 = pivT.fillna(0).rolling(14).mean()
        rol30 = pivT.fillna(0).rolling(30).mean()


        for tb, column in zip([ema_s, ema_m, ema_l, macd, sig, rol14, rol30], ['ema_s', 'ema_m', 'ema_l', 'macd', 'sig', 'rol14', 'rol30']):
            new_columns = list(map(lambda x : '_'.join((column, x)), tb.columns))
            tb.columns = new_columns

        timeS = pd.concat([ema_s, ema_m, ema_l, macd, sig, rol14, rol30], axis = 1)
        timeS = timeS.drop(timeS.columns[timeS.columns.str.contains('무형')], axis = 1)
        timeS = timeS.reset_index()
        timeS['방송날'] = pd.to_datetime(timeS['방송날'])
        
        joblib.dump(timeS,
                   os.path.join('..','data', '01_제공데이터', 'data4time.pkl'))
        
    elif dataset == 'test' or dataset == 'recommend':
        
        timeS = joblib.load(os.path.join('..','data', '01_제공데이터', 'data4time.pkl'))
        df['방송날'] = df['방송날'].apply(lambda x : x - relativedelta(years=1))

    timeFE = ['ema_s', 'ema_m', 'ema_l', 'macd', 'sig', 'rol14', 'rol30']
    temp = pd.DataFrame(columns = timeFE)
    df = pd.concat([df, temp], axis = 1)
    
    for dt, cate in df[['방송날', '상품군']].drop_duplicates().values:
        try:
            df.loc[(df['방송날'] == dt) & (df['상품군'] == cate), ['ema_s', 'ema_m', 'ema_l', 'macd', 'sig', 'rol14', 'rol30']] = timeS.loc[timeS['방송날'] == dt, timeS.columns[timeS.columns.str.contains(cate)]].values
        except:
            continue
    df = df.drop('방송날', axis = 1)
    df[timeFE] = df[timeFE].astype(float)
    
    return df