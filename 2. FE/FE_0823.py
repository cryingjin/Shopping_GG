import os
import sys
import math
import joblib
import numpy as np
import pandas as pd
import json
import requests
from datetime import datetime, timedelta

# 동일 상품 / 브랜드 총 방송횟수 - 먼저 돌리기 !!
## 판매횟수만 따지기 위해서는 원래 데이터를 봐야 하고,
## 방송횟수를 따지기 위해서는 원래 데이터에서 '노출(분)' NaN값을 drop한 dataframe을 봐야 함 !

## -- JB -- 
def engineering_TimeDiff(df) :

    # [동일상품 방송 시간차] 동일 상품 별 시간 간격
    df['방송일'] = df['방송일시'].dt.date
    t = df.groupby(['NEW상품명','방송일'])['취급액'].sum().reset_index()[['NEW상품명','방송일']]
    total = []
    for item in t['NEW상품명'].unique():
        timediff = t.loc[t['NEW상품명'] == item, '방송일']
        a = pd.DataFrame(list(zip(timediff, timediff.diff())))
        a['NEW상품명'] = item
        total.extend(a.values)
    timediff = pd.DataFrame(total, columns = ['방송일', '방송시간차', 'NEW상품명'])
    timediff.loc[timediff['방송시간차'].isnull(), '방송시간차'] = 0
    timediff['방송시간차'] = timediff['방송시간차'].apply(lambda x : x.days if x != 0 else x)
    df = df.merge(timediff, on = ['방송일', 'NEW상품명'], how = 'left')
    
    # [상품 및 브랜드 총 판매 횟수 ] 동일 상품 / 브랜드 총 방송횟수
    df = df.merge(df.groupby('NEW상품명')['방송일'].nunique().reset_index().rename(columns = {'방송일' : '상품방송횟수'}), on = 'NEW상품명', how = 'left')
    
    df = df.merge(df.groupby('브랜드')['방송일'].nunique().reset_index().rename(columns = {'방송일' : '브랜드방송횟수'}), on = '브랜드', how = 'left')
    
    df = df.drop('방송일', axis = 1)
    
    # [조기매진] (20분 이하 혹은 20분과 30분 사이에 조기 종료된 프로그램 선별)
    df['조기매진'] = df['노출(분)'].map(lambda x: 1 if ((x < 20) | (x > 20) & (x < 30)) else 0)     # 20분 이하, 20-30 분 사이
    # 방송일시의 nan 값 채워주는 부분
    df["노출(분)"] = df["노출(분)"].fillna(method='ffill')
    return df 

## -- JS --

def engineering_DatePrice(df):
    # 데이터 로드
    item = pd.read_excel(os.path.join('..', '..','0.Data', '01_제공데이터', 'item_meta_v03_0823.xlsx'))
    data = joblib.load(os.path.join('..', '..','0.Data', '01_제공데이터', '0823_prep4data.pkl'))
    itemcategory = data['itemcategory']
    mothercode = data['mothercode']
    brand = data['brand']
    
    # 공휴일여부
    key = '8wBiTSHPiK2z%2By8ETu%2FpYv%2FMAAdZoR8rZg3PIvSNCcD%2F26BiBPaosFs2dzrVJ%2BHUeaQGWb9c3T4vvNgMpI7fdw%3D%3D'
    def getHoliday(year):
        url = f'http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo?solYear={year}&ServiceKey={key}&_type=json&numOfRows=20'
        response = requests.get(url)
        holidays = response.json()['response']['body']['items']['item']
        holidays = pd.DataFrame(holidays)
        holidays['locdate'] = holidays['locdate'].astype(str).apply(lambda x : '-'.join([x[:4], x[4:6], x[6:]]))
        return holidays
    
    # 무형 상품군 NEW상품명 채워주기(무형 상품군은 어짜피 나중에 버릴것)
    df.loc[df['NEW상품명'].isnull(), 'NEW상품명'] = df.loc[df['NEW상품명'].isnull(), '상품명']
    
    ## [공휴일여부]
    if df.shape[0] > 30000:
        year = 2019
    else:
        year = 2020
    holidays = getHoliday(year)
    df = df.merge(holidays[['locdate', 'isHoliday']], left_on = df['방송일시'].dt.date.astype(str), right_on = 'locdate', how = 'left').drop('locdate', axis = 1)
    df['isHoliday'] = df['isHoliday'].apply(lambda x : 1 if x == 'Y' else 0)
    ## [날짜 및 시간대]
    df['방송년도'] = df['방송일시'].dt.year
    df['방송월'] = df['방송일시'].dt.month
    df['방송일'] = df['방송일시'].dt.day
    df['방송시간(시간)'] = df['방송일시'].dt.hour.apply(lambda x : 24 if x == 0 else x)
    df['방송시간(분)'] = df['방송일시'].dt.minute
    df.loc[df['방송년도'] == 2020, '방송월'] = 12 # 2019년 12월31일의 연장방송이므로 
    df.loc[df['방송년도'] == 2020, '방송일'] = 31
    df = df.drop('방송년도', axis = 1)
    
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
    
    ## [성별] (공용 : 0, 남성 : 1, 여성 : 2)
    df['성별'] = df['NEW상품명'].apply(lambda x : 1 if '남성' in x else (2 if '여성' in x else 0))
    
    ## [상품군]별 가격 summary
    df = df.merge(itemcategory, on = '상품군', how = 'left')
    
    ## [마더코드]별 가격 summary
    df = df.merge(mothercode, on = '마더코드', how = 'left')
    
    ## [브랜드]별 가격 summary
    df = df.merge(brand, on = ['브랜드', '상품군'], how = 'left')
    
    ## [NEW아이템]별 가격 summary + [상품군] 별 가격대, 전체 가격대
    df = df.merge(item, on = ['NEW상품코드', 'NEW상품명', '상품군'], how = 'left')
    
    ## [상품군] 평균 판매단가 - 해당 상품 판매단가
    df['상품군평균판매단가차이'] = df['상품군_평균판매단가'] - df['판매단가']
    
    # [결합상품여부]
    df['결합상품'] = df['NEW상품명'].apply(lambda x : 1 if '+' in x else 0)
    
    return df


def engineering_trendnorder(df):
    # 방송날짜별 상품군별 취급액 계산
    sale = pd.read_excel(os.path.join('..', '..', '0.Data', '01_제공데이터', 'sale_data_v05_0828.xlsx'))
    sale['방송날짜'] = sale['방송일시'].dt.date
    temp = sale.groupby(['상품군', '방송날짜'])['취급액'].sum().reset_index()
    temp['방송날짜'] = pd.to_datetime(temp['방송날짜'])
    
    df['log최근3개월상품군추세'] = None
    
    # 방송날짜 90일 전부터 방송날짜 전까지의 취급액 log 추세 구하기
    df['방송날짜'] = df['방송일시'].dt.date
    df['방송날짜'] = pd.to_datetime(df['방송날짜'])
    for cate, date in df[['상품군', '방송날짜']].drop_duplicates().values:
        log_y = np.log(list(temp.loc[(temp['상품군'] == cate) & (temp['방송날짜'] < date) & (date - timedelta(days = 90) < temp['방송날짜']), '취급액'] + 1))
        log_x = np.arange(len(log_y))
        try:
            log_z = np.polyfit(log_x, log_y, 1)[0]
        except:
            log_z = 0

        df.loc[(df['상품군'] == cate) & (df['방송날짜'] == date), 'log최근3개월상품군추세'] = log_z
    df['log최근3개월상품군추세'] = df['log최근3개월상품군추세'].astype(float)
    
    sale['방송월'] = sale['방송일시'].dt.month
    sale['방송시간(시간)'] = sale['방송일시'].dt.hour
    sale['방송시간(분)'] = sale['방송일시'].dt.minute
    sale['판매량'] = sale['취급액'] / sale['판매단가']
    sale['판매량'] = sale['판매량'].fillna(0).apply(lambda x : math.ceil(x))
    
#     # 월별 상품군 판매량
#     temp = pd.pivot_table(sale, index = '상품군', columns = '방송월', values = '판매량', aggfunc = np.sum).T.reset_index()
#     temp.columns = [temp.columns[0]] + list(map(lambda x : '월별판매랑_' + x, temp.columns[1:]))
#     df = df.merge(temp, on = '방송월', how = 'left')
    
#     # 시간대별 상품군 판매량
#     temp = pd.pivot_table(sale, index = '상품군', columns = '방송시간(시간)', values = '판매량', aggfunc = np.sum).T.reset_index()
#     temp.columns = [temp.columns[0]] + list(map(lambda x : '시간대별판매랑_' + x, temp.columns[1:]))
#     df = df.merge(temp, on = '방송시간(시간)', how = 'left')
    
#     # 시간별 상품군 판매량
#     temp = pd.pivot_table(sale, index = '상품군', columns = '방송시간(분)', values = '판매량', aggfunc = np.mean).T.reset_index()
#     temp.columns = [temp.columns[0]] + list(map(lambda x : '시간별판매랑_' + x, temp.columns[1:]))
#     df = df.merge(temp, on = '방송시간(분)', how = 'left')
    
    # 월별 상품군 판매량
    temp = pd.pivot_table(sale, index = '상품군', columns = '방송월', values = '판매량', aggfunc = np.mean).T.reset_index()
    temp.columns = [temp.columns[0]] + list(map(lambda x :  x, temp.columns[1:]))
    df['상품군별월별평균판매량'] = None
    for i in range(1, 13):
        for cate in df['상품군'].unique():
            df.loc[(df['방송월'] == i) & (df['상품군'] == cate), '상품군별월별평균판매량'] = temp[cate].loc[temp['방송월'] == i].values[0]

    # 시간대별 상품군 판매량
    temp = pd.pivot_table(sale, index = '상품군', columns = '방송시간(시간)', values = '판매량', aggfunc = np.mean).T.reset_index()
    temp.columns = [temp.columns[0]] + list(map(lambda x : x, temp.columns[1:]))
    df['상품군별시간대별평균판매량'] = None
    for i in range(24):
        for cate in df['상품군'].unique():
            try:
                df.loc[(df['방송시간(시간)'] == i) & (df['상품군'] == cate), '상품군별시간대별평균판매량'] = temp[cate].loc[temp['방송시간(시간)'] == i].values[0]
            except:
                continue
    
    # 시간별 상품군 판매량
    temp = pd.pivot_table(sale, index = '상품군', columns = '방송시간(분)', values = '판매량', aggfunc = np.mean).T.reset_index()
    temp.columns = [temp.columns[0]] + list(map(lambda x : x, temp.columns[1:]))
    sale['상품군별시간분별평균판매량'] = None
    for i in df['방송시간(분)'].unique():
        for cate in df['상품군'].unique():
            try:
                df.loc[(df['방송시간(분)'] == i) & (df['상품군'] == cate), '상품군별시간분별평균판매량'] = temp[cate].loc[temp['방송시간(분)'] == i].values[0]
            except:
                continue
    df[['상품군별월별평균판매량', '상품군별시간대별평균판매량', '상품군별시간분별평균판매량']] = df[['상품군별월별평균판매량', '상품군별시간대별평균판매량', '상품군별시간분별평균판매량']].astype(float)
    # 할인율
    rt = pd.read_csv(os.path.join('..', '..', '0.Data', '01_제공데이터', 'prep_discountRt.csv'), encoding = 'cp949')
    df['할인율'] = rt.values
    
    return df
    

def engineering_zscore(df):

    def zscore(price, mean, std):
        if std == 0:
            return 0
        else:
            return (price - mean) / std

    df["상품군_zscore"] = df.apply(lambda x: zscore(x["판매단가"], x["상품군_평균판매단가"], x["상품군_표준편차"]), axis=1)
    df["상품군&브랜드_zscore"] = df.apply(lambda x: zscore(x["판매단가"], x["상품군&브랜드_평균판매단가"], x["상품군&브랜드_표준편차"]), axis=1)
    df["마더코드_zscore"] = df.apply(lambda x: zscore(x["판매단가"], x["마더코드_평균판매단가"], x["마더코드_표준편차"]), axis=1)
    df["NEW_zscore"] = df.apply(lambda x: zscore(x["판매단가"], x["NEW_평균판매단가"], x["NEW_표준편차"]), axis=1)

    # z-score 만들어 줬으니 평균, 분산, 표준편차 제거!
    stat_mean = df.columns[df.columns.str.contains("_평균판매단가")]
    stat_var = df.columns[df.columns.str.contains("분산")]
    stat_std = df.columns[df.columns.str.contains("표준편차")]

    stats = [stat_mean, stat_var, stat_std]
    for stat in stats:
        df.drop(stat, axis=1, inplace=True)

    return df