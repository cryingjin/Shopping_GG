import os
import sys
import joblib
import numpy as np
import pandas as pd
import json
import requests

def preprocessing(df):
    # 데이터 로드
    item = pd.read_excel(os.path.join('..', '0.Data', '01_제공데이터', 'item_meta_v03_0823.xlsx'))
    data = joblib.load(os.path.join('..', '0.Data', '01_제공데이터', '0823_prep4data.pkl'))
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
    
    df.loc[df['NEW상품명'].isnull(), 'NEW상품명'] = df.loc[df['NEW상품명'].isnull(), '상품명']
    
    ## [공휴일여부]
    print('연도입력')
    year = input()
    holidays = getHoliday(year)
    df = df.merge(holidays[['locdate', 'isHoliday']], left_on = df['방송일시'].dt.date.astype(str), right_on = 'locdate', how = 'left')
    
    ## 날짜 및 시간대
    df['방송월'] = df['방송일시'].dt.month
    df['방송시간(시간)'] = df['방송일시'].dt.hour.apply(lambda x : 24 if x == 0 else x)
    df['방송시간(분)'] = df['방송일시'].dt.minute
    
    
    ## [평일여부] (평일 : 0, 주말 : 1)
    df['평일여부'] = df['방송일시'].dt.weekday.apply(lambda x : 0 if x < 5 else 1)
    
    ## [방송시간대] (아침/오전/오후/저녁/밤 (아침(6:00~9:00)/오전(9:00~12:00)/오후(12:00~18:00)/저녁(18:00~22:00)/밤(22:00~2:00))
    df['방송시간대'] = df['방송일시'].dt.hour.apply(lambda x : '아침' if 5 < x <= 9 else 
                                          ('오전' if 10 <= x <= 12 else
                                           ( '오후' if 13 <= x <= 18 else
                                            ('저녁' if 19 <= x <= 22 else
                                             ('밤' if 23 <= x < 3 else x)
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
    
    return df