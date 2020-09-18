# -*- coding: utf-8 -*-

import os
import sys
import gc
import joblib
import FE_innData as FEin
import FE_extData as FEex
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--embedding', required = True)
parser.add_argument('--dataset', required = True)
args = parser.parse_args()

print('========== Start Feature Engineering ==========')
# 본 데이터 불러오기
print(f'{args.dataset} Data Load.....')

if args.dataset == 'train':
    sale = pd.read_excel(os.path.join('..', '..', '0.Data', '01_제공데이터', '2020 빅콘테스트 데이터분석분야-챔피언리그_2019년 실적데이터_v1_200818.xlsx'), skiprows = 1)
elif args.dataset == 'test':
    sale = pd.read_excel(os.path.join('..', '..', '0.Data', '02_평가데이터', '2020 빅콘테스트 데이터분석분야-챔피언리그_2020년 6월 판매실적예측데이터(평가데이터).xlsx'), skiprows = 1)
else:
    print('dataset error.....')

############## 내부 데이터 FE ##############

# 기본 데이터 FE
print('Inner Feature enginnering.....')
sale = FEin.engineering_data(sale)
sale = FEin.engineering_TimeDiff(sale)
sale = FEin.engineering_DatePrice(sale, args.dataset)
sale = FEin.engineering_order(sale)

# 시계열 데이터 FE
print('Time Feature enginnering.....')
sale = FEin.engineering_timeSeries(sale)

# 임베딩 데이터 FE
print('emb Feature enginnering.....')
emb = pd.read_excel(os.path.join('..', '..', '0.Data', '04_임베딩데이터', f'{args.embedding}.xlsx'), index_col = 0)
sale = sale.merge(emb.drop_duplicates(), on = 'NEW상품명', how = 'left')

del emb
gc.collect()

# print('Complete emb Feature enginnering!')

############## 외부 데이터 FE ##############
print('external Feature enginnering.....')
print('external Data Load.....')

# 날씨 데이터 FE
w_19 = pd.read_csv(os.path.join('..', '..', '0.Data', '03_외부데이터', '2019_weather.csv'), encoding = 'cp949', dtype='unicode')
w_20 = pd.read_csv(os.path.join('..', '..', '0.Data', '03_외부데이터', '2020_weather.csv'), encoding = 'cp949', dtype='unicode')
df_wth = pd.concat([w_19, w_20], axis = 0)
df_wth = FEex.preprocessing_weather(df_wth)

# 미세먼지 데이터 FE
dust_2019 = pd.read_csv(os.path.join('..', '..', '0.Data', '03_외부데이터', '2019_dust.csv'), encoding = 'cp949')
dust_2020 = pd.read_csv(os.path.join('..', '..', '0.Data', '03_외부데이터', '2020_dust.csv'), encoding = 'cp949')
df_dust = pd.concat([dust_2019, dust_2020], axis = 0)
df_dust = FEex.preprocessing_dust(df_dust)

# 경제 데이터 FE
df_eco = FEex.preprocessing_economy()

# 외부 데이터 merge
data = sale.merge(df_eco, left_on = ['방송년도', '방송월'], right_on = ['연도', '월'], how = 'left').drop(['연도', '월'], axis = 1)

del sale
gc.collect()

data = data.merge(df_wth, left_on = ['방송년도', '방송월', '방송일', '방송시간(시간)'], right_on = ['연도', '월', '일' ,'시간'], how = 'left').drop(['연도', '월', '일', '시간'], axis = 1)

data = data.merge(df_dust, left_on = ['방송년도', '방송월', '방송일', '방송시간(시간)'], right_on = ['연도', '월', '일', '시간'], how ='left').drop(['연도', '월', '일', '시간'], axis = 1)
print('Complete external Feature enginnering!')

############## 데이터 Preprocessing ##############
print('========== Start Data preprocessing ==========')
categorys = ['결제방법', '상품군_가격대', '전체_가격대', '상품군', '방송시간(시간)', '방송시간(분)', '성별']
drop_columns = ['방송일시', # 월, 일, 시간, 분으로 표현
                '마더코드', # 
                '상품코드', #
                '상품명', # 임베딩
                'NEW상품코드', #
                'NEW상품명', # 임베딩
                '단위', # 임베딩
                '브랜드', # 임베딩
                '취급액', # target
                '모델명',
                '상품명다시']

data[categorys] = data[categorys].astype(str)
    
# 예측 상품 중 판매가 0인 프로그램 실적은 예측에서 제외함 -> 무형 제외
data = data.loc[data['상품군'] != '무형']
# 주문이 0인, 취급액이 0인 데이터 제외함
data = data.loc[data['취급액'].notnull()]

y = data['취급액']
drop_data = data[drop_columns]
data = data.drop(drop_columns, axis = 1)

today = datetime.today().strftime('%Y%m%d%H%M')

joblib.dump({
    'X' : data,
    'y' : y
}, os.path.join('..', '..', '0.Data', '05_분석데이터', '6th_FE_{}_before.pkl').format(today))
print('before Data saved!')

X = pd.get_dummies(data)
print('Complete Data preprocessing!')
print('complete Data saving.....')

joblib.dump({
    'X' : X,
    'y' : y
}, os.path.join('..', '..', '0.Data', '05_분석데이터', '6th_FE_{}.pkl').format(today))
print('Data saved!')