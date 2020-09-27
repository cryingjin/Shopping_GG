# -*- coding: utf-8 -*-

import os
import sys
import gc
import joblib
import FE_innData as FEin
import FE_extData as FEex
import FE_make_corpus as MC 
import FE_NLP_ours as FE_NLP
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from datetime import datetime

parser = argparse.ArgumentParser()
# parser.add_argument('--embedding', required = True)
parser.add_argument('--dataset', required = True)
args = parser.parse_args()

print('========== Start Feature Engineering ==========')
# 본 데이터 불러오기
print(f'{args.dataset} Data Load.....')

if args.dataset == 'train':
    sale = pd.read_excel(os.path.join('..', '..', '0.Data', '01_제공데이터', '2020 빅콘테스트 데이터분석분야-챔피언리그_2019년 실적데이터_v1_200818.xlsx'), skiprows = 1)
elif args.dataset == 'test':
    sale = pd.read_excel(os.path.join('..', '..', '0.Data', '02_평가데이터', '2020 빅콘테스트 데이터분석분야-챔피언리그_2020년 6월 판매실적예측데이터(평가데이터).xlsx'), skiprows = 1)
    test_index = sale.index
else:
    print('dataset error.....')

############## 내부 데이터 FE ##############

# 기본 데이터 FE
print('Inner Feature enginnering.....')
sale = FEin.engineering_data(sale, args.dataset)

sale = FEin.engineering_TimeDiff(sale)

sale = FEin.engineering_DatePrice(sale, args.dataset)

sale = FEin.engineering_order(sale, args.dataset)


# 시계열 데이터 FE
print('Time Feature enginnering.....')
sale = FEin.engineering_timeSeries(sale, args.dataset)

# 임베딩 데이터 FE
print('emb Feature enginnering.....')

# if args.dataset == 'train':
#     meta = pd.read_excel(os.path.join('..', '..', '0.Data', '01_제공데이터', 'train수작업_meta.xlsx'))
# elif args.dataset == 'test':
meta_train = pd.read_excel(os.path.join('..', '..', '0.Data', '01_제공데이터', 'train수작업_meta.xlsx'))
meta_test = pd.read_excel(os.path.join('..', '..', '0.Data', '02_평가데이터', 'test수작업_meta.xlsx'))
meta = pd.concat([meta_train, meta_test], axis = 0)
meta = meta.drop_duplicates('NEW상품명')[['NEW상품명', '브랜드', '상품명다시', '단위']]

corpus_our = MC.make_corpus_our(meta)
FE_NLP_our = FE_NLP.FE_W2V(meta, corpus_our)
FE_NLP_our.W2V()
emb = FE_NLP_our.product_name_embedding_ver4()

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
if args.dataset == 'train':
    dust_2019 = pd.read_csv(os.path.join('..', '..', '0.Data', '03_외부데이터', '2019_dust.csv'), encoding = 'cp949')
    dust_2020 = pd.read_csv(os.path.join('..', '..', '0.Data', '03_외부데이터', '2020_dust.csv'), encoding = 'cp949')
    df_dust = pd.concat([dust_2019, dust_2020], axis = 0)
    df_dust = FEex.preprocessing_dust(df_dust, args.dataset)
elif args.dataset == 'test':
    df_dust = pd.read_excel(os.path.join('..', '..', '0.Data', '03_외부데이터', '2020_dust', '2020년 6월.xlsx'))

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
drop_columns = [
                '방송일시', # 월, 일, 시간, 분으로 표현
                '방송년도',
                '마더코드', # 
                '상품코드', #
                '상품명', # 임베딩
                'NEW상품코드', #
                'NEW상품명', # 임베딩
                '단위', # 임베딩
                '브랜드', # 임베딩
                '취급액', # target
                '모델명', # 
                '상품명다시', # 임베딩
                '옵션', # 옵션여부,
                '판매량']

data[categorys] = data[categorys].astype(str)
# 예측 상품 중 판매가 0인 프로그램 실적은 예측에서 제외함 -> 무형 제외
data = data.loc[data['상품군'] != '무형']

today = datetime.today().strftime('%Y%m%d%H%M')

if args.dataset == 'train':
    # 주문이 0인, 취급액이 0인 데이터 제외함
    data = data.loc[data['취급액'].notnull()]
    y = data['취급액']
    
    label4WnD = data['상품군']
    drop_data = data[drop_columns]
    data = data.drop(drop_columns, axis = 1)
    
    joblib.dump({
        'X' : data,
        'label' : label4WnD
    },
        os.path.join('..', '..', '0.Data', '05_분석데이터', 'train_data4WnD.pkl'))
    
    X = pd.get_dummies(data)
    print('Complete Data preprocessing!')
    print('Data saving.....')
    joblib.dump({
        'X' : X,
        'y' : y
    }, os.path.join('..', '..', '0.Data', '05_분석데이터', 'train_FE.pkl'))
    print('Train Data saved!')
    
elif args.dataset == 'test':
    drop_columns.remove('판매량')
    drop_data = data[drop_columns]
    label4WnD = data['상품군']
    data = data.drop(drop_columns, axis = 1)
    test_index = data.index
    
    joblib.dump({
        'X' : data,
        'label' : label4WnD
    },
        os.path.join('..', '..', '0.Data', '05_분석데이터', 'test_data4WnD.pkl'))
    
    X = pd.get_dummies(data)
    
    train = joblib.load(os.path.join('..', '..', '0.Data', '05_분석데이터', 'train_FE.pkl'))
    train_columns = train['X'].columns
    test_columns = X.columns
    temp = pd.DataFrame(columns = list(set(train_columns) - set(test_columns)))
    X = pd.concat([X, temp], axis = 1)
    X[temp.columns] = X[temp.columns].fillna(0)
    X = X[train_columns]
    
    print('Complete Data preprocessing!')
    print('Data saving.....')
    joblib.dump({
        'X' : X,
        'idx' : test_index
    }, os.path.join('..', '..', '0.Data', '05_분석데이터', 'test_FE.pkl'))
    print('Test Data saved!')
    
else:
    print('dataset error.....')

        


    




