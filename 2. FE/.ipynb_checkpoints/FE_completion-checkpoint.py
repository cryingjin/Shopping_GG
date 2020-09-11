# -*- coding: utf-8 -*-

import os
import sys
import joblib
import FE_0823 as FE
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--embedding', required = True)
# parser.add_argument('--end', required = True)
args = parser.parse_args()
print('start Feature Engineering.....')

# 본 데이터 불러오기
print('Data Load.....')
sale = pd.read_excel(os.path.join('..', '..', '0.Data', '01_제공데이터', 'sale_data_v05_0828.xlsx'))
meta = pd.read_excel(os.path.join('..', '..', '0.Data', '01_제공데이터', 'sale_meta_v04_0823.xlsx'))
item = pd.read_excel(os.path.join('..', '..', '0.Data', '01_제공데이터', 'item_meta_v04_0828.xlsx'))


# 내부데이터 FE
print('Feature enginnering.....')
sale = FE.engineering_TimeDiff(sale)
sale = FE.engineering_DatePrice(sale)
sale = FE.engineering_trendnorder(sale)
sale = FE.engineering_zscore(sale)
print('Complete Feature enginnering!')

# 임베딩데이터 FE
print('emb Feature enginnering.....')
emb = pd.read_excel(os.path.join('..', '..', '0.Data', '04_임베딩데이터', f'{args.embedding}.xlsx'), index_col = 0)
sale = sale.merge(emb.drop_duplicates(), on = 'NEW상품명', how = 'left')
print('Complete emb Feature enginnering!')

# 외부데이터 FE
print('external Feature enginnering.....')
df_eco = pd.read_excel(os.path.join('..', '..', '0.Data', '03_외부데이터', '전처리', 'prep_2차외부데이터_0908.xlsx'))
df_wth = pd.read_csv(os.path.join('..', '..', '0.Data', '03_외부데이터', '전처리', 'prep_2019_pb_weather.csv'), encoding = 'cp949')
df_dust = pd.read_csv(os.path.join('..', '..', '0.Data', '03_외부데이터', '전처리', 'prep_2019_pb_dust.csv'), encoding = 'cp949')


from sklearn.decomposition import PCA
pca = PCA(n_components = 0.9)
X = pca.fit_transform(df_eco.iloc[:,35:])
df_eco = pd.concat([df_eco.iloc[:,:35], pd.DataFrame(X, columns = ['pca_1',  'pca_2'])], axis = 1)
data = sale.merge(df_eco[df_eco['년도'] == 2019], left_on = '방송월', right_on = '월', how = 'left').drop(['날짜', '월'], axis = 1)

data = data.merge(df_wth, left_on = ['방송월', '방송일', '방송시간(시간)'], right_on = ['월', '일', '시간'], how ='left').drop(['날짜', '월', '일', '시간'], axis = 1)
data = data.merge(df_dust, left_on = ['방송월', '방송일', '방송시간(시간)'], right_on = ['월', '일', '시간'], how ='left').drop(['날짜', '월', '일', '시간'], axis = 1)
print('Complete edternal Feature enginnering!')

print('Data preprocessing.....')
categorys = ['결제방법', '상품군_가격대', '전체_가격대', '상품군', '방송시간(시간)', '방송시간(분)', '성별']
drop_columns = ['방송일시','마더코드', '상품코드', '상품명', 'NEW상품코드', 'NEW상품명', '단위', '브랜드', '취급액', '상품코드', '옵션', '종류', '년도', '상품명다시', '방송날짜']
data[categorys] = data[categorys].astype(str)
    
# 예측 상품 중 판매가 0인 프로그램 실적은 예측에서 제외함 -> 무형 제외
data = data.loc[data['상품군'] != '무형']
# 주문이 0인, 취급액이 0인 데이터 제외함
data = data.loc[data['취급액'].notnull()]

y = data['취급액']
drop_data = data[drop_columns]
data = data.drop(drop_columns, axis = 1)

today = datetime.today().strftime('%Y%m%d%H%M')

# joblib.dump({
#     'X' : data,
#     'y' : y
# }, os.path.join('..', '..', '0.Data', '05_분석데이터', 'Thd_FE_{}_before.pkl').format(today))
# print('Data saved!')

X = pd.get_dummies(data)
print('Complete Data preprocessing!')
print('Data saving.....')

joblib.dump({
    'X' : X,
    'y' : y
}, os.path.join('..', '..', '0.Data', '05_분석데이터', '4th_FE2_{}.pkl').format(today))
print('Data saved!')