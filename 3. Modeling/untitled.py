# -*- coding: utf-8 -*-

import os
import sys
import gc
import joblib
sys.path.append(os.path.join('..', '2. FE'))
import FE_innData as FEin
import FE_extData as FEex

import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required = True)
args = parser.parse_args()

print('========== Start Feature Engineering ==========')
# 본 데이터 불러오기
print(f'{args.dataset} Data Load.....')


if args.dataset == 'recommend':
    rec = pd.read_excel(os.path.join('..', '..', '0.Data', '01_제공데이터', '2020 빅콘테스트 데이터분석분야-챔피언리그_방송편성표추천데이터'))
else:
    print('dataset error.....')
    
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
    
    
    # 내부데이터
    prep = pd.read_excel('data/prep4wnd.xlsx')