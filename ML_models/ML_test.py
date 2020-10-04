'''
ML_test.py

학습시킨 모델로 예측을 진행하는 def 파일.
새로운 test 데이터가 들어왔을 때 예측값을 추론하는 과정을 거친다.

ML_main.py 로 부터 나온 학습된 모델과 예측값을 통해 stacking 을 진행하여, 최종 예측값을 리턴한다.

'''


import os
import sys
import joblib
import glob
import time
import pickle as pkl
import argparse
import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd
import numpy as np


from IPython.display import display
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
pd.set_option('display.max_info_columns', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# model
# import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from sklearn.linear_model import LinearRegression


# sklearn
from sklearn.model_selection import KFold, train_test_split



# Make DataSet for Stacking
def Stacking_df(X, y, pred_dir, seed=117):
    # --------------------------------------------------------------------------------------------
    # pickle 불러오기
    # --------------------------------------------------------------------------------------------

    # Pred : 경로 설정 주의
    tmp = os.listdir(pred_dir)
    pkls = []
    for t in tmp:
      if str(seed) in t:
        filename, extension = os.path.splitext(t)
        pkls.append(filename)


    final = dict()
    #print(pkls)
    for p in pkls : 
      with open(os.path.join(pred_dir,'{}'.format(p)+".pkl"), 'rb') as f:
        final[p] = pkl.load(f)

    final = dict(sorted(final.items(), key = lambda x : x[0], reverse = False))

    print('='*80)
    print('* Seed : ', seed, '\t * Used Model len : ', len(final))     # len(final) 확인


    # --------------------------------------------------------------------------------------------
    # index 설정
    # --------------------------------------------------------------------------------------------
    # test
    X_test = X.loc[final[f'pred_catBO_{seed}']['test_idx'][0]]
    y_test = y.loc[final[f'pred_catBO_{seed}']['test_idx'][0]]
    # train
    X_train_, X_test_, y_train_, y_test_ = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_train, y_train = dict(), dict()

    cv = KFold(n_splits=5, random_state=77, shuffle=True)

    i = 0
    for t, v in cv.split(X_train_):
      X_train['{}'.format(i + 1)] = X_train_.iloc[v]
      y_train['{}'.format(i + 1)] = y_train_.iloc[v]
      i += 1

    # --------------------------------------------------------------------------------------------
    # Final Stacking dataset 설정
    # --------------------------------------------------------------------------------------------
    # cv=5 갯수만큼 설정

    # train : y_true
    y_train1, y_train2, y_train3, y_train4, y_train5 = y_train['1'], y_train['2'], y_train['3'], y_train['4'], y_train['5']

    # shape 확인
    print('=' * 80)
    print(y_train1.shape, y_train2.shape, y_train3.shape, y_train4.shape, y_train5.shape)
    print('-' * 80)

    # train : y_pred
    x_train1, x_train2, x_train3, x_train4, x_train5 = [], [], [], [], []
    for model in list(final.keys()):
      x_train1.append(final[model]['val_pred'][0])
      x_train2.append(final[model]['val_pred'][1])
      x_train3.append(final[model]['val_pred'][2])
      x_train4.append(final[model]['val_pred'][3])
      x_train5.append(final[model]['val_pred'][4])

    # transpose
    X_train1 = pd.DataFrame(np.array(x_train1).T)
    X_train2 = pd.DataFrame(np.array(x_train2).T)
    X_train3 = pd.DataFrame(np.array(x_train3).T)
    X_train4 = pd.DataFrame(np.array(x_train4).T)
    X_train5 = pd.DataFrame(np.array(x_train5).T)

    # shape : (5661, n) - n : 각 seed의 모델 갯수
    print(X_train1.shape, X_train2.shape, X_train3.shape, X_train4.shape, X_train5.shape)
    print('-' * 80)

    # test : y_true
    y_test = y_test

    # test : y_pred
    x_test = []
    for model in list(final.keys()):
      x_test.append(final[model]['final_pred'][0])

    X_test = pd.DataFrame(np.array(x_test).T)
    X_test.columns = list(final.keys())

    # --------------------------------------------------------------------------------------------
    # Stacking dataset 설정
    # --------------------------------------------------------------------------------------------

    # train for stacking
    st_x_train = pd.concat([X_train1, X_train2, X_train3, X_train4, X_train5], axis=0)  # 예측값
    st_y_train = pd.concat([y_train1, y_train2, y_train3, y_train4, y_train5], axis=0)  # 실제값

    # test for stacking
    st_x_test = X_test  # 예측값
    st_y_test = y_test  # 실제값

    st_x_train.columns, st_x_test.columns = list(final.keys()), list(final.keys())

    # shape 확인
    print(st_x_train.shape, st_y_train.shape)
    print(st_x_test.shape, st_y_test.shape)
    print('-' * 80)

    return st_x_train, st_x_test, st_y_train, st_y_test



# Stacking
def stacking_coef(X, y, pred_dir, seed=117):
    # --------------------------------------------------------------------------------------------
    # Meta Train, Meta Test 값으로 최종 Stacking Model인 Linear Regression 수행
    # --------------------------------------------------------------------------------------------

    x_train, x_test, y_train, y_test = Stacking_df(X, y, pred_dir, seed)

    lr = LinearRegression()

    # seed 117
    lr.fit(x_train, y_train)
    coef = lr.coef_
    intercept = lr.intercept_

    # coef 확인
    print('coef : ', lr.coef_)
    print('intercept : ', lr.intercept_)
    print('=' * 80)

    return coef, intercept



# y inference
def seed_y_pred(train_X, train_y, test_X, pred_dir, model_dir,seed=117):
    # --------------------------------------------------------------------------------------------
    # Stacking Model에서 찾아낸 coef, intercept 값을 이용해 prediction y값 예측
    # --------------------------------------------------------------------------------------------

    coef, intercept = stacking_coef(train_X, train_y, pred_dir, seed)

    # Models
    tmp = os.listdir(model_dir)
    pkls = []
    for t in tmp:
      if str(seed) in t:
        filename, extension = os.path.splitext(t)
        pkls.append(filename)
    #pkls = glob.glob(os.path.join(model_dir, f'*_{seed}.pickle'))
    models = dict()

    # 학습시킨 Models 불러오기 : 경로 설정 주의

    for p in pkls : 
      with open(os.path.join(model_dir,'{}'.format(p)+".pkl"), 'rb') as f:
        models[p] = pkl.load(f)

    models = dict(sorted(models.items(), key = lambda x : x[0], reverse = False))
    
    # len(models) 확인
    print('* Seed : ', seed, '\t * Train Model len : ', len(models))
    print('=' * 80)

    # --------------------------------------------------------------------------------------------
    # 학습시킨 모델로 test_X 에 대한 값 예측
    # --------------------------------------------------------------------------------------------

    start_time = time.time()

    test_pred = dict()
    for model, regressor in models.items():
      y_pred = np.zeros(len(test_X))
      for reg in regressor:
        y_pred += np.expm1(reg.predict(test_X)) / 5
      test_pred[model] = y_pred

    print('Finish Prediction!')
    print("Working Time: {} seconds".format(time.time() - start_time))
    print('=' * 80)

    # --------------------------------------------------------------------------------------------
    # 각 모델에 대한 값을 예측해 dataframe을 만들고,
    # 해당 모델에 적합한 coef 값을 곱하고, intercept 값을 더하여 final_y 값 예측
    # --------------------------------------------------------------------------------------------

    pred_df = pd.DataFrame(test_pred)
    
    pred_df["y_pred"] = pred_df.multiply(coef.T).sum(axis=1) + intercept

    return pred_df["y_pred"]



def final_y_pred(train_X, train_y, test_X, pred_dir, model_dir, seedlist=[117,318,821,1009]):
    final_pred = np.zeros(len(test_X))

    # --------------------------------------------------------------------------------------------
    # 하나의 seed에 대해 overfitting 되는 것을 방지하기 위해,
    # 여러 seed에서 값을 예측하고, 각 seed에서 예측된 값들의 평균을 사용
    # --------------------------------------------------------------------------------------------

    for seed in seedlist:
      final_pred += seed_y_pred(train_X, train_y, test_X, pred_dir, model_dir, seed) / len(seedlist)

    return final_pred



def submission(train_X, train_y, test_X, test_idx, pred_dir, model_dir, result_dir, sub_dir, seedlist=[117,318,821,1009], mkfile=True):
    # --------------------------------------------------------------------------------------------
    # 최종 예측 값 dataframe 생성
    # --------------------------------------------------------------------------------------------

    # final predict
    y = final_y_pred(train_X, train_y, test_X, pred_dir, model_dir, seedlist = seedlist)
    y.index = test_idx

    # 취급액의 특성을 반영하기 위해, 백의자리 기준으로 반올림
    sub = np.round(y, -2)

    if test_idx.max() < 10000:
        
      data = pd.read_excel(
        os.path.join(result_dir, "2020 빅콘테스트 데이터분석분야-챔피언리그_2020년 6월 판매실적예측데이터(평가데이터).xlsx"),
          skiprows=1)
          
      data["취급액"] = sub
  
      if mkfile:
        data.to_excel(os.path.join(sub_dir, "쇼핑광고등어_2020년 6월 판매실적예측데이터(평가데이터).xlsx"), index=False)
    else:
        
      data = pd.read_excel(
        os.path.join('..', 'data','01_제공데이터', 'data4recommend.xlsx'))
            
      data["취급액"] = sub

      if mkfile:
        data.to_excel(os.path.join(sub_dir, "쇼핑광고등어_편성표후보상품취급액예측데이터.xlsx"), index=False)
        
    return data


def main():


    parser = argparse.ArgumentParser()

    parser.add_argument('--train_data_dir', type = str, default = os.path.join('..', 'data', '05_분석데이터', 'train_FE.pkl'))
    parser.add_argument('--test_data_dir', type = str, default = os.path.join('..', 'data', '05_분석데이터', 'test_FE.pkl'))
    parser.add_argument('--rec_data_dir', type = str, default = os.path.join('..', 'data', '05_분석데이터', 'Rec_FE.pkl'))
    parser.add_argument('--result_dir', type = str, default = os.path.join("..", "data", "02_평가데이터"))
    parser.add_argument('--pred_dir', type = str, default = os.path.join('preds'))
    parser.add_argument('--model_dir', type = str, default = os.path.join('models'))
    parser.add_argument('--sub_dir', type = str, default = os.path.join("..", "data", "04_임시데이터"))
    
    arg = parser.parse_args()
    #path load
    pred_dir = arg.pred_dir
    model_dir = arg.model_dir
    result_dir = arg.result_dir
    sub_dir = arg.sub_dir
    #data load
    train_data_dir = arg.train_data_dir
    test_data_dir = arg.test_data_dir
    rec_data_dir = arg.rec_data_dir

    train = joblib.load(train_data_dir)
    locals().update(train)

    # X, y 설정
    train_X = train['X'] ; train_y = train['y'] 
    train_log_y = np.log1p(train_y)


    # load test data
    test = joblib.load(os.path.join(test_data_dir))
    locals().update(test)

    # X, y 설정
    test_X = test['X'] ; test_idx = test['idx']


    #seed list
    seedlist = [117, 318, 821, 1009]
    submission(train_X, train_y, test_X, test_idx, pred_dir, model_dir, result_dir, sub_dir, seedlist)
    


if __name__ == '__main__':
    main()