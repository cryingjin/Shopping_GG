'''
ML_main.py

사용하는 모델의 def 파일, 
최적의 하이퍼파라미터로 튜닝한 모델을 사용한다.
학습 데이터를 Train 과 Test로 나눈 후 5개로 Cross validation 마다 
예측값(pred)와 학습된 model, shap valus 값을 저장하고 
prediction 값과 shap value 값들을 리턴한다
'''

import pandas as pd
import numpy as np
import joblib
import pickle as pkl
import argparse
import os
import json

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore')
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from sklearn.model_selection import KFold, cross_val_score, train_test_split


import lightgbm as lgb
from catboost import CatBoostRegressor

import shap



# ---------------------------------------------------------------------------------------------------
# MAPE
# ---------------------------------------------------------------------------------------------------


# MAPE
def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# MAPE_exp
def MAPE_exp(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((np.expm1(y_true) - np.expm1(y_pred)) / np.expm1(y_true))) * 100




# ---------------------------------------------------------------------------------------------------
# LightGBM
#
# <params>
# best_lgb_BO.json
# best_lgb_OP.json
#
# ---------------------------------------------------------------------------------------------------
def lgbm_pred(X, y, model_dir, pred_dir, params, version, seed=77, cv_splits=5, epoch=30000, shap_=False):
    pred = {'val_idx': [], 'val_pred': [],
            'test_idx': [], 'test_pred': [],
            'final_pred': []}  # final : test set mean

    SHAP = {'shap_values': [], 'expected_values': [],
            'shap_value': [], 'expected_value': []}

    # mape = {'val_mape': [], 'test_mape': [], 'final_mape': []}


    # models save
    models = []


    # train, test split
    X_train_, X_test_, y_train_, y_test_ = train_test_split(X, y, test_size=0.2, random_state=seed)
    pred['test_idx'].append(X_test_.index)


    # K Fold Cross Validation
    cv = KFold(n_splits=cv_splits, random_state=77, shuffle=True)
    for t, v in cv.split(X_train_):
        X_train, X_val = X_train_.iloc[t], X_train_.iloc[v]
        y_train, y_val = y_train_.iloc[t], y_train_.iloc[v]

        pred['val_idx'].append(v)

        # modeling
        train_T = lgb.Dataset(X_train.values, label=y_train.values)
        val_T = lgb.Dataset(X_val.values, label=y_val.values)

        model = lgb.train(params, train_T, epoch, valid_sets=val_T, verbose_eval=2500, early_stopping_rounds=500)
        models.append(model)

        val_pred = model.predict(X_val.values)
        pred['val_pred'].append(np.expm1(val_pred))

        test_pred = model.predict(X_test_.values)
        pred['test_pred'].append(np.expm1(test_pred))

        # mape
        # mape['val_mape'].append(MAPE_exp(y_val, val_pred))
        # mape['test_mape'].append(MAPE_exp(y_test_, test_pred))

        # SHAP
        if shap_:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_)

            SHAP['shap_values'].append(shap_values)
            SHAP['expected_values'].append(explainer.expected_value)
        
        # print
        print('-' * 80)
        print('finish CV ...')
        print('-' * 80)


    # final values
    final_test = np.mean(pred['test_pred'], axis=0)
    pred['final_pred'].append(final_test)

    # final mape
    # final_mape = MAPE(np.exp(y_test_), final_test)
    # mape['final_mape'].append(final_mape)


    if shap_:
        SHAP['shap_value'].append(np.mean(SHAP['shap_values'], axis=0))
        SHAP['expected_value'].append(np.mean(SHAP['expected_values']))


    # save models
    save_dir_model = os.path.join(model_dir,'model_' + version + '_' + str(seed) + '.pkl')
    with open(save_dir_model, 'wb') as f:
        pkl.dump(models, f, pkl.HIGHEST_PROTOCOL)

    # save preds
    save_dir_pred = os.path.join(pred_dir,'pred_' + version + '_' + str(seed) + '.pkl')
    with open(save_dir_pred, 'wb') as f:
        pkl.dump(pred, f, pkl.HIGHEST_PROTOCOL)

    # save shap values
    if shap_:
        with open('shap_' + version + '_' + str(seed) + '.pkl', 'wb') as f:
            pkl.dump(SHAP, f, pkl.HIGHEST_PROTOCOL)

    # save mapes
    # with open('lgbm_mape' + version + str(seed) + '.pkl', 'wb') as f:
    #     pkl.dump(mape, f, pkl.HIGHEST_PROTOCOL)

    # print
    print('-' * 80)
    print('Save Model Information!')
    print('-' * 80)

    return pred, SHAP


# ---------------------------------------------------------------------------------------------------
# CatBoost
#
# <params>
# best_cb_BO.json
# best_cb_OP.json
#
# ---------------------------------------------------------------------------------------------------
def cat_pred(X, y, model_dir, pred_dir, params, version, seed=77, cv_splits=5, epoch=30000, shap_=False):
    pred = {'val_idx': [], 'val_pred': [],
            'test_idx': [], 'test_pred': [],
            'final_pred': []}  # final : test set mean 값

    SHAP = {'shap_values': [], 'expected_values': [],
            'shap_value': [], 'expected_value': []}

    # mape = {'val_mape': [], 'test_mape': [], 'final_mape': []}


    # models save
    models = []


    # split train, test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    pred['test_idx'].append(X_test.index)


    cv = KFold(n_splits=cv_splits, random_state=77, shuffle=True)
    for t, v in cv.split(X_train):
        X_train_ , X_val_ = X_train.iloc[t] , X_train.iloc[v]
        y_train_ , y_val_ = y_train.iloc[t] , y_train.iloc[v]

        pred['val_idx'].append(v)


        # modeling
        model = CatBoostRegressor(**params)
        model.fit(X_train_, y_train_)
        models.append(model)  # save


        # Save model    # 0924 추가
        # joblib.dump(model, 'model_' + version + f'_{number}' + '.pkl')
        # number += 1  # save file count


        # predict
        y_pred = model.predict(X_val_)
        pred['val_pred'].append(np.expm1(y_pred))
        test_pred = model.predict(X_test)
        pred['test_pred'].append(np.expm1(test_pred))


        # mape
        # mape['val_mape'].append(MAPE_exp(y_val_, y_pred))
        # mape['test_mape'].append(MAPE_exp(y_test, test_pred))

        # 중간 확인 0925 추가
        # print('-' * 80)
        # print('중간 MAPE test_mape : ', np.mean(mape['test_mape'], axis=0))
        # print('중간 모델 MAPE test_mape : ', np.mean(MAPE_exp(y_test, test_pred)))
        # print('-' * 80)


        # SHAP
        if shap_ :
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

            SHAP['shap_values'].append(shap_values)
            SHAP['expected_value'].append(explainer.expected_value)

        # print
        print('-' * 80)
        print('finish CV ...')
        print('-' * 80)


    # final values
    final_test = np.mean(pred['test_pred'], axis=0)
    pred['final_pred'].append(final_test)

    # mape
    # final_mape = np.mean(mape['test_mape'], axis=0)
    # mape['final_mape'].append(final_mape)

    # save pkl
    # with open('mape_' + version + '.pkl', 'wb') as f:
    #     pkl.dump(mape, f, pkl.HIGHEST_PROTOCOL)


    # save models
    save_dir_model = os.path.join(model_dir,'model_' + version + '_' + str(seed) + '.pkl')
    with open(save_dir_model, 'wb') as f:
        pkl.dump(models, f, pkl.HIGHEST_PROTOCOL)

    # save preds
    save_dir_pred = os.path.join(pred_dir,'pred_' + version + '_' + str(seed) + '.pkl')
    with open(save_dir_pred, 'wb') as f:
        pkl.dump(pred, f, pkl.HIGHEST_PROTOCOL)



    if shap_ :
        with open('SHAP_' + version + '_' + str(seed) + '.pkl', 'wb') as f:
            pkl.dump(SHAP, f, pkl.HIGHEST_PROTOCOL)
    
    # print
    print('-' * 80)
    print('Save Model Information!' )
    print('-' * 80)

    return pred, SHAP



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type = str, default = os.path.join('..', 'data', '05_분석데이터', 'train_FE.pkl'))
    parser.add_argument('--model_dir', type = str, default = os.path.join('models'))
    parser.add_argument('--pred_dir', type = str, default = os.path.join('preds'))
    parser.add_argument('--epoch', type = int, default = 30000)
    parser.add_argument('--param_dir', type = str, default = os.path.join('params'))
    
    arg = parser.parse_args()
    
    epoch = arg.epoch

    #data load
    data_dir = arg.data_dir
    model_dir = arg.model_dir
    pred_dir = arg.pred_dir
    data = joblib.load(data_dir)
    locals().update(data)
    X = data['X']
    y = data['y']
    y2 = np.log(y)

    #parameter load
    param_dir = arg.param_dir
    tmplist = os.listdir(param_dir)
    param_dic = {} # parameter's dictionary
    param_name = [] #file name

    for i, tmp in enumerate(tmplist):
        filename,extension = os.path.splitext(tmp)
        if extension == '.json':
            param_name.append(filename)
            param_file =  os.path.join(param_dir, tmp)
            with open(param_file,'rb') as f:
                param_dic[filename] = json.load(f)

    """
    param_name = 
        best_lgb_BO
        best_lgb_OP
        best_cb_BO
        best_cb_OP
    """

    #seed list
    seeds = [117, 318, 821, 1009]
    versions = ['lgbBO','lgbOP','catBO','catOP']


    #train
    for i,param in enumerate(param_name):
        print(param_dic[param])
        if "lgb" in param :
            for seed in seeds:
                print("* model LightGBM : ", param, "\t * seed : ", seed)
                _, _ = lgbm_pred(X, y2, model_dir, pred_dir, param_dic[param], version = versions[i], seed=seed, cv_splits=5, epoch=epoch, shap_=False)
        elif "cb" in param :
            for seed in seeds:
                print("* model CatBoost : ", param, "\t * seed : ", seed)
                _, _ = cat_pred(X, y2, model_dir, pred_dir, param_dic[param], version = versions[i], seed=seed, cv_splits=5, epoch=epoch, shap_=False)



if __name__ == '__main__':
    main()
