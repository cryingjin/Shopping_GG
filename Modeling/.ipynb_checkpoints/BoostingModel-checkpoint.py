import pandas as pd 
import numpy as np

import joblib
import pickle as pkl

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore')
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler



# MAPE 
def MAPE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# MAPE_exp  
def MAPE_exp(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((np.exp(y_true) - np.exp(y_pred)) / np.exp(y_true))) * 100




# XGBoost
def xgb_model(X, y, params, version, cv_splits=5, scaling=False, epoch=20000):        
    mape = {'val_mape' : [], 'test_mape' : [], 'final_mape' : []}
    pred = {'val_idx'  : [], 'val_pred'  : [],
            'test_idx' : [], 'test_pred' : [],
            'final_pred' : []}      # final : test set mean 값
    

    # train, test split
    X_train_, X_test_, y_train_, y_test_ = train_test_split(X, y, test_size=0.2, random_state=77)
    pred['test_idx'].append(X_test_.index)

    
    # K Fold Cross Validation
    cv = KFold(n_splits=cv_splits, random_state=77, shuffle=True)
    for t,v in cv.split(X_train_):
        X_train , X_val = X_train_.iloc[t] , X_train_.iloc[v]            
        y_train , y_val = y_train_.iloc[t] , y_train_.iloc[v]

        pred['val_idx'].append(v)


        # scaling : MinMax or Standard 
        if scaling : 
            if scaling == 'MinMax' : 
                scaler = MinMaxScaler()
            elif scaling == 'Standard' : 
                scaler = StandardScaler() 

            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test_)

        else: 
            X_test = X_test_



        # modeling 
        train_T = xgb.DMatrix(X_train, label=y_train) 
        val_T   = xgb.DMatrix(X_val,   label=y_val)     

        watchlist = [(train_T, 'train'), (val_T, 'valid')]  
            

        model = xgb.train(params, train_T, epoch, watchlist, verbose_eval=2500, early_stopping_rounds=500)
        

        val_pred = model.predict(val_T)
        pred['val_pred'].append(np.exp(val_pred))

        test_T = xgb.DMatrix(X_test)
        test_pred = model.predict(test_T)
        pred['test_pred'].append(np.exp(test_pred))


        # mape
        mape['val_mape'].append(MAPE_exp(y_val, val_pred))
        mape['test_mape'].append(MAPE_exp(y_test_, test_pred))

        

    # final values
    final_test = np.mean(pred['test_pred'], axis=0)
    final_mape = MAPE(np.exp(y_test_), final_test)
    
    pred['final_pred'].append(final_test)
    mape['final_mape'].append(final_mape)

    
    # save pickle
    with open('xgb_pred.pickle' + version, 'wb') as f:
        pkl.dump(pred, f, pkl.HIGHEST_PROTOCOL)
    with open('xgb_mape.pickle' + version, 'wb') as f:
        pkl.dump(mape, f, pkl.HIGHEST_PROTOCOL)

        
    return mape, pred




# LightGBM
def lgbm_model(X, y, params, version, cv_splits=5, scaling=False, epoch=20000):        
    mape = {'val_mape' : [], 'test_mape' : [], 'final_mape' : []}
    pred = {'val_idx'  : [], 'val_pred'  : [],
            'test_idx' : [], 'test_pred' : [],
            'final_pred' : []}      # final : test set mean 값
    

    # train, test split
    X_train_, X_test_, y_train_, y_test_ = train_test_split(X, y, test_size=0.2, random_state=77)
    pred['test_idx'].append(X_test_.index)


    # K Fold Cross Validation
    cv = KFold(n_splits=cv_splits, random_state=77, shuffle=True)
    for t,v in cv.split(X_train_):
        X_train , X_val = X_train_.iloc[t] , X_train_.iloc[v]            
        y_train , y_val = y_train_.iloc[t] , y_train_.iloc[v]

        pred['val_idx'].append(v)


        # scaling : MinMax or Standard 
        if scaling : 
            if scaling == 'MinMax' : 
                scaler = MinMaxScaler()
            elif scaling == 'Standard' : 
                scaler = StandardScaler() 

            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test_)

        else : 
            X_train = X_train.values
            X_val = X_val.values
            X_test = X_test_.values

            

        # modeling 
        train_T = lgb.Dataset(X_train, label=y_train.values) 
        val_T   = lgb.Dataset(X_val, label=y_val.values)   


        model = lgb.train(params, train_T, epoch, valid_sets = val_T, verbose_eval=2500, early_stopping_rounds=500)


        val_pred = model.predict(X_val)
        pred['val_pred'].append(np.exp(val_pred))

        test_pred = model.predict(X_test)
        pred['test_pred'].append(np.exp(test_pred))


        # mape
        mape['val_mape'].append(MAPE_exp(y_val, val_pred))
        mape['test_mape'].append(MAPE_exp(y_test_, test_pred))

        

    # final values
    final_test = np.mean(pred['test_pred'], axis=0)
    final_mape = MAPE(np.exp(y_test_), final_test)
    
    pred['final_pred'].append(final_test)
    mape['final_mape'].append(final_mape)


    # save pickle
    with open('lgbm_pred.pickle' + version, 'wb') as f:
        pkl.dump(pred, f, pkl.HIGHEST_PROTOCOL)
    with open('lgbm_mape.pickle' + version, 'wb') as f:
        pkl.dump(mape, f, pkl.HIGHEST_PROTOCOL)

        
    '''
    # load
    with open('data.pickle', 'rb') as f:
    data = pkl.load(f)
    '''

    
    return mape, pred