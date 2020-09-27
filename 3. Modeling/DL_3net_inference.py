# 2. Model 불러오기
from tensorflow.python.keras.models import load_model
import joblib
import os
import numpy as np
import pickle as pkl
import pandas as pd
import argparse


def DataLoad_DL(data_dir,timeser_dir):

    data = joblib.load(data_dir)
    X = data['X']

    
    
    column_list =[]
    emb_list = ['v'+str(j) for j in range(0,110)]
    with open(timeser_dir, 'rb') as f:
        time_ser = pkl.load(f)
    
    for col in X.columns:
        column_list.append(col)

    for i in emb_list:
        column_list.remove(i)

    for j in time_ser:
        column_list.remove(j)

    num_list = column_list

    X_num = X[num_list]
    X_emb = X[emb_list]
    X_time = X[time_ser]
    X_time = X_time.fillna(0)
    X_num = X_num.fillna(0)


    return X_num, X_emb, X_time

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default= os.path.join('..', '..', '0.Data', '06_DL데이터', 'multiNet3_0927_noscaling_nology.h5'))
    parser.add_argument('--data_dir', type=str, default= os.path.join('..', '..', '0.Data', '05_분석데이터', 'test_FE.pkl'))
    parser.add_argument('--timeser_dir', type=str, default= os.path.join('..', '..', '0.Data', '06_DL데이터', 'timeseries_list.pkl'))
    arg = parser.parse_args()

    model_dir = arg.model_dir
    data_dir = arg.data_dir
    timeser_dir = arg.timeser_dir
    
    print("----------------Data Load-------------")
    X_num, X_emb, X_time = DataLoad_DL(data_dir,timeser_dir)
    print(X_num.shape, X_num.shape, X_time.shape)
    # 2. Model 불러오기
    print("----------------Model Load-------------")
    model = load_model(model_dir)

    print("----------------Predict-------------")
    result = model.predict([X_num, X_emb, X_time])
    result_df = pd.DataFrame(result)
    print(result)

    #result_df.to_csv("DL_result.csv")


if __name__ == '__main__':
    main()
 