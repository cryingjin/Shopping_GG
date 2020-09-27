# 2. Model 불러오기
from tensorflow.python.keras.models import load_model
import joblib
import numpy as np
import pandas as pd


def DataLoad_DL(data_dir,timeseries_list_dir):

    data = joblib.load(data_dir)
    locals().update(data)
    
    X = data['X']
    
    column_list =[]
    emb_list = ['v'+str(j) for j in range(0,110)]
    with open(timeseries_list_dir, 'rb') as f:
        time_ser = pkl.load(f)
    
    for col in X.columns:
        column_list.append(col)
    for i in emb_list:
        column_list.remove(i)

    num_list = column_list

    X_num = X[num_list]
    X_emb = X[emb_list]
    X_time = X_tmp[time_ser]
    X_time = X_time.fillna(0)
    X_num = X_num.fillna(0)


    return X_num, X_emb, X_time

def main():

    preds = {'val_preds' : [], 'test_preds' : []} 
    mape = {'val_mape' : [], 'test_mape' : []} 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./multiNet3.h5')
    parser.add_argument('--data_dir', type=str, default='./test_FE.pkl')
    arg = parser.parse_args()


    X, X_num, X_emb, X_time, y = DataLoad_DL(arg.data_dir, arg.timeS_dir)
    # 2. Model 불러오기

    model = load_model(arg.model_dir)

    result = model.predict([X_num, X_emb,X_time])

    result.to_csv("DL_result.csv")


if __name__ == '__main__':
    main()
 
