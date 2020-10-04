# 2. Model 불러오기
from tensorflow.python.keras.models import load_model
import joblib
import os
import numpy as np
import pickle as pkl
import pandas as pd
import argparse


def DataLoad_DL(data_dir,timeser_dir,scaler_dir):

    data = joblib.load(data_dir)
    X = data['X']
    idx = data["idx"]

    with open(scaler_dir, 'rb') as f:
        scaler = pkl.load(f)

    X_c = scaler.transform(X)
    X_tmp = pd.DataFrame(X_c, columns=X.columns, index=list(X.index.values))
    
    
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

    X_num = X_tmp[column_list]
    X_emb = X_tmp[emb_list]
    X_time = X_tmp[time_ser]
    X_time = X_time.fillna(0)
    X_num = X_num.fillna(0)
    """
    X_emb = np.asarray(X_emb).astype(np.float32)
    X_emb = np.reshape(X_emb,(X_emb.shape[0],X_emb.shape[1],1))
    X_num = np.asarray(X_num).astype(np.float32)
    X_num = np.reshape(X_num,(X_num.shape[0],X_num.shape[1],1))
    X_time = np.asarray(X_time).astype(np.float32)
    X_time = np.reshape(X_time,(X_time.shape[0],X_time.shape[1],1))
    """

    return idx, X_num, X_emb, X_time

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default= os.path.join( '..', 'data', '05_분석데이터', 'test_FE.pkl'))
    parser.add_argument('--model_dir', type=str, default= './DL_model.h5',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default=os.path.join( '..', 'data', '02_평가데이터', '2020 빅콘테스트 데이터분석분야-챔피언리그_2020년 6월 판매실적예측데이터(평가데이터).xlsx'),
                        help='Directory name to load the time series list')
    parser.add_argument('--timeS_dir', type=str, default=os.path.join( '..', 'data', '04_임시데이터', 'timeseries_list.pkl'),
                        help='Directory name to load the time series list')
    parser.add_argument('--scaler_dir', type=str, default= './scaler.pkl',
                        help='Directory name to load the time series list')
    arg = parser.parse_args()

    model_dir = arg.model_dir
    data_dir = arg.data_dir
    result_dir = arg.result_dir
    timeser_dir = arg.timeS_dir
    scaler_dir = arg.scaler_dir

    print("----------------Data Load-------------")
    result = pd.read_excel(result_dir,skiprows=1)
    idx, X_num, X_emb, X_time = DataLoad_DL(data_dir,timeser_dir,scaler_dir)
    print(X_num.shape, X_num.shape, X_time.shape)
    # 2. Model 불러오기
    print("----------------Model Load-------------")
    model = load_model(model_dir)

    print("----------------Predict-------------")
    pred = model.predict([X_num, X_emb, X_time])
    pred = pd.DataFrame(np.exp(pred))


    pred.index = idx
    #print(pred)

    print("----------------Result Save-------------")
    result["취급액"] = pred
    result.to_excel("DL_result.xlsx", index = False)
    print("Finish!")




if __name__ == '__main__':
    main()
 