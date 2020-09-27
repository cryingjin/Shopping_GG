# 2. Model 불러오기
from tensorflow.python.keras.models import load_model
import joblib
import numpy as np
import pandas as pd
import argparse


def DataLoad_DL(data_dir):

    data = joblib.load(data_dir)
    X = data['X']
    
    column_list =[]
    emb_list = ['v'+str(j) for j in range(0,110)]
    
    for col in X.columns:
        column_list.append(col)
    for i in emb_list:
        column_list.remove(i)

    num_list = column_list

    X_num = X[num_list]
    X_emb = X[emb_list]
    X_num = X_num.fillna(0)


    return X_num, X_emb

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./DL_model.h5')
    parser.add_argument('--data_dir', type=str, default='./test_FE.pkl')
    arg = parser.parse_args()


    X_num, X_emb = DataLoad_DL(arg.data_dir)
    # 2. Model 불러오기

    model = load_model(arg.model_dir)

    result = model.predict([X_num, X_emb])
    result_df = pd.DataFrame(np.exp(result))
    print(result)

    result_df.to_csv("DL_result.csv")


if __name__ == '__main__':
    main()
 