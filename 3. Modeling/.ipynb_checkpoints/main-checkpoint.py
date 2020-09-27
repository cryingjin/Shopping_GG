import os
import sys
import json
import joblib
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder, MinMaxScaler
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, SpatialDropout1D, Activation, concatenate, Conv2D, Conv1D, Lambda, GlobalMaxPooling1D
from keras.optimizers import Adam, SGD
from keras.layers.advanced_activations import ReLU, PReLU, LeakyReLU, ELU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from tensorflow.keras.utils import plot_model
from keras.metrics import top_k_categorical_accuracy, top_k_categorical_accuracy

import WideNDeep as WnD


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required = True)
args = parser.parse_args()

data = joblib.load(os.path.join('..', '..', '0.Data', '05_분석데이터', args.dataset))
locals().update(data)

COLUMNS = ['방송월',
                 '방송일',
                 '방송시간(시간)',
                 '경상지수',
                 '불변지수',
                 'pca_1',
                 'pca_2',
                 'pca_3',
                 'pca_4',
                 'pca_5',
                 '강수량(mm)_경기',
                 '강수량(mm)_광주',
                 '강수량(mm)_대구',
                 '강수량(mm)_대전',
                 '강수량(mm)_부산',
                 '강수량(mm)_서울',
                 '강수량(mm)_울산',
                 '강수량(mm)_인천',
                 '기온(°C)_경기',
                 '기온(°C)_광주',
                 '기온(°C)_대구',
                 '기온(°C)_대전',
                 '기온(°C)_부산',
                 '기온(°C)_서울',
                 '기온(°C)_울산',
                 '기온(°C)_인천',
                 '습도(%)_경기',
                 '습도(%)_광주',
                 '습도(%)_대구',
                 '습도(%)_대전',
                 '습도(%)_부산',
                 '습도(%)_서울',
                 '습도(%)_울산',
                 '습도(%)_인천',
                 '시정(10m)_경기',
                 '시정(10m)_광주',
                 '시정(10m)_대구',
                 '시정(10m)_대전',
                 '시정(10m)_부산',
                 '시정(10m)_서울',
                 '시정(10m)_울산',
                 '시정(10m)_인천',
                 '지면온도(°C)_경기',
                 '지면온도(°C)_광주',
                 '지면온도(°C)_대구',
                 '지면온도(°C)_대전',
                 '지면온도(°C)_부산',
                 '지면온도(°C)_서울',
                 '지면온도(°C)_울산',
                 '지면온도(°C)_인천',
                 '체감온도_경기',
                 '체감온도_광주',
                 '체감온도_대구',
                 '체감온도_대전',
                 '체감온도_부산',
                 '체감온도_서울',
                 '체감온도_울산',
                 '체감온도_인천',
                 '풍속(m/s)_경기',
                 '풍속(m/s)_광주',
                 '풍속(m/s)_대구',
                 '풍속(m/s)_대전',
                 '풍속(m/s)_부산',
                 '풍속(m/s)_서울',
                 '풍속(m/s)_울산',
                 '풍속(m/s)_인천',
                 '최고PM10_경기',
                 '최고PM10_광주',
                 '최고PM10_부산',
                 '최고PM10_서울',
                 '최고PM10_울산',
                 '최고PM10_인천',
                 '최고PM25_경기',
                 '최고PM25_광주',
                 '최고PM25_대구',
                 '최고PM25_대전',
                 '최고PM25_부산',
                 '최고PM25_서울',
                 '최고PM25_울산',
                 '최고PM25_인천',
                 '최저PM10_경기',
                 '최저PM10_광주',
                 '최저PM10_대구',
                 '최저PM10_대전',
                 '최저PM10_부산',
                 '최저PM10_서울',
                 '최저PM10_울산',
                 '최저PM10_인천',
                 '최저PM25_경기',
                 '최저PM25_광주',
                 '최저PM25_대구',
                 '최저PM25_대전',
                 '최저PM25_부산',
                 '최저PM25_서울',
                 '최저PM25_울산',
                 '최저PM25_인천',
                 '평균PM10_경기',
                 '평균PM10_광주',
                 '평균PM10_대구',
                 '평균PM10_대전',
                 '최고PM10_대구',
                 '최고PM10_대전',
                 '평균PM10_부산',
                 '평균PM10_서울',
                 '평균PM10_울산',
                 '평균PM10_인천',
                 '평균PM25_경기',
                 '평균PM25_광주',
                 '평균PM25_대구',
                 '평균PM25_대전',
                 '평균PM25_부산',
                 '평균PM25_서울',
                 '평균PM25_울산',
                 '평균PM25_인천',
                 'isHoliday',
                 '평일여부',
                 '방송시간대',
                 '계절',
                 '분기',
                 '일별평균시청률',
                 '일별시간별최대시청률',
                 '일별시간별평균시청률',
                 '일별시간별중간시청률',
                 '시간별월별최대시청률',
                 '시간별월별평균시청률',
                 '시간별월별중간시청률',
                 '월별시간별평균판매량',
                 '월별시간별중간판매량',
                 '월별시간별평균판매단가',
                 '월별시간별중간판매단가',
                 '시간별평균판매량',
                 '시간별중간판매량',
                 '시간별평균판매단가',
                 '시간별중간판매단가',
                 'count_가구',
                 'count_가전',
                 'count_건강기능',
                 'count_농수축',
                 'count_생활용품',
                 'count_속옷',
                 'count_의류',
                 'count_이미용',
                 'count_잡화',
                 'count_주방',
                 'count_침구',
                 'hour_가구',
                 'hour_가전',
                 'hour_건강기능',
                 'hour_농수축',
                 'hour_생활용품',
                 'hour_속옷',
                 'hour_의류',
                 'hour_이미용',
                 'hour_잡화',
                 'hour_주방',
                 'hour_침구',
                 'type1_0',
                 'type1_1',
                 'type1_2',
                 'type1_3',
                 'type1_4',
                 'type1_5',
                 'type1_6',
                 'type1_7',
                 'type1_8',
                 'type1_9',
                 'type1_10',
                 'type1_11',
                 'type1_12',
                 'type1_13',
                 'type1_14',
                 'type1_15',
                 'type1_16',
                 'type1_17',
                 'type1_18',
                 'type1_19',
                 'type1_20',
                 'type2_0',
         'type2_1',
         'type2_2',
         'type2_3',
         'type2_4',
         'type2_5',
         'type2_6',
         'type2_7',
         'type2_8',
         'type2_9',
         'type2_10',
         'type2_11',
         'type2_12',
         'type2_13',
         'type2_14',
         'type2_15',
         'type2_16',
         'type2_17',
         'type2_18',
         'type2_19',
         'type2_20',
         'type2_21',
         'type2_22',
         'type2_23',
         'type2_24',
         'type2_25',
         'type2_26',
         'type2_27',
         'type2_28',
         'type2_29',
         'type2_30',
         'type2_31',
         'type2_32',
         'type2_33',
         'type2_34',
         'type2_35',
         'type2_36',
         'type2_37',
         'type2_38',
         'type2_39',
         'type2_40',
         'type2_41',
         'type2_42',
         'type2_43',
                 'type3_0',
                 'type3_1',
                 'type3_2',
                 'type3_3',
                 'type3_4',
                 'type3_5',
                 'type3_6',
                 'type3_7',
                 'type3_8',
                 'type3_9',
                 'type3_10'
          ]
CATEGORICAL_COLUMNS = ['isHoliday', '평일여부', '방송시간대', '계절', '분기']
CONTINUOUS_COLUMNS = list(set(COLUMNS) - set(CATEGORICAL_COLUMNS))


x_train_continue, x_train_category, x_train_category_poly, train_label = data4train[0], data4train[1], data4train[2], data4train[3]
x_test_continue, x_test_category, x_test_category_poly, test_label = data4valid[0], data4valid[1], data4valid[2], data4valid[3]

category_inputs, continue_input, deep_model = WnD.Deep_model(X, COLUMNS, CATEGORICAL_COLUMNS, CONTINUOUS_COLUMNS)
wide_model = WnD.Wide_model(x_train_category_poly)

out_layer = concatenate([deep_model, wide_model])
inputs = [continue_input] + category_inputs + [wide_model]
output = Dense(len(train_label[0]), activation='softmax')(out_layer)
model = Model(inputs=inputs, outputs=output)


early_stopping = EarlyStopping(monitor = 'val_top_3_categorical_accuracy', patience = 30)

input_data = [x_train_continue] + [x_train_category[:, i] for i in range(x_train_category.shape[1])] + [x_train_category_poly]

epochs = 3000
optimizer = Adam(learning_rate=0.01)
batch_size = 64

def top_3_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

model.compile(optimizer = optimizer,
              loss = 'categorical_crossentropy',
              metrics = ['accuracy',  top_3_categorical_accuracy]
             )

model.fit(input_data,
          train_label,
          epochs = epochs,
          batch_size = batch_size,
          validation_split = 0.2,
          callbacks = [early_stopping]
         )


model.save('./model/model.h5')
model_json = model.to_json()

with open('./model/model.json', 'w') as f:
    json.dump(model_json, f)
model.save_weights('./model/model_weights.h5')

print('model saved!')        