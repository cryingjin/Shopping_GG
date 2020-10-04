import os
import sys
import json
import joblib
import time
import argparse
import numpy as np
import pandas as pd

from keras.metrics import top_k_categorical_accuracy
from keras.models import model_from_json

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required = True)
args = parser.parse_args()

data = joblib.load(os.path.join( '..', 'data', '05_분석데이터', args.dataset))
raw = pd.read_excel(os.path.join( '..', 'data', '01_제공데이터', '2020 빅콘테스트 데이터분석분야-챔피언리그_방송편성표추천데이터.xlsx'))
locals().update(data)

# 모델 load
with open('./model/model.json', 'r') as f:
    wnd_model = json.load(f)

wnd_model = model_from_json(wnd_model,
                    custom_objects = {'top_k_categorical_accuracy':top_k_categorical_accuracy}
                    )

wnd_model.load_weights('./model/model_weights.h5')

x_test_continue, x_test_category, x_test_category_poly=data4test[0], data4test[1], data4test[2]
test_input_data = [x_test_continue] + [x_test_category[:, i] for i in range(x_test_category.shape[1])] + [x_test_category_poly]

# 예측
answer = {
  '0' : '가구',
  '1':	'가전',
	'2':	'건강기능',
	'3':	'농수축',
	'4':	'생활용품',
	'5':	'속옷',
	'6':	'의류',
	'7':	'이미용',
	'8':	'잡화',
	'9':	'주방',
	'10':	'침구'
}

  
pred = wnd_model.predict(test_input_data)
pred_values = [list(pd.Series(p).sort_values(ascending = False).index[:3]) for p in pred]
values = np.array(list(answer[str(p)] for p in np.array(pred_values).flatten())).reshape(len(pred_values), 3)
recommend = pd.DataFrame(values, columns = ['추천상품군1', '추천상품군2', '추천상품군3'])

complete = pd.concat([raw, recommend], axis = 1)

complete.to_csv('./res/recommend1.csv', encoding = 'cp949')

print('Complete Recommend')

