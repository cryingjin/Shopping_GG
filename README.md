# Shopping_GG

## Data
|Data|type|Link|
|:---:|:---|:---|
|corpus_ours|list|[download link](https://drive.google.com/file/d/1SdiuAOdOgHCuuYHPKkGWq3M5W406B-2N/view?usp=sharing)|

## Model
|model|cv별 모델초기화|EPCOH|Link|
|:---:|:---|:---|:---|
|MLP&CNN-MLP|O|500|[link](https://github.com/cryingjin/Shopping_GG/blob/minjung/DLmodel/MLP_CNN_MLP(version1).ipynb)|
|LSTM&CNN-MLP|O|파일 어디갔지..?|learning...|
|LSTM&CNN-MLP|X|5000|learning...|
|MultiLSTM&CNN-MLP|O|5000|learning...|
|MultiLSTM&CNN-MLP|X|500|learning...|
|MultiLSTM&CNN-MLP|X|5000|learning...|


## 09/09 ISSUE
1. Epoch 5000 earlystopping 300으로 변경

## 09/07 ISSUE
1. embedding에는 linear, numeric data에는 relu, JR후에는 selu,relu 좋음
  numeric에 swish, embedding에 selu합치면 loss nan값
2. 학습중에 val loss는 2.7까지 내려가는데 MAPE 결과를 확인하면 90~100 왜그러지

## 09/03 ISSUE

1. 지금 있는 월별 CV 수정해야함 -> DL에서는 Weight를 저장하거나, pytorch로 짜야함
2. 모듈 전체를 캡슐화하여 GridSearch할 예정
3. Strucuured가 너무 많아서 embedding vector concate version써봐야할듯
4. JR한 vector는 lstm 통과보다 Dense통과가 좋다
5. AutoKeras는 넘모 오려걸리고 기본 Dense만 하고있네...흥


