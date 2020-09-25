# Shopping_GG

## Data
|Data|type|Link|
|:---:|:---|:---|
|corpus_ours|list|[download link](https://drive.google.com/file/d/1SdiuAOdOgHCuuYHPKkGWq3M5W406B-2N/view?usp=sharing)|

## Model
|model|Result(val loss)|Link|
|:---:|:---|:---|
|MLP&CNN-MLP|4|[link](https://github.com/cryingjin/Shopping_GG/blob/minjung/DLmodel/MLP_CNN_MLP(version1).ipynb)|
|LSTM&CNN-MLP| |[link](https://github.com/cryingjin/Shopping_GG/blob/minjung/DLmodel/MLP_CNN_MLP(version1).ipynb)|
|MultiLSTM&CNN-MLP|4|[link]([link](https://github.com/cryingjin/Shopping_GG/blob/minjung/DLmodel/MLP_CNN_MLP(version1).ipynb))|

## New Network (3_way)
![image](https://user-images.githubusercontent.com/41895063/94240577-6e394380-ff4e-11ea-82cb-5668d009d323.png)



## 0925

version2_origin
![image](https://user-images.githubusercontent.com/41895063/94230846-cd428c80-ff3d-11ea-8fed-dbb6c5df88cd.png)

![image](https://user-images.githubusercontent.com/41895063/94230780-a6845600-ff3d-11ea-93ce-d6e968ecc084.png)


## 0917
#### Feature 안줄인 ver batch_size = 128, epoch = 500 earlystopping = 20 val_loss = 129.04905279477438(망한거)
![image](https://user-images.githubusercontent.com/41895063/93476771-1dfe2800-f935-11ea-944d-b53a781d5bda.png)


## 09/17 ISSUE
1. Scaling안하고 해야겠다
2. MLP&CNN보다는 LSTM&CNN이 좋다

## 09/11 ISSUE
1. Colab에서는 메모리가 터짐 -> feature줄이고 층도 얕게 해서 다시 시도
2. 로컬에서는 loss가 nan. y_pred = 0으로 찍힘


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


