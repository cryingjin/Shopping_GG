<<<<<<< HEAD
# 쇼핑광고등어_코드설명

# 쇼핑광고등어 코드설명 README 파일

- 재현가능한 코드들과 제출 파일들의 설명을 담고 있는 README 파일입니다.
- 데이터 전처리 및 분석과정 Raw 데이터 일체(Zip형식), 프로그램 코딩자료 제출가 모두 들어가있습니다.
- 코드 내용이 정리된 '마크다운'문서는 해당 문서이며, 실행결과 파일 또한 설명된 디렉토리에 HTML로 저장되어있습니다.

## 1. 파일 구조

```python
쇼핑광고등어
├── 쇼핑광고등어_코드설명.md
├── main.ipynb    
├── Preprocessing
│   ├───FE_main.py
│   ├───FE_make_corpus.py
│   ├───FE_innData.py
│   ├───FE_extData.py
│   ├───FE_NLP.py
│   │
├── data
│   ├───01_제공데이터
│   │    └───2020 빅콘테스트 데이터분석분야-챔피언리그_시청률 데이터.xlsx
│   │    └───2020 빅콘테스트 데이터분석분야-챔피언리그_방송편성표추천데이터
│   │    └───2020 빅콘테스트 데이터분석분야-챔피언리그_2019년 실적데이터_v1_200818
│   ├───02_평가데이터
│   │    └───2020 빅콘테스트 데이터분석분야-챔피언리그_2020년 6월 판매실적예측데이터(평가데이터)
│   ├───03_외부데이터
│   │    └───전처리
│   │          └─── ...
│   │    └───2020_dust
│   │          └─── ...
│   │    └───2019_dust
│   │          └─── ...
│   │    └───특일정보.xlsx
│   │    └───지역별 소비유형별 개인 신용카트.xlsx
│   │          ...
│   │ 
│   ├───04_임시데이터
│   │    └───data4time.pkl
│   │    └───rate44wnd.pkl
│   │    └───test_dataWnD.pkl
│   │    └───time4wnd.pkl
│   │    └───train_dataWnD.pkl
│   │    └───volume4wnd.pkl
│   └───05_분석데이터
│        └───train_Rec.pkl
│        └───train_FE.pkl
│        └───test_Rec.pkl
│        └───test_FE.pkl
│        └─── Rec_FE.pkl
│      
│   
├── DL_models
│    ├── DL_main.py
│    └── DL_test.py
│
│
├── ML_models
│   ├── ML_main.py
│   ├── ML_test.py
│   │
│   ├── models
│   │      ├── model_catBO_117.pkl
│   │      ├── model_lgbBO_117.pkl
│   │      ...
│   ├── params
│   │      ├── best_cb_BO.json
│   │      ├── best_cb_OP.json
│   │      ...
│   └─── preds
│          ├── pred_catBO_117.pkl
│          ├── pred_lgbBO_117.pkl
│          ...
└───Rec_models
```

## 2. 실행 결과 파일

데이터 전처리는 FE_main.py 실행으로 진행됩니다.

**취급액 예측 모델**은 ML_models 폴더에 정리되어 있으며, 최상위에 있는 `main.ipynb` 파일이 학습과 예측하는 과정을 보여줍니다. `main.ipynb` 파일은 HTML 형식으로 변환하여 제출합니다.

`main.ipynb` 실행 형식

```python
# 데이터 전처리는 FE_main.py --dataset=train 실행
!python FE_main.py --dataset=train
!python FE_main.py --dataset=test
!python FE_main.py --dataset=recommend

# ML_models 폴더에서 실행
cd ML_models

# ML_main.py 파일로 모델 학습
!python ML_main.py --epoch 30000

# ML_test.py 파일로 test 데이터 예측
!python ML_test.py --model_dir models --pred_dir preds
```

Deep learning model 의 코드 설명 및 실행결과 파일은 다음 경로에 다음과 같은 파일명으로 저장되어있습니다.

쇼핑광고등어 > DL_model 

`DL_train&test.html`

`DL_train&test.ipynb`

## 3. 최적화 편성표 모델

Rec_model 폴더를 참고하면 됩니다.

HTML 파일이 실행결과입니다.

##### [결과보고서](https://drive.google.com/file/d/11mZn7tsR0U7DvrMJ-D19wSfBRIyI9or5/view?usp=sharing)
=======
# Shopping_GG

## Data
|Data|type|Link|
|:---:|:---|:---|
|corpus_ours|list|[download link](https://drive.google.com/file/d/1SdiuAOdOgHCuuYHPKkGWq3M5W406B-2N/view?usp=sharing)|



## Model (.h5)
|model|Result(val loss)|Link|
|:---:|:---|:---|
|original version|total|-|
|3Net|total |-|
|3Net 0926|total|-|

## Model
|model|Result(val loss)|Link|
|:---:|:---|:---|
|3 multi input Net|total 71|-|
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

>>>>>>> origin/minjung

