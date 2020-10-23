# 쇼핑광고등어_코드설명


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
# 데이터 전처리는 FE_main.py 실행
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

## 최종보고서 :clipboard:
You can check our report [here](https://drive.google.com/file/d/11mZn7tsR0U7DvrMJ-D19wSfBRIyI9or5/view?usp=sharing)

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/minjung-s"><img src="https://user-images.githubusercontent.com/41895063/96987616-3974d800-155e-11eb-8dee-55ef4a423563.jpg" width="100px;" alt=""/><br /><sub><b>MinJung Shin</b></sub></a><br /><a href="https://github.com/cryingjin/Shopping_GG/commits?author=minjung-s" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/jinseock95"><img src="https://user-images.githubusercontent.com/41895063/96989102-2e22ac00-1560-11eb-84e4-0584c51bbe60.png" width="100px;" alt=""/><br /><sub><b>JinSeok Oh</b></sub></a><br /><a href="https://github.com/cryingjin/Shopping_GG/commits?author=jinseock95" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/cryingjin"><img src="https://user-images.githubusercontent.com/41895063/96989335-7fcb3680-1560-11eb-95d1-7cbb4fc4dd18.png" width="100px;" alt=""/><br /><sub><b>YeJin Lee</b></sub></a><br /><a href="https://github.com/cryingjin/Shopping_GG/commits?author=cryingjin" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/jbeen2"><img src="https://user-images.githubusercontent.com/41895063/96989477-b0ab6b80-1560-11eb-9beb-64f5253ccb2a.png" width="100px;" alt=""/><br /><sub><b>JaeBeen Lee</b></sub></a><br /><a href="https://github.com/cryingjin/Shopping_GG/commits?author=jbeen2" title="Code">💻</a></td>
