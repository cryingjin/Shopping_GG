## Modeling ver. 
|Date|Model|Comment|Link|
|:---:|:---|:---|:---|
|08.24|xgb, lgbm|한달을 기준으로 cv|[0824_xgb+lgbm.ipynb](https://github.com/cryingjin/Shopping_GG/blob/jbeen2/Modeling/0824_xgb%2Blgbm_JB.ipynb)|
|08.27|xgb, lgbm|임베딩 피쳐 포함 ver1 (v129)|[0827_wordembedding_ver1_colab.ipynb](https://github.com/cryingjin/Shopping_GG/blob/jbeen2/Modeling/0827_wordembedding_ver1_colab.ipynb)|
|08.27|xgb, lgbm|임베딩 피쳐 포함 ver2 (v9)|[0827_wordembedding_ver2_colab.ipynb](https://github.com/cryingjin/Shopping_GG/blob/jbeen2/Modeling/0827_wordembedding_ver2_colab.ipynb)|
|08.27|xgb|optuna hyperparameter tuning 시도|[0827_optuna.ipynb](https://github.com/cryingjin/Shopping_GG/blob/jbeen2/Modeling/0827_optuna.ipynb)|
|09.01|lgbm|FE (파라미터 튜닝 노노)|[0901_lgbm_JB.ipynb](https://github.com/cryingjin/Shopping_GG/blob/jbeen2/Modeling/0901_lgbm_JB.ipynb)|
|09.03|lgbm + xgb|FE, 앙상블 쪼끔..|[0902_Ensemble.ipynb](https://github.com/cryingjin/Shopping_GG/blob/jbeen2/Modeling/0902_Ensemble.ipynb)|


.



## issue (0903) :sweat_drops:
   
   
:star2: 앙상블 했더니 성능 쪼끔 좋아졌엉 :exclamation: (그냥 두개 아웃풋 섞어 보기만 함... ㅎㅎ)          


:star2: n_estimators 겁나 늘렸더니 성능 쫌 좋아졋스 ~!!   


:heavy_check_mark: hyperparameter tuning 시도해보기 (optuna도 좋고 Grid Search도 좋고.. 다들 쓴 Bayesian Optimization 해보고 싶음 ㅠㅠ)          


:heavy_check_mark: MAPE 28 어케 찍지 :sweat::question: (젤 낮은 달 36까지는 찍어봤다..)               

