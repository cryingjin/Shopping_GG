## Results :zap: 
|ver.|Hyperparameter|MAPE|
|:---:|:---|:---:|
|-|lgb_params1 = {'learning_rate': 0.02, 'max_depth' : 24, 'objective': 'regression', 'metric': 'mape', 'is_training_metric': True, 'lambda_l1': 0.7, 'min_child_samples' : 35, 'n_estimators' : 10000, 'num_leaves' : 30, 'bagging_fraction': 0.8}|[val] 33.3 [test] 32.8|
|-|lgbm_params2 = {'boost_from_average': False, 'learning_rate': 0.025, 'max_depth': 38, 'min_child_samples': 45, 'min_child_weight': 0.05, 'min_split_gain': 0.035, 'num_leaves': 12, 'reg_alpha': 0.02, 'reg_lambda': 0.7, 'subsample': 0.87, 'objective': 'regression',  'metric': 'mape', 'is_training_metric': True, 'n_estimators' : 10000} - 33.09|[val] 34.3 [test] 33.8|
|-|lgbm_params3 = {'num_leaves': 29, 'max_depth': 11, 'min_child_samples': 64, 'learning_rate': 0.02494389432079128, 'reg_alpha': 0.5519374108040493, 'min_split_gain': 0.006282136088690626, 'colsample_bytree': 0.7776910275713489, 'subsample': 0.7564838288884206, 'subsample_freq': 3, 'max_bin': 43, 'boosting': 'gbdt', 'objective': 'regression',  'metric': 'mape', 'is_training_metric': True, 'n_estimators' : 10000} - 33.37|[val] 33.9 [test] 33.3|
|0912_pred.pkl|lgbm_params4 = {'num_leaves': 47, 'max_depth': 8, 'min_child_samples': 39, 'learning_rate': 0.019268151551455108, 'reg_lambda': 0.2959823312094688, 'min_split_gain': 0.01022963196332564, 'colsample_bytree': 0.7957384348169647, 'subsample': 0.8124183802143509, 'subsample_freq': 5, 'max_bin': 56, 'boosting': 'dart', 'objective': 'regression',  'metric': 'mape', 'is_training_metric': True, 'n_estimators' : 10000} - 31.41|[val] 31.7 [test] 30.7|
|0915_pred.pkl|lgbm_params5 = {'colsample_bytree': 0.7, 'learning_rate': 0.005, 'max_bin': 70.0, 'max_depth': 8, 'min_child_samples': 30, 'min_split_gain': 0.005, 'num_leaves': 32, 'reg_lambda': 0.4, 'subsample': 0.85, 'subsample_freq': 4, 'boosting': 'dart', 'objective': 'regression',  'metric': 'mape', 'is_training_metric': True, 'n_estimators' : 25000} - 32.85|[val] 33.4 [test] 31.9|


#### 파일 저장  
```
final = {'lgbm_5th' : [lgbm_mape5, lgbm_pred5, lgbm_feat_imp5, lgbm_SHAP5]}  # xgb, lgbm 
```
.

#### 워너비 앙상블...? 
![KakaoTalk_20200915_191843481](https://user-images.githubusercontent.com/43749571/93202894-71824180-f78e-11ea-8981-fd41688abeea.jpg)



.



## Modeling ver. 
|Date|Model|Comment|Link|
|:---:|:---|:---|:---|
|08.24|xgb, lgbm|한달을 기준으로 cv|[0824_xgb+lgbm.ipynb](https://github.com/cryingjin/Shopping_GG/blob/jbeen2/Modeling/0824_xgb%2Blgbm_JB.ipynb)|
|08.27|xgb, lgbm|임베딩 피쳐 포함 ver1 (v129)|[0827_wordembedding_ver1_colab.ipynb](https://github.com/cryingjin/Shopping_GG/blob/jbeen2/Modeling/0827_wordembedding_ver1_colab.ipynb)|
|08.27|xgb, lgbm|임베딩 피쳐 포함 ver2 (v9)|[0827_wordembedding_ver2_colab.ipynb](https://github.com/cryingjin/Shopping_GG/blob/jbeen2/Modeling/0827_wordembedding_ver2_colab.ipynb)|
|08.27|xgb|optuna hyperparameter tuning 시도|[0827_optuna.ipynb](https://github.com/cryingjin/Shopping_GG/blob/jbeen2/Modeling/0827_optuna.ipynb)|
|09.01|lgbm|FE (파라미터 튜닝 노노)|[0901_lgbm_JB.ipynb](https://github.com/cryingjin/Shopping_GG/blob/jbeen2/Modeling/0901_lgbm_JB.ipynb)|
|09.03|lgbm + xgb|FE, 앙상블 쪼끔..|[0902_Ensemble.ipynb](https://github.com/cryingjin/Shopping_GG/blob/jbeen2/Modeling/0902_Ensemble.ipynb)|
|09.05|lgbm|TimeSeriesSplit CV + Bayesian Opt|[0905_ParamTuning.ipynb](https://github.com/cryingjin/Shopping_GG/blob/jbeen2/Modeling/0905_ParamTuning.ipynb)|
|09.07|lgbm + xgb|앙상블.. |[0906_Ensemble2.ipynb](https://github.com/cryingjin/Shopping_GG/blob/jbeen2/Modeling/0906_Ensemble2.ipynb)|
|09.07|lgbm|Time & Month CV|[0907_CV.ipynb](https://github.com/cryingjin/Shopping_GG/blob/jbeen2/Modeling/0907_CV.ipynb)|
|09.12|lgbm|CV5, optuna, SHAP|[0912_Tuning1.ipynb](https://github.com/cryingjin/Shopping_GG/blob/jbeen2/Modeling/0912_Tuning1.ipynb)|
|09.15|lgbm|CV5, Bayesian, SHAP|[0915_Tuning2.ipynb](https://github.com/cryingjin/Shopping_GG/blob/jbeen2/Modeling/0915_Tuning2.ipynb)|
|09.16|xgb|CV5, optuna, SHAP|[0916_Tuning3.ipynb](https://github.com/cryingjin/Shopping_GG/blob/jbeen2/Modeling/0916_Tuning3.ipynb)|




.



## issue (0917) :sweat_drops:
   
   
:star2: best result : 30.7 ~!!             


:star2: SHAP 시각화 했고 ([0915_Tuning2.ipynb](https://github.com/cryingjin/Shopping_GG/blob/jbeen2/Modeling/0915_Tuning2.ipynb)) 해석도 조금 달아 봤어염                 


:heavy_check_mark: 파라미터 조정해도 딱히 성능이 익스트림하게 좋아지지 않는다링 ..                    
                  

:heavy_check_mark: bayesian 찾았는데 성능 더 안조아져뜸... ㅎㅎㅎ 


:heavy_check_mark: SHAP value 뽑는 데에 너무 오래 걸려서... 일단 pred 값만 쭉쭉 뽑아서 다 저장해 둘게용 
             

