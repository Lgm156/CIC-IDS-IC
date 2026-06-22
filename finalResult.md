# Evaluation: 

## MLP


```output
Final Metrics Comparison:
                   Baseline        IC
Accuracy           0.878286  0.963429
Balanced Accuracy  0.878286  0.963429
Macro Precision    0.922969  0.964435
Macro Recall       0.878286  0.963429
Macro F1           0.880483  0.963251
Recall_BENIGN      0.962000  0.882000
Recall_Bot         0.514000  0.990000
Recall_BruteForce  0.964000  0.968000
Recall_DDoS        1.000000  1.000000
Recall_DoS         0.994000  0.982000
Recall_PortScan    0.820000  0.948000
Recall_WebAttack   0.894000  0.974000

Evaluation complete. Plots and metrics saved in results/

## LSTM

Evaluating model: lstm
Best Lambda for lstm: 1.0400 (Val Acc: 0.9409)
                   Baseline        IC
Accuracy           0.914286  0.945429
Balanced Accuracy  0.914286  0.945429
Macro Precision    0.936577  0.950046
Macro Recall       0.914286  0.945429
Macro F1           0.917732  0.945874
Recall_BENIGN      0.982000  0.876000
Recall_Bot         0.834000  0.970000
Recall_BruteForce  0.968000  0.968000
Recall_DDoS        0.974000  1.000000
Recall_DoS         0.972000  0.994000
Recall_PortScan    0.774000  0.870000
Recall_WebAttack   0.896000  0.940000
Evaluation complete for lstm. Outputs saved to results/lstm/

## Logistic

Evaluating model: logistic
Best Lambda for logistic: 1.9000 (Val Acc: 0.8946)
                   Baseline        IC
Accuracy           0.731714  0.906000
Balanced Accuracy  0.731714  0.906000
Macro Precision    0.901648  0.918825
Macro Recall       0.731714  0.906000
Macro F1           0.716475  0.905260
Recall_BENIGN      0.988000  0.674000
Recall_Bot         0.510000  0.984000
Recall_BruteForce  0.972000  0.968000
Recall_DDoS        0.998000  0.984000
Recall_DoS         0.920000  0.906000
Recall_PortScan    0.692000  0.882000
Recall_WebAttack   0.042000  0.944000
Evaluation complete for logistic. Outputs saved to results/logistic/


## Random Forest: 

Evaluating model: random_forest
Best Lambda for random_forest: 10.0000 (Val Acc: 0.9766)
                   Baseline        IC
Accuracy           0.965429  0.972000
Balanced Accuracy  0.965429  0.972000
Macro Precision    0.970726  0.973135
Macro Recall       0.965429  0.972000
Macro F1           0.965888  0.971733
Recall_BENIGN      0.998000  0.874000
Recall_Bot         0.848000  0.996000
Recall_BruteForce  0.996000  0.974000
Recall_DDoS        1.000000  1.000000
Recall_DoS         1.000000  0.978000
Recall_PortScan    0.970000  0.988000
Recall_WebAttack   0.946000  0.994000
Evaluation complete for random_forest. Outputs saved to results/random_forest/


## XGBoost: 

Evaluating model: xgboost
Best Lambda for xgboost: 1.4900 (Val Acc: 0.9934)
                   Baseline        IC
Accuracy           0.982571  0.990286
Balanced Accuracy  0.982571  0.990286
Macro Precision    0.983704  0.990372
Macro Recall       0.982571  0.990286
Macro F1           0.982667  0.990269
Recall_BENIGN      0.996000  0.962000
Recall_Bot         0.932000  0.994000
Recall_BruteForce  0.998000  1.000000
Recall_DDoS        1.000000  1.000000
Recall_DoS         1.000000  0.990000
Recall_PortScan    0.968000  0.990000
Recall_WebAttack   0.984000  0.996000
Evaluation complete for xgboost. Outputs saved to results/xgboost/


## FT transformer: 


Evaluating model: ft_transformer
Best Lambda for ft_transformer: 1.6300 (Val Acc: 0.9546)
                   Baseline        IC
Accuracy           0.945143  0.960286
Balanced Accuracy  0.945143  0.960286
Macro Precision    0.956333  0.962839
Macro Recall       0.945143  0.960286
Macro F1           0.946778  0.959831
Recall_BENIGN      0.990000  0.832000
Recall_Bot         0.826000  0.996000
Recall_BruteForce  0.972000  0.970000
Recall_DDoS        0.998000  1.000000
Recall_DoS         0.990000  0.946000
Recall_PortScan    0.904000  0.988000
Recall_WebAttack   0.936000  0.990000
Evaluation complete for ft_transformer. Outputs saved to results/ft_transformer/
```
