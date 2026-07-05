Evaluating model: ft_transformer
Best Lambda for ft_transformer: 1.3200 (Val Acc: 0.9563)
                   Baseline        IC
Accuracy           0.949143  0.956286
Balanced Accuracy  0.949143  0.956286
Macro Precision    0.957805  0.959603
Macro Recall       0.949143  0.956286
Macro F1           0.950401  0.955931
Recall_BENIGN      0.982000  0.820000
Recall_Bot         0.834000  0.996000
Recall_BruteForce  0.966000  0.960000
Recall_DDoS        0.982000  0.984000
Recall_DoS         0.994000  0.968000
Recall_PortScan    0.950000  0.970000
Recall_WebAttack   0.936000  0.996000
Evaluation complete for ft_transformer. Outputs saved to results/ft_transformer/

Evaluating model: lstm
Best Lambda for lstm: 1.0000 (Val Acc: 0.9266)
                   Baseline        IC
Accuracy           0.919143  0.932571
Balanced Accuracy  0.919143  0.932571
Macro Precision    0.939052  0.935327
Macro Recall       0.919143  0.932571
Macro F1           0.922966  0.933103
Recall_BENIGN      0.978000  0.904000
Recall_Bot         0.834000  0.888000
Recall_BruteForce  0.964000  0.968000
Recall_DDoS        0.984000  1.000000
Recall_DoS         0.948000  0.976000
Recall_PortScan    0.830000  0.880000
Recall_WebAttack   0.896000  0.912000
Evaluation complete for lstm. Outputs saved to results/lstm/

Evaluating model: logistic
Best Lambda for logistic: 2.0200 (Val Acc: 0.8943)
                   Baseline        IC
Accuracy           0.868000  0.906571
Balanced Accuracy  0.868000  0.906571
Macro Precision    0.884612  0.920569
Macro Recall       0.868000  0.906571
Macro F1           0.865581  0.906072
Recall_BENIGN      0.962000  0.672000
Recall_Bot         0.512000  0.986000
Recall_BruteForce  0.972000  0.968000
Recall_DDoS        1.000000  0.986000
Recall_DoS         0.918000  0.908000
Recall_PortScan    0.820000  0.880000
Recall_WebAttack   0.892000  0.946000
Evaluation complete for logistic. Outputs saved to results/logistic/

Evaluating model: random_forest
Best Lambda for random_forest: 4.9900 (Val Acc: 0.9757)
                   Baseline        IC
Accuracy           0.968857  0.972000
Balanced Accuracy  0.968857  0.972000
Macro Precision    0.973188  0.973495
Macro Recall       0.968857  0.972000
Macro F1           0.969234  0.971800
Recall_BENIGN      0.998000  0.872000
Recall_Bot         0.862000  0.996000
Recall_BruteForce  0.996000  0.974000
Recall_DDoS        1.000000  1.000000
Recall_DoS         1.000000  0.978000
Recall_PortScan    0.972000  0.990000
Recall_WebAttack   0.954000  0.994000
Evaluation complete for random_forest. Outputs saved to results/random_forest/

Evaluating model: xgboost
Best Lambda for xgboost: 1.8500 (Val Acc: 0.9874)
                   Baseline        IC
Accuracy           0.985429  0.987143
Balanced Accuracy  0.985429  0.987143
Macro Precision    0.986144  0.987367
Macro Recall       0.985429  0.987143
Macro F1           0.985479  0.987095
Recall_BENIGN      0.996000  0.942000
Recall_Bot         0.946000  0.994000
Recall_BruteForce  1.000000  1.000000
Recall_DDoS        1.000000  1.000000
Recall_DoS         1.000000  0.986000
Recall_PortScan    0.968000  0.990000
Recall_WebAttack   0.988000  0.998000
Evaluation complete for xgboost. Outputs saved to results/xgboost/
