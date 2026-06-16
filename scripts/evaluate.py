import numpy as np
import pandas as pd
import json
import os
import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.imbalance_calibration import calibrate

def evaluate():
    pd_test = np.load("results/test_posteriors.npy")
    y_test = np.load("results/test_targets.npy")
    ps = np.load("results/source_prior.npy")
    
    with open("results/best_lambda.json", "r") as f:
        best_lambda = json.load(f)['best_lambda']
        
    num_classes = pd_test.shape[1]
    pt = np.ones(num_classes) / num_classes
    
    le = joblib.load("models/label_encoder.joblib")
    class_names = le.classes_
    
    # Baseline
    preds_baseline = np.argmax(pd_test, axis=1)
    
    # IC
    pf_test = calibrate(pd_test, ps, pt, best_lambda)
    preds_ic = np.argmax(pf_test, axis=1)
    
    def get_metrics(y_true, y_pred, name):
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        
        per_class_recall = precision_recall_fscore_support(y_true, y_pred, average=None)[1]
        
        metrics = {
            'Accuracy': acc,
            'Balanced Accuracy': bal_acc,
            'Macro Precision': precision,
            'Macro Recall': recall,
            'Macro F1': f1
        }
        
        for i, class_name in enumerate(class_names):
            metrics[f'Recall_{class_name}'] = per_class_recall[i]
            
        return metrics

    baseline_metrics = get_metrics(y_test, preds_baseline, "Baseline")
    ic_metrics = get_metrics(y_test, preds_ic, "IC")
    
    df_metrics = pd.DataFrame([baseline_metrics, ic_metrics], index=['Baseline', 'IC']).T
    df_metrics.to_csv("results/final_metrics_comparison.csv")
    print("\nFinal Metrics Comparison:")
    print(df_metrics)
    
    # Confusion Matrices
    def plot_cm(y_true, y_pred, title, filename):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
        plt.title(title)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(f"results/{filename}")
        plt.close()
        
    plot_cm(y_test, preds_baseline, "Baseline Confusion Matrix", "confusion_matrix_baseline.png")
    plot_cm(y_test, preds_ic, "IC Confusion Matrix", "confusion_matrix_ic.png")
    
    # Lambda Search Curve
    df_history = pd.read_csv("results/lambda_search_history.csv").sort_values('lambda')
    plt.figure(figsize=(8, 6))
    plt.plot(df_history['lambda'], df_history['accuracy'], marker='o')
    plt.axvline(best_lambda, color='r', linestyle='--', label=f'Best Lambda: {best_lambda:.4f}')
    plt.title("Lambda Search History (Validation Accuracy)")
    plt.xlabel("Lambda")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/lambda_search_curve.png")
    plt.close()
    
    print("\nEvaluation complete. Plots and metrics saved in results/")

if __name__ == "__main__":
    evaluate()
