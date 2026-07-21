import numpy as np
import pandas as pd
import json
import os
import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.imbalance_calibration import calibrate

def search_lambda_grid(pd_val, y_val, ps):
    num_classes = pd_val.shape[1]
    pt = np.ones(num_classes) / num_classes
    
    # Robust grid search over lambda values from 0.0 to 3.0 with a step of 0.01.
    best_lambda = 0.0
    best_acc = 0.0
    history = []
    
    for lam in np.arange(0.0, 3.01, 0.01):
        lam = round(float(lam), 2)
        pf = calibrate(pd_val, ps, pt, lam)
        preds = np.argmax(pf, axis=1)
        acc = np.mean(preds == y_val)
        history.append({'lambda': lam, 'accuracy': acc})
        if acc > best_acc:
            best_acc = acc
            best_lambda = lam
            
    return best_lambda, best_acc, history

def main():
    model_name = "mlp"
    print(f"Evaluating {model_name} on Forest Cover Dataset using GRID SEARCH for Lambda...")
    
    # Load posteriors and targets
    pd_val = np.load("models/forest/mlp/validation_posteriors.npy")
    y_val = np.load("models/forest/mlp/validation_targets.npy")
    pd_test = np.load("models/forest/mlp/test_posteriors.npy")
    y_test = np.load("models/forest/mlp/test_targets.npy")
    ps = np.load("results/forest/source_prior.npy")
    
    num_classes = pd_test.shape[1]
    pt = np.ones(num_classes) / num_classes
    
    le = joblib.load("models/forest/label_encoder.joblib")
    classes = le.classes_
    
    # Run lambda search (using the grid search method)
    best_lambda, best_val_acc, lambda_history = search_lambda_grid(pd_val, y_val, ps)
    print(f"\nBest Lambda (Grid Search) for {model_name}: {best_lambda:.4f} (Val Acc: {best_val_acc:.4f})")
    
    # Save lambda search results
    output_dir = "results/forest/mlp"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "best_lambda_grid.json"), "w") as f:
        json.dump({'best_lambda': best_lambda, 'best_validation_accuracy': best_val_acc}, f)
    pd.DataFrame(lambda_history).to_csv(os.path.join(output_dir, "lambda_search_history_grid.csv"), index=False)
    
    # Baseline Predictions (standard softmax argmax)
    preds_baseline = np.argmax(pd_test, axis=1)
    
    # Calibrated Predictions
    pf_test = calibrate(pd_test, ps, pt, best_lambda)
    preds_ic = np.argmax(pf_test, axis=1)
    
    # Compute Metrics
    def get_metrics(y_true, y_pred):
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
        for i, name in enumerate(classes):
            metrics[f'Recall_{name}'] = per_class_recall[i]
        return metrics

    baseline_metrics = get_metrics(y_test, preds_baseline)
    ic_metrics = get_metrics(y_test, preds_ic)
    
    df_metrics = pd.DataFrame([baseline_metrics, ic_metrics], index=['Baseline', 'IC']).T
    df_metrics.to_csv(os.path.join(output_dir, "final_metrics_comparison_grid.csv"))
    print("\nFinal Metrics Comparison (Grid Search):")
    print(df_metrics.to_string())
    
    # Confusion Matrices
    def plot_cm(y_true, y_pred, title, filename):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
        plt.title(title)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        
    plot_cm(y_test, preds_baseline, f"Baseline Confusion Matrix ({model_name})", "confusion_matrix_baseline_grid.png")
    plot_cm(y_test, preds_ic, f"IC Confusion Matrix ({model_name} - Grid)", "confusion_matrix_ic_grid.png")
    
    # Plot Lambda Search Curve
    df_history = pd.DataFrame(lambda_history).sort_values('lambda')
    plt.figure(figsize=(8, 6))
    plt.plot(df_history['lambda'], df_history['accuracy'], marker='o')
    plt.axvline(best_lambda, color='r', linestyle='--', label=f'Best Lambda: {best_lambda:.4f}')
    plt.title(f"Lambda Search History (Grid Search) - {model_name}")
    plt.xlabel("Lambda")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "lambda_search_curve_grid.png"))
    plt.close()
    
    print(f"\nEvaluation complete (Grid Search). Outputs saved to {output_dir}/")

if __name__ == "__main__":
    main()
